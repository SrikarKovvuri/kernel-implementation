"""Fill the two C4 notebook functions; preserve the supplied correctness check."""
import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'src/partc/c4_triton.ipynb'
nb = json.loads(path.read_text())
nb['cells'] = [cell for cell in nb['cells'] if cell.get('id') != 'extra-validation']
def source(i, text):
    nb['cells'][i]['source'] = text.strip().splitlines(keepends=True)
    nb['cells'][i]['source'][-1] += '\n'

source(5, '''
@triton.jit
def attn_kernel(
    Q, K, V, O,
    N, scale,
    sqm, sqd, skn, skd, svn, svd, som, sod,
    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr, D_PAD: tl.constexpr,
):
    # One program owns BM query rows; keep only a BM x BN score tile.
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, D_PAD)
    keys = tl.arange(0, BN)
    q = tl.load(Q + rows[:, None] * sqm + cols[None, :] * sqd,
                (rows[:, None] < N) & (cols[None, :] < BD), other=0)
    m = tl.full((BM,), -float('inf'), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D_PAD), tl.float32)
    for start in range(0, N, BN):
        kn = start + keys
        k = tl.load(K + kn[None, :] * skn + cols[:, None] * skd,
                    (kn[None, :] < N) & (cols[:, None] < BD), other=0)
        # TF32x3 retains substantially more FP32 accuracy than a single TF32 dot.
        scores = tl.dot(q, k, input_precision='tf32x3') * scale
        scores = tl.where(kn[None, :] < N, scores, -float('inf'))
        m_new = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(scores - m_new[:, None])
        v = tl.load(V + kn[:, None] * svn + cols[None, :] * svd,
                    (kn[:, None] < N) & (cols[None, :] < BD), other=0)
        # Rescale BOTH accumulated numerator and denominator to the new max.
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision='tf32x3')
        denom = denom * alpha + tl.sum(p, axis=1)
        m = m_new
    out = acc / denom[:, None]
    tl.store(O + rows[:, None] * som + cols[None, :] * sod, out,
             (rows[:, None] < N) & (cols[None, :] < BD))
''')
old = ''.join(nb['cells'][7]['source'])
check = old[old.index('O_triton ='):]
source(7, '''
def my_attention(Q, K, V, scale, BM=64, BN=64):
    if Q.ndim != 2 or Q.shape != K.shape or Q.shape != V.shape:
        raise ValueError('Q, K, V must have matching (N, d) shapes')
    if any(t.dtype != torch.float32 or not t.is_cuda for t in (Q, K, V)):
        raise ValueError('Q, K, V must be CUDA float32 tensors')
    if Q.device != K.device or Q.device != V.device:
        raise ValueError('Q, K, V must be on the same CUDA device')
    n, head_dim = Q.shape
    if n <= 0 or head_dim <= 0:
        raise ValueError('N and d must be positive')
    if any(b < 16 or b & (b - 1) for b in (BM, BN)):
        raise ValueError('BM and BN must be powers of two >= 16')
    # Larger heads need smaller tiles to fit shared memory.
    if head_dim > 128:
        raise ValueError("This launcher supports head dimensions up to 128")
    stages = 2
    if head_dim > 64:
        BM, BN, stages = min(BM, 32), min(BN, 32), 1
    O = torch.empty_like(Q)
    with torch.cuda.device(Q.device):
        attn_kernel[(triton.cdiv(n, BM),)](
            Q, K, V, O, n, scale,
            *Q.stride(), *K.stride(), *V.stride(), *O.stride(),
            BM=BM, BN=BN, BD=head_dim, D_PAD=triton.next_power_of_2(max(head_dim, 16)),
            num_warps=4, num_stages=stages,
        )
    return O

''' + check)
src = ''.join(nb['cells'][1]['source']).split('# Keep the independent reference')[0].rstrip()
source(1, src + '''

# Keep the independent reference in FP32 rather than enabling TF32 matmul.
torch.backends.cuda.matmul.allow_tf32 = False
print('GPU:', torch.cuda.get_device_name())
print('PyTorch:', torch.__version__, '| Triton:', triton.__version__, '| CUDA:', torch.version.cuda)
''')
source(8, '''
## Benchmark against PyTorch SDPA

SDPA selects a backend according to the input dtype, shape, GPU, and installed
PyTorch version. We measure its default selection and inspect the dispatched
operator separately instead of assuming it uses FlashAttention for FP32.
Both functions use the same resident FP32 inputs. `do_bench` warms each function
and uses CUDA events; compilation and input generation are excluded.
''')
nb['cells'][0]['source'] = ''.join(nb['cells'][0]['source']).replace(
    'which dispatches to a fused\nFlashAttention backend.',
    'whose selected backend is recorded below.').splitlines(keepends=True)
extra = '''
# Additional checks: partial tiles, padded heads, non-contiguous inputs, and stability.
validation = []
gen = torch.Generator(device=device).manual_seed(123)
for n_test, d_test, strided, magnitude in [
    (1, 1, False, 1), (63, 32, False, 1), (65, 64, False, 1),
    (129, 19, False, 1), (257, 64, True, 1), (129, 128, False, 1),
    (129, 64, False, 10),
]:
    inputs = [torch.randn(n_test, d_test * (2 if strided else 1),
                           generator=gen, device=device) for _ in range(3)]
    if strided:
        inputs = [t[:, ::2] for t in inputs]
    qt, kt, vt = inputs
    qt = qt * magnitude if magnitude != 1 else qt
    kt = kt * magnitude if magnitude != 1 else kt
    st = d_test ** -0.5
    actual = my_attention(qt, kt, vt, st)
    expected = reference_attention(qt, kt, vt, st)
    error = (actual - expected).abs().max().item()
    assert torch.isfinite(actual).all() and error < 1e-2, (n_test, d_test, error)
    validation.append(dict(N=n_test, d=d_test, strided=strided,
                           magnitude=magnitude, max_abs_error=error))
    print('PASS', validation[-1])

# Inspect dispatch outside timing. SDPA is also checked against the same reference.
sdpa_error = (sdpa()[0, 0] - O_ref).abs().max().item()
assert sdpa_error < 1e-2
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    sdpa()
    torch.cuda.synchronize()
backend_ops = [event.key for event in prof.key_averages()
               if 'scaled_dot_product' in event.key]
print('SDPA operators:', backend_ops)
print('SDPA max abs error:', sdpa_error)

import json
results = dict(N=N, d=d, dtype=str(Q.dtype), BM=64, BN=64, num_warps=4, num_stages=stages,
               gpu=torch.cuda.get_device_name(), torch=torch.__version__,
               triton=triton.__version__, cuda=torch.version.cuda,
               checksum=O_triton.sum().item(), reference_checksum=O_ref.sum().item(),
               max_abs_error=err, triton_ms=ms_triton, sdpa_ms=ms_sdpa,
               sdpa_max_abs_error=sdpa_error, backend_ops=backend_ops, validation=validation)
print('C4_RESULT=' + json.dumps(results))
'''
nb['cells'].insert(10, dict(cell_type='code', id='extra-validation', metadata={},
    execution_count=None, outputs=[], source=extra.strip().splitlines(keepends=True)))
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
path.write_text(json.dumps(nb, indent=1) + '\n')
print(path)
