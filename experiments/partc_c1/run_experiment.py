"""Review validation and repeatable C1 timings. Run with WSL Python 3."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.util, json, math, os, shlex, statistics, subprocess

out=Path(__file__).resolve().parent
assignment=out.parents[1]; root=assignment.parent; src=assignment/'src/partc'
tk=root/'tmp/cuda126-linux'
nvcc=next(tk.glob('cuda_nvcc-*-archive'))/'bin/nvcc'
rt=next(tk.glob('cuda_cudart-*-archive')); cccl=next(tk.glob('cuda_cccl-*-archive'))
env=os.environ.copy(); env['CUDA_MODULE_LOADING']='EAGER'
env['LD_LIBRARY_PATH']=str(rt/'lib')+':'+env.get('LD_LIBRARY_PATH','')
flags=['-O3','-std=c++14','-arch=sm_89','-I'+str(rt/'include'),'-I'+str(cccl/'include'),'-L'+str(rt/'lib'),'--cudart=shared']
files=[src/'c1_naive.cu',src/'common.h',src/'Makefile']
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
meta={'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':hashes,'CUDA_MODULE_LOADING':'EAGER',
      'protocol':'One discarded full-size warm-up process, then five fresh-process default-size runs. Each process measures its first three-kernel pipeline; no in-process data warm-up. CUDA events surround only the three launches.',
      'correctness':[],'timing_runs':[]}
def execute(cmd,log,timeout=300):
    p=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    (out/log).write_text(p.stdout+('\nSTDERR:\n'+p.stderr if p.stderr else ''))
    if p.returncode: raise RuntimeError(f'{log}: exit {p.returncode}\n{p.stderr}')
    return p.stdout
cmd=['make','-B','-C',str(src),'c1_naive','ARCH=sm_89','NVCC='+shlex.quote(str(nvcc)),'NVCCFLAGS='+shlex.join(flags)]
meta['build_command']=cmd
execute(cmd,'build.log')
print('C1 build passed',flush=True)
spec=importlib.util.spec_from_file_location('public_grader',assignment/'grade_local.py')
grader=importlib.util.module_from_spec(spec); spec.loader.exec_module(grader)
for N,d in [(1,1),(8,4),(16,16),(17,19),(33,65),(63,64),(129,4),(257,8),(3,1024)]:
    stdout=execute([str(src/'c1_naive'),str(N),str(d)],f'check_{N}_{d}.log')
    actual=grader.parse_out_line(stdout); expected=grader.reference_attention_output(N,d)
    err=grader.max_abs_err(actual,expected)
    ok=err is not None and err<=1e-6
    row={'N':N,'d':d,'elements':len(expected),'max_abs_error':err,'tolerance':1e-6,'passed':ok}
    meta['correctness'].append(row); print(json.dumps(row),flush=True)
    if not ok: raise RuntimeError('Element-wise comparison failed')
execute([str(nvcc)]+flags+[str(out/'check_stability.cu'),'-o',str(out/'check_stability')],'stability_build.log')
meta['stability']=execute([str(out/'check_stability')],'stability.log')
print(meta['stability'],flush=True)
for i in range(6):
    phase='warmup' if i==0 else 'measured'
    stdout=execute([str(src/'c1_naive')],f'{phase}_{i:02}.log')
    row=next(r for r in grader.parse_result_lines(stdout) if r['program']=='c1_naive')
    row={**row,'phase':phase,'run':i,'required_output':stdout.splitlines()[0]}
    assert int(row['N'])==2048 and int(row['d'])==64
    assert math.isfinite(float(row['checksum'])) and float(row['time_ms'])>0
    meta['timing_runs'].append(row); print(json.dumps(row),flush=True)
values=[float(r['time_ms']) for r in meta['timing_runs'] if r['phase']=='measured']
checksums=[float(r['checksum']) for r in meta['timing_runs'] if r['phase']=='measured']
assert max(checksums)==min(checksums)
meta['summary']={'N':2048,'d':64,'checksum':checksums[0],'median_ms':statistics.median(values),'mean_ms':statistics.mean(values),'min_ms':min(values),'max_ms':max(values),'runs':5}
meta['environment']={
    'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
    'nvcc':subprocess.check_output([str(nvcc),'--version'],text=True).strip(),
    'cpu':next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
    'os':subprocess.check_output(['uname','-a'],text=True).strip()}
meta['sources_unchanged']=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[p.name] for p in files)
assert meta['sources_unchanged']
meta['finished_utc']=datetime.now(timezone.utc).isoformat()
(out/'results.json').write_text(json.dumps(meta,indent=2)+'\n')
s=meta['summary']
report=f'''C1: Naive attention in CUDA - review and measured results

Required report line (checksum, median kernel milliseconds):
{s['checksum']:.3f},{s['median_ms']:.3f}

Default size: N=2048, d=64, double precision.
Hardware: {meta['environment']['gpu']}; {meta['environment']['cpu']}.
CUDA compiler: 12.6.85; build -O3 -std=c++14 -arch=sm_89; WSL2.

The scores kernel assigns one thread to each score and computes the scaled
dot product Q[i,:] dot K[j,:]. The full N x N double scores matrix occupies
32 MiB in device global memory at the default size. Softmax overwrites it
in place, subtracting each row's maximum before exponentiating and dividing
by the sum. The output kernel computes each element of P V with a dot product.
Scores and output use 16x16 blocks. Softmax uses 256 threads per block, with
one thread owning each row. At N=2048 this gives only eight softmax blocks;
the serial row scans and repeated exponentials are performance limitations
of this valid naive baseline, not correctness failures.

The existing driver sums all N*d output values for the checksum and prints
the required comma-separated line with three decimals. CUDA events enclose
all three kernel launches; synchronization on the stop event ensures they
finish before elapsed time is read. Allocation, input generation, explicit
copies, the CPU checksum, printing, and cleanup are outside this interval.

Protocol: CUDA_MODULE_LOADING=EAGER; one discarded full-size warm-up process,
then five fresh-process measurements. Each process measures its first pipeline;
this is not a warmed-data, repeatedly launched pipeline within one process.
Measured times (ms): {', '.join(f'{v:.3f}' for v in values)}.
Median: {s['median_ms']:.3f} ms; mean: {s['mean_ms']:.3f} ms;
range: {s['min_ms']:.3f}-{s['max_ms']:.3f} ms. All default runs printed the same checksum.

Correctness: all nine small/edge sizes passed the independent public-grader
CPU reference element by element at absolute tolerance 1e-6. They include
the grader's (8,4), (129,4), (257,8) cases plus singleton, unaligned shapes,
and d=1024. Printed tensors use nine significant digits, limiting the
precision of these comparisons. The full default tensor is not dumped by
the unchanged driver, so full-size element-wise validation is not claimed.
Additional direct-kernel tests passed with mixed-sign values, large logits,
softmax row-sum checks, finite-value checks, and output buffer guards.

The student's C1 source, common.h, and Makefile were not modified.
Raw build/output logs and exact source hashes are preserved alongside this
report in experiments/partc_c1. These are text logs, not screenshots; capture
the default output separately for the assignment's screenshot requirement.
AI-use disclosure: Codex reviewed the implementation, built and tested it,
collected timings, and drafted this C1 results text. The C1 kernels were
provided by the student and left unchanged.
'''
(out/'report.txt').write_text(report)
print('SUMMARY '+json.dumps(s),flush=True)
