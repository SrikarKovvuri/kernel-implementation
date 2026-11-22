"""Execute C4 on a real GPU and preserve notebook output and provenance."""
import datetime, hashlib, json, os, subprocess, sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
path = ROOT / 'src/partc/c4_triton.ipynb'
kernel_dir = ROOT.parent / 'tmp/jupyter/kernels/hpml-c4'
kernel_dir.mkdir(parents=True, exist_ok=True)
(kernel_dir / 'kernel.json').write_text(json.dumps(dict(
    argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    display_name='HPML C4 (CUDA)', language='python')))
os.environ['JUPYTER_PATH'] = str(kernel_dir.parents[1])
os.environ['TRITON_CACHE_DIR'] = str(ROOT.parent / 'tmp/triton-c4-cache')
nb = nbformat.read(path, as_version=4)
source_hash = hashlib.sha256(''.join(c.source for c in nb.cells).encode()).hexdigest()
try:
    NotebookClient(nb, timeout=600, kernel_name='hpml-c4',
                   resources={'metadata': {'path': str(path.parent)}}).execute()
finally:
    nbformat.write(nb, path)
    output = '\n'.join(o.get('text', '') for c in nb.cells for o in c.get('outputs', [])
                       if o.output_type == 'stream')
    (HERE / 'stdout.log').write_text(output)
    print(output, flush=True)
line = next(x for x in output.splitlines() if x.startswith('C4_RESULT='))
results = json.loads(line.split('=', 1)[1])
results['timestamp_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
results['notebook_source_sha256'] = source_hash
results['nvidia_smi'] = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip()
results['python'] = sys.version
results['command'] = [sys.executable, str(Path(__file__).resolve())]
(HERE / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
(HERE / 'pip_freeze.txt').write_text(subprocess.check_output(
    [sys.executable, '-m', 'pip', 'freeze'], text=True))
assert all(c.execution_count is not None for c in nb.cells if c.cell_type == 'code')
assert not any(o.output_type == 'error' for c in nb.cells for o in c.get('outputs', []))
print('All notebook cells executed successfully.')
