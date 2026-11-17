"""Compile the provided matmult00 and measure the three Q3 matrix sizes."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

root = Path(__file__).resolve().parents[3]
assignment = root/'HPML_HW3_starter'
src = assignment/'src/parta'
toolkit = root/'tmp/cuda126-linux'
nvcc_dir = next(toolkit.glob('cuda_nvcc-*-archive'))
runtime = next(toolkit.glob('cuda_cudart-*-archive'))
cccl = next(toolkit.glob('cuda_cccl-*-archive'))
out = Path(__file__).resolve().parent
env = os.environ.copy()
env['LD_LIBRARY_PATH'] = str(runtime/'lib')+':'+env.get('LD_LIBRARY_PATH','')
source_paths = [src/f for f in ['matmult.cu','matmultKernel00.cu','matmultKernel.h']]
source_paths += [assignment/'src/common'/f for f in ['timer.cu','timer.h']]
hashes = {str(p.relative_to(assignment)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
command = [str(nvcc_dir/'bin/nvcc'),'-O3','-std=c++14','-arch=sm_89','-DFOOTPRINT_SIZE=16',
    '-I'+str(runtime/'include'),'-I'+str(cccl/'include'),
    '-I'+str(assignment/'src/common'),'-L'+str(runtime/'lib'),'--cudart=shared',
    str(src/'matmult.cu'),str(src/'matmultKernel00.cu'),str(assignment/'src/common/timer.cu'),
    '-o',str(out/'matmult00')]
print('Compiling unchanged matmult00, FOOTPRINT_SIZE=16...',flush=True)
build=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(out/'build.log').write_text(build.stdout)
if build.returncode: raise RuntimeError(build.stdout)
def capture(cmd):
    return subprocess.check_output(cmd,env=env,text=True).strip()
meta={'started_utc':datetime.now(timezone.utc).isoformat(),'build_command':command,
      'source_sha256':hashes,'launch':{'block':[16,16],'footprint_size':16,'outputs_per_thread':1},
      'environment':{'gpu':capture(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader']),
        'nvcc':capture([str(nvcc_dir/'bin/nvcc'),'--version']),
        'g++':capture(['g++','--version']).splitlines()[0],
        'cpu':next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
        'os':capture(['uname','-a']),'distribution':Path('/etc/os-release').read_text()},
      'protocol':'One warm-up process per size, excluded. Five measured rounds; shuffled size order, seed 6998. CLI multipliers 16,32,64 produce matrix sizes 256,512,1024. Each new process runs the unchanged driver and one kernel launch. CUDA events time only the kernel. No trials discarded.',
      'runs':[]}
(out/'environment.json').write_text(json.dumps(meta,indent=2)+'\n')
rng=random.Random(6998)
schedule=[('warmup',0,n) for n in [256,512,1024]]
for trial in range(1,6):
    sizes=[256,512,1024]; rng.shuffle(sizes)
    schedule += [('measured',trial,n) for n in sizes]
with (out/'raw_runs.csv').open('w',newline='') as stream:
    fields=['phase','trial','argument','size','footprint','kernel_ms','process_wall_ms','correct','max_abs_err','exit_code']
    writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
    for index,(phase,trial,n) in enumerate(schedule):
        start=time.perf_counter()
        run=subprocess.run([str(out/'matmult00'),str(n//16)],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
        wall_ms=(time.perf_counter()-start)*1000
        (out/f'run_{index:02d}_{phase}_{n}.log').write_text(run.stdout)
        line=next((l for l in run.stdout.splitlines() if l.startswith('RESULT,')),None)
        if line is None: raise RuntimeError(run.stdout)
        parsed=dict(part.split('=',1) for part in line.split(',')[1:])
        row={'phase':phase,'trial':trial,'argument':n//16,'size':int(parsed['size']),
             'footprint':int(parsed['footprint']),'kernel_ms':float(parsed['time_ms']),
             'process_wall_ms':round(wall_ms,3),'correct':int(parsed['correct']),
             'max_abs_err':float(parsed['max_abs_err']),'exit_code':run.returncode}
        assert row['size']==n and row['footprint']==16
        writer.writerow(row); stream.flush(); meta['runs'].append(row)
        print(json.dumps(row),flush=True)
        if run.returncode or row['correct']!=1: raise RuntimeError('Correctness failure; see raw log')
summary=[]
for n in [256,512,1024]:
    rows=[r for r in meta['runs'] if r['phase']=='measured' and r['size']==n]
    times=[r['kernel_ms'] for r in rows]; med=statistics.median(times)
    summary.append({'size':n,'argument':n//16,'runs':len(rows),'grid':[n//16,n//16],
        'block':[16,16],'mean_kernel_ms':statistics.mean(times),'median_kernel_ms':med,
        'stdev_kernel_ms':statistics.stdev(times),'min_kernel_ms':min(times),'max_kernel_ms':max(times),
        'mean_process_wall_ms':statistics.mean(r['process_wall_ms'] for r in rows),
        'max_abs_err':max(r['max_abs_err'] for r in rows),'approx_flops':2*n**3,
        'GFLOPs_per_s_from_median':2*n**3/1e6/med})
meta['summary']=summary
meta['finished_utc']=datetime.now(timezone.utc).isoformat()
meta['sources_unchanged']=all(hashlib.sha256((assignment/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
assert meta['sources_unchanged']
(out/'results.json').write_text(json.dumps(meta,indent=2)+'\n')
print('SUMMARY '+json.dumps(summary),flush=True)
