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
out = assignment/'experiments/parta_q1'
out.mkdir(parents=True,exist_ok=True)
binary = out/'vecadd00'
env = os.environ.copy()
env['LD_LIBRARY_PATH'] = str(runtime/'lib')+':'+env.get('LD_LIBRARY_PATH','')
command = [str(nvcc_dir/'bin/nvcc'),'-O3','-std=c++14','-arch=sm_89',
           '-I'+str(runtime/'include'),'-I'+str(cccl/'include'),
           '-I'+str(assignment/'src/common'),'-L'+str(runtime/'lib'),'--cudart=shared',
           str(src/'vecadd.cu'),str(src/'vecaddKernel00.cu'),str(assignment/'src/common/timer.cu'),
           '-o',str(binary)]
print('Compiling unchanged driver, baseline kernel, and timer...',flush=True)
build=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(out/'build.log').write_text(build.stdout)
if build.returncode:
    print(build.stdout,flush=True)
    raise SystemExit(build.returncode)
def capture(cmd):
    return subprocess.check_output(cmd,env=env,text=True).strip()
source_paths=[src/'vecadd.cu',src/'vecaddKernel00.cu',src/'vecaddKernel.h',assignment/'src/common/timer.cu',assignment/'src/common/timer.h']
meta={'started_utc':datetime.now(timezone.utc).isoformat(),
      'build_command':command,'environment':{'gpu':capture(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader']),
      'nvcc':capture([str(nvcc_dir/'bin/nvcc'),'--version']),
      'g++':capture(['g++','--version']).splitlines()[0],
      'os':capture(['uname','-a'])},
      'source_sha256':{str(p.relative_to(assignment)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
      'launch':{'blocks':240,'threads_per_block':512},
      'protocol':'One separate warm-up process per size, then five measured rounds with each round shuffled using seed 6998. Unchanged program performs one kernel launch per process. Kernel time is CUDA event elapsed time; wall time is process elapsed time.',
      'runs':[]}
(out/'environment.json').write_text(json.dumps(meta,indent=2)+'\n')
rng=random.Random(6998)
schedule=[('warmup',0,v) for v in [500,1000,2000]]
for trial in range(1,6):
    order=[500,1000,2000]; rng.shuffle(order)
    schedule += [('measured',trial,v) for v in order]
with (out/'raw_runs.csv').open('w',newline='') as stream:
    fields=['phase','trial','values_per_thread','elements','kernel_ms','process_wall_ms','correct','exit_code']
    writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
    for index,(phase,trial,v) in enumerate(schedule):
        start=time.perf_counter()
        run=subprocess.run([str(binary),str(v)],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
        wall_ms=(time.perf_counter()-start)*1000
        (out/f'run_{index:02d}_{phase}_{v}.log').write_text(run.stdout)
        line=next((l for l in run.stdout.splitlines() if l.startswith('RESULT,')),None)
        if line is None: raise RuntimeError(run.stdout)
        parsed=dict(part.split('=',1) for part in line.split(',')[1:])
        row={'phase':phase,'trial':trial,'values_per_thread':v,'elements':int(parsed['N']),
             'kernel_ms':float(parsed['time_ms']),'process_wall_ms':round(wall_ms,3),
             'correct':int(parsed['correct']),'exit_code':run.returncode}
        writer.writerow(row); stream.flush(); meta['runs'].append(row)
        print(json.dumps(row),flush=True)
        if run.returncode or row['correct']!=1: raise RuntimeError('Correctness failure; see raw log')
summary=[]
for v in [500,1000,2000]:
    rows=[r for r in meta['runs'] if r['phase']=='measured' and r['values_per_thread']==v]
    times=[r['kernel_ms'] for r in rows]
    n=240*512*v
    summary.append({'values_per_thread':v,'elements':n,'runs':len(rows),
        'mean_kernel_ms':statistics.mean(times),'median_kernel_ms':statistics.median(times),
        'stdev_kernel_ms':statistics.stdev(times),'min_kernel_ms':min(times),'max_kernel_ms':max(times),
        'mean_process_wall_ms':statistics.mean(r['process_wall_ms'] for r in rows),
        'minimum_array_traffic_GB':n*12/1e9,'effective_bandwidth_GBps':n*12/1e6/statistics.mean(times)})
meta['summary']=summary
meta['finished_utc']=datetime.now(timezone.utc).isoformat()
meta['sources_unchanged']=all(hashlib.sha256((assignment/p).read_bytes()).hexdigest()==sha for p,sha in meta['source_sha256'].items())
(out/'results.json').write_text(json.dumps(meta,indent=2)+'\n')
print('SUMMARY '+json.dumps(summary),flush=True)
