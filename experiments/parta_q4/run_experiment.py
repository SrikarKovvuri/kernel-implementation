"""Compare the saved 16- and 32-footprint matmul kernels on equal matrix sizes."""
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
env['CUDA_MODULE_LOADING']='EAGER'
env['LD_LIBRARY_PATH']=str(runtime/'lib')+':'+env.get('LD_LIBRARY_PATH','')
footprints={'matmult00':16,'matmult01':32}
source_paths=[src/f for f in ['matmult.cu','matmultKernel00.cu','matmultKernel01.cu','matmultKernel.h']]
source_paths += [assignment/'src/common'/f for f in ['timer.cu','timer.h']]
hashes={str(p.relative_to(assignment)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
commands={}
for name,footprint in footprints.items():
    command=[str(nvcc_dir/'bin/nvcc'),'-O3','-std=c++14','-arch=sm_89',f'-DFOOTPRINT_SIZE={footprint}',
        '-I'+str(runtime/'include'),'-I'+str(cccl/'include'),'-I'+str(assignment/'src/common'),
        '-L'+str(runtime/'lib'),'--cudart=shared',str(src/'matmult.cu'),
        str(src/f'matmultKernel{name[-2:]}.cu'),str(assignment/'src/common/timer.cu'),'-o',str(out/name)]
    print('Compiling '+name,flush=True)
    build=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/f'build_{name}.log').write_text(build.stdout)
    if build.returncode: raise RuntimeError(build.stdout)
    commands[name]=command
def capture(cmd): return subprocess.check_output(cmd,env=env,text=True).strip()
meta={'started_utc':datetime.now(timezone.utc).isoformat(),'build_commands':commands,'source_sha256':hashes,
      'CUDA_MODULE_LOADING':'EAGER','footprints':footprints,'block':[16,16],
      'environment':{'gpu':capture(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader']),
        'nvcc':capture([str(nvcc_dir/'bin/nvcc'),'--version']),'g++':capture(['g++','--version']).splitlines()[0],
        'cpu':next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
        'os':capture(['uname','-a']),'distribution':Path('/etc/os-release').read_text()},
      'protocol':'Smoke tests at sides 32 and 96. One excluded warm-up per program/required size. Five measured rounds, shuffled size and program order (seed 6998). Both programs use EAGER loading and identical compiler flags except footprint. New process and one kernel launch per invocation; no trials discarded. Speedup is the ratio of median event times.',
      'runs':[]}
(out/'environment.json').write_text(json.dumps(meta,indent=2)+'\n')
schedule=[('smoke',0,n,name) for n in [32,96] for name in footprints]
schedule += [('warmup',0,n,name) for n in [256,512,1024] for name in footprints]
rng=random.Random(6998)
for t in range(1,6):
    sizes=[256,512,1024]; rng.shuffle(sizes)
    for n in sizes:
        names=list(footprints); rng.shuffle(names)
        schedule += [('measured',t,n,name) for name in names]
with (out/'raw_runs.csv').open('w',newline='') as stream:
    fields=['phase','trial','program','argument','size','footprint','kernel_ms','process_wall_ms','correct','max_abs_err','exit_code']
    writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
    for index,(phase,t,n,name) in enumerate(schedule):
        arg=n//footprints[name]
        start=time.perf_counter()
        run=subprocess.run([str(out/name),str(arg)],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
        wall=(time.perf_counter()-start)*1000
        (out/f'run_{index:02d}_{phase}_{name}_{n}.log').write_text(run.stdout)
        line=next((l for l in run.stdout.splitlines() if l.startswith('RESULT,')),None)
        if line is None: raise RuntimeError(run.stdout)
        p=dict(x.split('=',1) for x in line.split(',')[1:])
        row={'phase':phase,'trial':t,'program':name,'argument':arg,'size':int(p['size']),'footprint':int(p['footprint']),
             'kernel_ms':float(p['time_ms']),'process_wall_ms':round(wall,3),'correct':int(p['correct']),
             'max_abs_err':float(p['max_abs_err']),'exit_code':run.returncode}
        writer.writerow(row); stream.flush(); meta['runs'].append(row)
        print(json.dumps(row),flush=True)
        assert row['size']==n and row['footprint']==footprints[name]
        if row['correct']!=1 or row['exit_code']: raise RuntimeError('Correctness failure; see logs')
summary=[]
for n in [256,512,1024]:
    result={'size':n}
    for name,fp in footprints.items():
        rows=[r for r in meta['runs'] if r['phase']=='measured' and r['size']==n and r['program']==name]
        ts=[r['kernel_ms'] for r in rows]; med=statistics.median(ts)
        result[name]={'argument':n//fp,'runs':len(rows),'grid':[n//fp,n//fp],
            'mean_kernel_ms':statistics.mean(ts),'median_kernel_ms':med,'stdev_kernel_ms':statistics.stdev(ts),
            'min_kernel_ms':min(ts),'max_kernel_ms':max(ts),'max_abs_err':max(r['max_abs_err'] for r in rows),
            'mean_process_wall_ms':statistics.mean(r['process_wall_ms'] for r in rows),
            'GFLOPs_per_s_from_median':2*n**3/1e6/med}
    result['median_speedup']=result['matmult00']['median_kernel_ms']/result['matmult01']['median_kernel_ms']
    result['mean_speedup']=result['matmult00']['mean_kernel_ms']/result['matmult01']['mean_kernel_ms']
    summary.append(result)
meta['summary']=summary; meta['finished_utc']=datetime.now(timezone.utc).isoformat()
meta['sources_unchanged']=all(hashlib.sha256((assignment/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
assert meta['sources_unchanged']
(out/'results.json').write_text(json.dumps(meta,indent=2)+'\n')
print('SUMMARY '+json.dumps(summary),flush=True)
