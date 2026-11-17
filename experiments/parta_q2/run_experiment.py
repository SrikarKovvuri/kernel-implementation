"""Compile unchanged vector-add programs and compare CUDA-event timings."""
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
source_paths = [src/f for f in ['vecadd.cu','vecaddKernel00.cu','vecaddKernel01.cu','vecaddKernel.h']]
source_paths += [assignment/'src/common'/f for f in ['timer.cu','timer.h']]
hashes = {str(p.relative_to(assignment)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
commands = {}
for variant in ['00','01']:
    name='vecadd'+variant
    command = [str(nvcc_dir/'bin/nvcc'),'-O3','-std=c++14','-arch=sm_89',
        '-I'+str(runtime/'include'),'-I'+str(cccl/'include'),
        '-I'+str(assignment/'src/common'),'-L'+str(runtime/'lib'),'--cudart=shared',
        str(src/'vecadd.cu'),str(src/f'vecaddKernel{variant}.cu'),str(assignment/'src/common/timer.cu'),
        '-o',str(out/name)]
    print('Compiling '+name,flush=True)
    build=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/f'build_{name}.log').write_text(build.stdout)
    if build.returncode: raise RuntimeError(build.stdout)
    commands[name]=command

def capture(cmd):
    return subprocess.check_output(cmd,env=env,text=True).strip()

meta={'started_utc':datetime.now(timezone.utc).isoformat(),'build_commands':commands,
      'source_sha256':hashes,'launch':{'blocks':240,'threads_per_block':512},
      'environment':{'gpu':capture(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader']),
        'nvcc':capture([str(nvcc_dir/'bin/nvcc'),'--version']),
        'g++':capture(['g++','--version']).splitlines()[0],
        'cpu':next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
        'os':capture(['uname','-a']), 'distribution':Path('/etc/os-release').read_text()},
      'protocol':'One warm-up process per program and size, excluded. Five measured rounds; shuffled size order and shuffled program order within each size, seed 6998. Each process runs the unchanged driver and one kernel launch. No trials discarded. Speedup is ratio of median CUDA-event times.',
      'runs':[]}
(out/'environment.json').write_text(json.dumps(meta,indent=2)+'\n')
rng=random.Random(6998)
schedule=[('warmup',0,v,name) for v in [500,1000,2000] for name in ['vecadd00','vecadd01']]
for trial in range(1,6):
    sizes=[500,1000,2000]; rng.shuffle(sizes)
    for v in sizes:
        names=['vecadd00','vecadd01']; rng.shuffle(names)
        schedule += [('measured',trial,v,name) for name in names]
with (out/'raw_runs.csv').open('w',newline='') as stream:
    fields=['phase','trial','program','values_per_thread','elements','kernel_ms','process_wall_ms','correct','exit_code']
    writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
    for index,(phase,trial,v,name) in enumerate(schedule):
        start=time.perf_counter()
        run=subprocess.run([str(out/name),str(v)],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
        wall_ms=(time.perf_counter()-start)*1000
        (out/f'run_{index:02d}_{phase}_{name}_{v}.log').write_text(run.stdout)
        line=next((l for l in run.stdout.splitlines() if l.startswith('RESULT,')),None)
        if line is None: raise RuntimeError(run.stdout)
        parsed=dict(part.split('=',1) for part in line.split(',')[1:])
        row={'phase':phase,'trial':trial,'program':name,'values_per_thread':v,
             'elements':int(parsed['N']),'kernel_ms':float(parsed['time_ms']),
             'process_wall_ms':round(wall_ms,3),'correct':int(parsed['correct']),'exit_code':run.returncode}
        assert row['elements']==240*512*v
        writer.writerow(row); stream.flush(); meta['runs'].append(row)
        print(json.dumps(row),flush=True)
        if run.returncode or row['correct']!=1: raise RuntimeError('Correctness failure; see raw log')
summary=[]
for v in [500,1000,2000]:
    result={'values_per_thread':v,'elements':240*512*v}
    for name in ['vecadd00','vecadd01']:
        rows=[r for r in meta['runs'] if r['phase']=='measured' and r['values_per_thread']==v and r['program']==name]
        times=[r['kernel_ms'] for r in rows]
        med=statistics.median(times)
        result[name]={'runs':len(rows),'mean_kernel_ms':statistics.mean(times),'median_kernel_ms':med,
            'stdev_kernel_ms':statistics.stdev(times),'min_kernel_ms':min(times),'max_kernel_ms':max(times),
            'mean_process_wall_ms':statistics.mean(r['process_wall_ms'] for r in rows),
            'effective_bandwidth_GBps_from_median':12*240*512*v/1e6/med}
    result['median_speedup']=result['vecadd00']['median_kernel_ms']/result['vecadd01']['median_kernel_ms']
    result['mean_speedup']=result['vecadd00']['mean_kernel_ms']/result['vecadd01']['mean_kernel_ms']
    summary.append(result)
meta['summary']=summary
meta['finished_utc']=datetime.now(timezone.utc).isoformat()
meta['sources_unchanged']=all(hashlib.sha256((assignment/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
assert meta['sources_unchanged']
(out/'results.json').write_text(json.dumps(meta,indent=2)+'\n')
print('SUMMARY '+json.dumps(summary),flush=True)
