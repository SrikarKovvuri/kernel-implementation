"""Check first-launch module-loading effects without altering the provided program."""
import csv
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

out=Path(__file__).resolve().parent
root=out.parents[2]
runtime=next((root/'tmp/cuda126-linux').glob('cuda_cudart-*-archive'))
env=os.environ.copy()
inherited=env.get('CUDA_MODULE_LOADING','unset')
env['CUDA_MODULE_LOADING']='EAGER'
env['LD_LIBRARY_PATH']=str(runtime/'lib')+':'+env.get('LD_LIBRARY_PATH','')
initial=json.loads((out/'results.json').read_text())
meta={'started_utc':datetime.now(timezone.utc).isoformat(),'environment':initial['environment'],
      'source_sha256':initial['source_sha256'],'build_command':initial['build_command'],
      'inherited_CUDA_MODULE_LOADING':inherited,'CUDA_MODULE_LOADING':'EAGER',
      'protocol':'Same existing executable. EAGER loading set only for child processes. One warm-up per size, then five measured rounds in size order shuffled with seed 6998. No trials discarded.', 'runs':[]}
schedule=[('warmup',0,n) for n in [256,512,1024]]
rng=random.Random(6998)
for t in range(1,6):
    sizes=[256,512,1024]; rng.shuffle(sizes)
    schedule += [('measured',t,n) for n in sizes]
with (out/'eager_raw_runs.csv').open('w',newline='') as stream:
    fields=['phase','trial','argument','size','footprint','kernel_ms','process_wall_ms','correct','max_abs_err','exit_code']
    writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
    for index,(phase,t,n) in enumerate(schedule):
        start=time.perf_counter()
        run=subprocess.run([str(out/'matmult00'),str(n//16)],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
        wall=(time.perf_counter()-start)*1000
        (out/f'eager_run_{index:02d}_{phase}_{n}.log').write_text(run.stdout)
        p=dict(x.split('=',1) for x in next(x for x in run.stdout.splitlines() if x.startswith('RESULT,')).split(',')[1:])
        row={'phase':phase,'trial':t,'argument':n//16,'size':int(p['size']),'footprint':int(p['footprint']),
             'kernel_ms':float(p['time_ms']),'process_wall_ms':round(wall,3),'correct':int(p['correct']),
             'max_abs_err':float(p['max_abs_err']),'exit_code':run.returncode}
        assert row['size']==n and row['footprint']==16 and row['correct']==1 and row['exit_code']==0
        writer.writerow(row); stream.flush(); meta['runs'].append(row)
        print(json.dumps(row),flush=True)
meta['summary']=[]
for n in [256,512,1024]:
    rr=[r for r in meta['runs'] if r['phase']=='measured' and r['size']==n]
    ts=[r['kernel_ms'] for r in rr]; med=statistics.median(ts)
    meta['summary'].append({'size':n,'argument':n//16,'runs':len(rr),'median_kernel_ms':med,'mean_kernel_ms':statistics.mean(ts),
        'stdev_kernel_ms':statistics.stdev(ts),'min_kernel_ms':min(ts),'max_kernel_ms':max(ts),
        'max_abs_err':max(r['max_abs_err'] for r in rr),'mean_process_wall_ms':statistics.mean(r['process_wall_ms'] for r in rr),
        'GFLOPs_per_s_from_median':2*n**3/1e6/med})
meta['finished_utc']=datetime.now(timezone.utc).isoformat()
(out/'eager_results.json').write_text(json.dumps(meta,indent=2)+'\n')
print('SUMMARY '+json.dumps(meta['summary']),flush=True)
