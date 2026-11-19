import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

out=Path(__file__).resolve().parent
assignment=out.parents[1]; root=assignment.parent; src=assignment/'src/partb'
tk=root/'tmp/cuda126-linux'
nvcc=next(tk.glob('cuda_nvcc-*-archive'))/'bin/nvcc'
rt=next(tk.glob('cuda_cudart-*-archive')); cccl=next(tk.glob('cuda_cccl-*-archive'))
env=os.environ.copy(); env['CUDA_MODULE_LOADING']='EAGER'
env['LD_LIBRARY_PATH']=str(rt/'lib')+':'+env.get('LD_LIBRARY_PATH','')
flags=['-O3','-std=c++14','-arch=sm_89','-I'+str(assignment/'src/common'),'-I'+str(rt/'include'),'-I'+str(cccl/'include'),'-L'+str(rt/'lib'),'--cudart=shared']
predictions=out/'predictions.txt'
assert predictions.exists()
meta={'started_utc':datetime.now(timezone.utc).isoformat(),'predictions_sha256':hashlib.sha256(predictions.read_bytes()).hexdigest(),
    'predictions_mtime_utc':datetime.fromtimestamp(predictions.stat().st_mtime,timezone.utc).isoformat(),
    'CUDA_MODULE_LOADING':'EAGER','managed_memory_policy':'Fresh allocations per scenario, initialized on CPU. First GPU use. No prefetch or data warm-up.',
    'measurement_protocol':'One complete measured sweep per supplied binary; all launches in each scenario enclosed by supplied timer. Explicit copies outside timer. Small correctness checks use separate processes and allocations.',
    'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir() if p.is_file() and p.suffix in ['.cu','.cpp','.py']},'runs':[]}
command=['make','-C',str(src),'ARCH=sm_89','NVCC='+shlex.quote(str(nvcc)),'NVCCFLAGS='+shlex.join(flags)]
meta['build_command']=command
print('Building Part B using the unchanged Makefile...',flush=True)
r=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(out/'build.log').write_text(r.stdout)
if r.returncode: raise RuntimeError(r.stdout)
for unified in [False,True]:
    name='check_unified' if unified else 'check_malloc'
    cmd=[str(nvcc)]+flags+(['-DTEST_UNIFIED'] if unified else [])+[str(out/'check_scenarios.cu'),str(assignment/'src/common/timer.cu'),'-o',str(out/name)]
    r=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/(name+'_build.log')).write_text(r.stdout)
    if r.returncode: raise RuntimeError(r.stdout)
    r=subprocess.run([str(out/name)],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    (out/(name+'.log')).write_text(r.stdout)
    if r.returncode: raise RuntimeError(r.stdout)
    print(name+': 15 element-wise and guard checks passed',flush=True)
    meta[name]=r.stdout
meta['environment']={
    'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
    'nvcc':subprocess.check_output([str(nvcc),'--version'],text=True).strip(),
    'cpu':next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
    'os':subprocess.check_output(['uname','-a'],text=True).strip()}
(out/'environment.json').write_text(json.dumps(meta,indent=2)+'\n')
rows=[]
for name in ['q1_cpu','q2_malloc','q3_unified']:
    cmd=[str(src/name),'1','5','10','50','100']
    print('START '+name+' full sweep',flush=True)
    start=time.monotonic()
    # Line-buffer only stdout; no kernel, input, or timing modifications.
    with (out/(name+'.log')).open('w') as logfile:
        proc=subprocess.Popen(['stdbuf','-oL']+cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=1)
        for line in proc.stdout:
            logfile.write(line); logfile.flush()
            print(name+': '+line.rstrip(),flush=True)
            if line.startswith('RESULT,'):
                row=dict(kv.split('=',1) for kv in line.strip().split(',')[1:])
                row['program']=name
                if row.get('correct')!='1':
                    proc.terminate(); proc.wait(); raise RuntimeError('Incorrect result: '+line)
                rows.append(row)
        code=proc.wait()
    if code: raise RuntimeError(f'{name} exited {code}')
    expected=5 if name=='q1_cpu' else 13
    assert sum(r['program']==name for r in rows)==expected
    meta['runs'].append({'program':name,'arguments':cmd[1:],'process_wall_seconds':time.monotonic()-start,'exit_code':code})
    (out/'progress.json').write_text(json.dumps({'runs':meta['runs'],'rows':rows},indent=2)+'\n')
    print('COMPLETE '+name,flush=True)
assert len(rows)==31
for row in rows:
    n=int(row['K'])*1000000
    assert int(row['n'])==n and float(row['checksum'])==3*n and float(row['time_ms'])>0
plots=assignment/'plots'; plots.mkdir(exist_ok=True)
for dest in [out/'partb_measurements.csv',plots/'partb_measurements.csv']:
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['program','scenario','K','n','time_ms','checksum','correct'],extrasaction='ignore'); w.writeheader(); w.writerows(rows)
meta['rows']=rows; meta['finished_utc']=datetime.now(timezone.utc).isoformat()
meta['sources_unchanged_during_run']=all(hashlib.sha256((src/n).read_bytes()).hexdigest()==h for n,h in meta['source_sha256'].items())
assert meta['sources_unchanged_during_run']
(out/'results.json').write_text(json.dumps(meta,indent=2)+'\n')
print('COMPLETE: all 31 required measurements valid',flush=True)
