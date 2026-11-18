import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

out=Path(__file__).resolve().parent
root=out.parents[2]
src=root/'HPML_HW3_starter/src/parta'
tk=root/'tmp/cuda126-linux'
nvcc=next(tk.glob('cuda_nvcc-*-archive'))/'bin/nvcc'
rt=next(tk.glob('cuda_cudart-*-archive'))
cccl=next(tk.glob('cuda_cccl-*-archive'))
blas=next(tk.glob('libcublas-*-archive'))
flags=['-O3','-std=c++14','-arch=sm_89','-I'+str(src),'-I'+str(rt/'include'),'-I'+str(cccl/'include'),'-I'+str(blas/'include')]
env=os.environ.copy(); env['CUDA_MODULE_LOADING']='EAGER'; env['NVIDIA_TF32_OVERRIDE']='0'
env['LD_LIBRARY_PATH']=str(rt/'lib')+':'+str(blas/'lib')+':'+env.get('LD_LIBRARY_PATH','')
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [src/'matmultKernel00.cu',src/'matmultKernel01.cu',src/'matmultKernel.h']}
commands=[]
for variant,fp in [('00',16),('01',32)]:
    renames={name:name+variant for name in ['GetElement','SetElement','GetSubMatrix']}
    renames['MatMulKernel']='MatMul'+variant
    commands.append([str(nvcc)]+flags+[f'-DFOOTPRINT_SIZE={fp}']+[f'-D{k}={v}' for k,v in renames.items()]+['-c',str(src/f'matmultKernel{variant}.cu'),'-o',str(out/f'kernel{variant}.o')])
commands.append([str(nvcc)]+flags+['-DFOOTPRINT_SIZE=16',str(out/'benchmark.cu'),str(out/'kernel00.o'),str(out/'kernel01.o'),
    '-L'+str(rt/'lib'),'-L'+str(blas/'lib'),'--cudart=shared','-lcublas','-o',str(out/'benchmark')])
for i,command in enumerate(commands):
    print('Build step',i+1,flush=True)
    r=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/f'build_{i}.log').write_text(r.stdout)
    if r.returncode: raise RuntimeError(r.stdout)
meta={'started_utc':datetime.now(timezone.utc).isoformat(),'build_commands':commands,'source_sha256':hashes,
    'protocol':'1024x1024 FP32; identical assignment inputs, CPU reference; 10 warm-ups per implementation in the same process, then 50 rounds in shuffled implementation order (seed 6998). CUDA events, no transfers/allocations in timing; cuBLAS SGEMM pedantic FP32; NVIDIA_TF32_OVERRIDE=0; CUDA_MODULE_LOADING=EAGER. Ratio of medians.',
    'environment':{'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
        'nvcc':subprocess.check_output([str(nvcc),'--version'],text=True).strip(),'os':subprocess.check_output(['uname','-a'],text=True).strip()}}
print('Running warmed-up FP32 benchmark...',flush=True)
r=subprocess.run([str(out/'benchmark')],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
(out/'raw_stdout.log').write_text(r.stdout)
if r.returncode: raise RuntimeError(r.stdout)
rows=[]; checks=[]
for line in r.stdout.splitlines():
    if line.startswith(('RESULT,','CHECK,','CONFIG,')):
        fields=dict(x.split('=',1) for x in line.split(',')[1:])
        if line.startswith('RESULT,'): rows.append({'program':fields['program'],'trial':int(fields['trial']),'kernel_ms':float(fields['time_ms'])})
        elif line.startswith('CHECK,'): checks.append(fields)
        else: meta['configuration']=fields
assert len(rows)==150 and len(checks)==6 and all(c['correct']=='1' for c in checks)
with (out/'raw_runs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['program','trial','kernel_ms']); w.writeheader(); w.writerows(rows)
summary={}
for name in ['matmult00','matmult01','cublas_sgemm']:
    ts=[r['kernel_ms'] for r in rows if r['program']==name]
    summary[name]={'runs':len(ts),'median_ms':statistics.median(ts),'mean_ms':statistics.mean(ts),'stdev_ms':statistics.stdev(ts),'min_ms':min(ts),'max_ms':max(ts)}
meta['summary']=summary; meta['checks']=checks
meta['speedup_over_baseline']=summary['matmult00']['median_ms']/summary['matmult01']['median_ms']
meta['custom_time_divided_by_cublas_time']=summary['matmult01']['median_ms']/summary['cublas_sgemm']['median_ms']
meta['sources_unchanged']=all(hashlib.sha256((src/n).read_bytes()).hexdigest()==h for n,h in hashes.items()); assert meta['sources_unchanged']
meta['finished_utc']=datetime.now(timezone.utc).isoformat()
(out/'results.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2),flush=True)
