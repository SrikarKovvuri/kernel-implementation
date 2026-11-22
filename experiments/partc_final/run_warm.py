from pathlib import Path
import json,os,subprocess,statistics,hashlib
out=Path(__file__).resolve().parent;root=out.parents[2];tk=root/'tmp/cuda126-linux'
nvcc=next(tk.glob('cuda_nvcc-*-archive'))/'bin/nvcc';rt=next(tk.glob('cuda_cudart-*-archive'));cccl=next(tk.glob('cuda_cccl-*-archive'));blas=next(tk.glob('libcublas-*-archive'))
env=os.environ.copy();env['CUDA_MODULE_LOADING']='EAGER';env['LD_LIBRARY_PATH']=str(rt/'lib')+':'+str(blas/'lib')+':'+env.get('LD_LIBRARY_PATH','')
flags=['-O3','-std=c++14','-arch=sm_89','--cudart=shared','-I'+str(rt/'include'),'-I'+str(cccl/'include'),'-I'+str(blas/'include'),'-L'+str(rt/'lib'),'-L'+str(blas/'lib')]
cmd=[str(nvcc)]+flags+[str(out/'benchmark_warm.cu'),'-lcublas','-o',str(out/'benchmark_warm')]
p=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'warm_build.log').write_text(p.stdout);assert p.returncode==0,p.stdout
p=subprocess.run([str(out/'benchmark_warm')],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300);(out/'warm_stdout.log').write_text(p.stdout);assert p.returncode==0,p.stdout
rows=[dict(x.split('=',1) for x in l.split(',')[1:]) for l in p.stdout.splitlines() if l.startswith('BENCH,')];assert len(rows)==39
summary={}
for name in ['c1_naive','c2_tiled','c3_cublas']:
    r=[r for r in rows if r['program']==name and r['phase']=='measured'];ts=[float(r['time_ms']) for r in r]
    summary[name]={'checksum':float(r[0]['checksum']),'median_ms':statistics.median(ts),'mean_ms':statistics.mean(ts),'min_ms':min(ts),'max_ms':max(ts),'times_ms':ts,'max_abs_error':max(float(x['max_abs_error']) for x in r)}
meta={'protocol':'Same-process resident-data execution. Three warm-up pipelines per variant excluded, then ten rounds in rotating variant order. Original kernels included without edits; identical launch configurations and cuBLAS operand mapping. CUDA events cover kernels/GEMMs only. CPU reference, copies, allocation and handle creation excluded. All 39 outputs checked element-wise against full CPU reference.', 'build_command':cmd,'runs':rows,'summary':summary}
(out/'warm_results.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(summary,indent=2))
