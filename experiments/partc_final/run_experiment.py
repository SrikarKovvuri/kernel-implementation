from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,math,os,shlex,statistics,subprocess,sys
out=Path(__file__).resolve().parent; assignment=out.parents[1]; root=assignment.parent
src=assignment/'src/partc'; tk=root/'tmp/cuda126-linux'
nvcc=next(tk.glob('cuda_nvcc-*-archive'))/'bin/nvcc'
rt=next(tk.glob('cuda_cudart-*-archive')); cccl=next(tk.glob('cuda_cccl-*-archive')); blas=next(tk.glob('libcublas-*-archive'))
flags=['--cudart=shared','-I'+str(rt/'include'),'-I'+str(cccl/'include'),'-I'+str(blas/'include'),'-L'+str(rt/'lib'),'-L'+str(blas/'lib')]
wrapper=out/'toolchain';wrapper.mkdir(exist_ok=True)
w=wrapper/'nvcc';w.write_text('#!/bin/sh\nexec '+shlex.join([str(nvcc)]+flags)+' "$@"\n');w.chmod(0o755)
env=os.environ.copy();env['PATH']=str(wrapper)+':'+env['PATH'];env['CUDA_MODULE_LOADING']='EAGER'
env['LD_LIBRARY_PATH']=str(rt/'lib')+':'+str(blas/'lib')+':'+env.get('LD_LIBRARY_PATH','')
def run(cmd,log,timeout=600):
    p=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
    (out/log).write_text(p.stdout)
    if p.returncode: raise RuntimeError(f'{log}: {p.returncode}\n{p.stdout[-4000:]}')
    return p.stdout
if '--grade' in sys.argv:
    print(run(['python3',str(assignment/'grade_local.py'),'--arch','sm_89'],'grade_local.log',900),flush=True)
    result=json.loads((assignment/'results.json').read_text())
    assert result['meta']['mode']=='gpu'
    for section in ['part_a','part_b','part_c']:
        for name,item in result[section].items():
            if isinstance(item,dict) and 'status' in item: assert item['status']=='ok',(section,name,item)
    assert all(result['part_c']['correctness'].values())
    print('ALL REQUIRED GPU GRADER CHECKS PASSED',flush=True);sys.exit(0)
spec=importlib.util.spec_from_file_location('grader',assignment/'grade_local.py');grader=importlib.util.module_from_spec(spec);spec.loader.exec_module(grader)
files=[src/n for n in ['c1_naive.cu','c2_tiled.cu','c3_cublas.cu','common.h','Makefile']]
meta={'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'correctness':[],'runs':[],
      'protocol':'CUDA_MODULE_LOADING=EAGER. One excluded fresh-process run per program, then five interleaved fresh-process default-size measurements. Events enclose the supplied pipeline. C3 includes first-GEMM overhead inside the timed region; handle creation is outside.'}
print(run(['make','-B','-C',str(src),'ARCH=sm_89'],'build.log'),flush=True)
programs=['c1_naive','c2_tiled','c3_cublas']
sizes=[(1,1),(8,4),(17,19),(33,65),(63,64),(129,4),(257,8),(129,31),(3,1024)]
refs={s:grader.reference_attention_output(*s) for s in sizes}
for name in programs:
    for N,d in sizes:
        stdout=run([str(src/name),str(N),str(d)],f'check_{name}_{N}_{d}.log')
        err=grader.max_abs_err(grader.parse_out_line(stdout),refs[N,d])
        assert err is not None and err<1e-6,(name,N,d,err)
        meta['correctness'].append({'program':name,'N':N,'d':d,'max_abs_error':err,'passed':True})
    print(name+': all nine CPU reference cases PASS',flush=True)
print(run([str(nvcc)]+flags+['-O3','-std=c++14','-arch=sm_89',str(out/'check_tiled.cu'),'-o',str(out/'check_tiled')],'check_build.log'),flush=True)
meta['extra_checks']=run([str(out/'check_tiled')],'check_tiled.log')
print(meta['extra_checks'],flush=True)
for trial in range(6):
    for name in programs:
        stdout=run([str(src/name)],f'{name}_{trial:02}.log')
        r=next(r for r in grader.parse_result_lines(stdout) if r['program']==name)
        row={'program':name,'trial':trial,'phase':'warmup' if trial==0 else 'measured','checksum':float(r['checksum']),'time_ms':float(r['time_ms']),'output':stdout.splitlines()[0]}
        assert math.isfinite(row['checksum']) and row['time_ms']>0
        meta['runs'].append(row);print(json.dumps(row),flush=True)
meta['summary']={}
for name in programs:
    rows=[r for r in meta['runs'] if r['program']==name and r['phase']=='measured'];times=[r['time_ms'] for r in rows]
    assert max(r['checksum'] for r in rows)-min(r['checksum'] for r in rows)<1e-5
    meta['summary'][name]={'checksum':rows[0]['checksum'],'median_ms':statistics.median(times),'mean_ms':statistics.mean(times),'min_ms':min(times),'max_ms':max(times),'times_ms':times}
assert max(v['checksum'] for v in meta['summary'].values())-min(v['checksum'] for v in meta['summary'].values())<1e-4
meta['environment']={'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),'cpu':next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'cuda':'12.6.85','cublas':'12.6.4.1','os':subprocess.check_output(['uname','-a'],text=True).strip()}
meta['sources_unchanged']=all(hashlib.sha256(p.read_bytes()).hexdigest()==meta['source_sha256'][p.name] for p in files);assert meta['sources_unchanged']
meta['finished_utc']=datetime.now(timezone.utc).isoformat();(out/'results.json').write_text(json.dumps(meta,indent=2)+'\n')
print('SUMMARY '+json.dumps(meta['summary']),flush=True)
