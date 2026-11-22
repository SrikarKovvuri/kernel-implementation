"""Capture an actual headless-browser view of saved output, never a fake terminal."""
from pathlib import Path
import hashlib,html,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'experiments'; OUT=E/'report_evidence';OUT.mkdir(exist_ok=True)
chrome=Path('C:/Program Files/Google/Chrome/Application/chrome.exe')
profile=ROOT.parent/'tmp/chrome-report-evidence'
profile.mkdir(parents=True,exist_ok=True)
def capture(name,title,description,sections,height=900):
    parts=[]
    for label,path,content in sections:
        parts.append('<section><h2>'+html.escape(label)+'</h2><div class="source">'+html.escape(str(path.relative_to(ROOT)))+'</div><pre>'+html.escape(content)+'</pre></section>')
    document='''<!doctype html><meta charset="utf-8"><title>HPML recorded output</title>
    <style>body{margin:28px;background:#fff;color:#17232d;font-family:Arial,sans-serif}h1{font-size:27px;margin:0 0 10px}p{font-size:16px;line-height:1.4}h2{font-size:19px;margin:0 0 5px}.source{font-size:13px;color:#53616f;margin-bottom:9px}section{border:1px solid #b6c5ce;padding:15px;margin-top:16px;background:#f8fafc}pre{font:17px/1.5 Consolas,monospace;margin:0;white-space:pre-wrap;overflow-wrap:anywhere}.note{color:#4c5b66}</style>'''
    document+='<style>pre{font-size:20px;line-height:1.4}</style><h1>'+html.escape(title)+'</h1><p>'+html.escape(description)+'</p><p class="note">Screenshot of a local saved-log viewer. Output below is recorded program stdout, not a live terminal session.</p>'+''.join(parts)
    page=OUT/(name+'.html');page.write_text(document,encoding='utf-8')
    dest=OUT/(name+'.png')
    cmd=[str(chrome),'--headless','--disable-gpu','--no-first-run','--no-default-browser-check','--hide-scrollbars','--user-data-dir='+str(profile),'--window-size=1400,'+str(height),'--screenshot='+str(dest),page.as_uri()]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
    (OUT/(name+'_capture.log')).write_text(p.stdout+p.stderr)
    if p.returncode or not dest.exists():raise RuntimeError(p.stderr[-2000:])
    manifest.append({'name':name,'title':title,'image':str(dest),'html':str(page),'sources':[str(s[1]) for s in sections],'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
    print('Captured '+name,flush=True)
def recorded(folder,program,key,value,timing,prefix='run_'):
    for p in sorted((E/folder).glob(prefix+'*.log')):
        if 'warmup' in p.name or 'smoke' in p.name:continue
        text=p.read_text()
        for line in text.splitlines():
            if line.startswith('RESULT,'):
                row=dict(x.split('=',1) for x in line.split(',')[1:])
                if row.get('program')==program and int(row.get(key,-1))==value and abs(float(row['time_ms'])-timing)<0.0001:
                    return (program+' / '+str(value),p,text)
    raise ValueError((folder,program,key,value,timing))
manifest=[]
if '--smoke' in sys.argv:
    p=E/'partc_c1/measured_04.log';capture('capture_check','C1 - recorded output','N=2048, d=64. Checksum and time in milliseconds.',[('Measured run 4',p,p.read_text())],430)
elif '--c4-only' in sys.argv:
    manifest=json.loads((OUT/'manifest.json').read_text())
    manifest=[entry for entry in manifest if entry['name']!='c4_triton']
    p=E/'partc_c4/stdout.log'
    content='\n'.join(line for line in p.read_text().splitlines() if not line.startswith('C4_RESULT='))
    capture('c4_triton','C4 - Triton attention and PyTorch SDPA',
            'Executed notebook: N=2048, d=64, FP32. Correctness, timing, edge cases, and observed SDPA dispatch.',
            [('Recorded notebook stdout',p,content)],1400)
else:
    for q in [1,2,3,4]:
        data=json.loads((E/f'parta_q{q}'/('eager_results.json' if q==3 else 'results.json')).read_text())
        for row in data['summary']:
            if q<=2:
                n=row['values_per_thread'];progs=['vecadd00'] if q==1 else ['vecadd00','vecadd01'];key='values_per_thread'
            else:
                n=row['size'];progs=['matmult00'] if q==3 else ['matmult00','matmult01'];key='size'
            sections=[recorded(f'parta_q{q}',p,key,n,(row if q in [1,3] else row[p])['median_kernel_ms'],'eager_run_' if q==3 else 'run_') for p in progs]
            capture(f'a_q{q}_{n}',f'Part A Q{q} - {n}', 'Actual median observation(s) from the recorded five-run experiment. Times in milliseconds.',sections,650 if len(sections)==2 else 460)
    for k in [1,5,10,50,100]:
        p=E/f'partb/screenshots/K{k}_output.txt'
        capture(f'b_k{k}',f'Part B - K={k} million','Configuration annotations are derived from source; the stdout excerpts are unchanged.',[('CPU, explicit memory, and managed memory',p,p.read_text())],1000)
    data=json.loads((E/'partc_final/results.json').read_text())
    for prog,s in data['summary'].items():
        r=next(r for r in data['runs'] if r['program']==prog and r['phase']=='measured' and r['time_ms']==s['median_ms'])
        p=E/f'partc_final/{prog}_{r["trial"]:02}.log'
        capture(prog,'Part C - '+prog,'N=2048, d=64, double precision. Actual median observation from five fresh-process runs.',[('Recorded output',p,p.read_text())],430)
    warm=json.loads((E/'partc_final/warm_results.json').read_text())
    raw=E/'partc_final/warm_stdout.log'
    for prog,s in warm['summary'].items():
        content='\n'.join(line for line in raw.read_text().splitlines() if f'program={prog},' in line and 'phase=measured,' in line)
        content+='\n\nComputed summary (median of ten times above):\n'+f"{s['checksum']:.3f},{s['median_ms']:.3f}\n"
        capture('warm_'+prog,'Part C steady-state - '+prog,'Three same-process warm-ups excluded, followed by ten measured resident-data pipelines. CUDA events; full CPU reference checks.',[('Original measured BENCH lines + labeled computed summary',raw,content)],1100)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
