from pathlib import Path
import hashlib,json,zipfile
import jsonschema
root=Path(__file__).resolve().parents[1]
result=json.loads((root/'results.json').read_text())
jsonschema.validate(result,json.loads((root/'results.schema.json').read_text()))
assert result['meta']['mode']=='gpu'
files=[]
for name in ['grade_local.py','results.json','results.schema.json','README.md','requirements.txt']:
    files.append(root/name)
for folder in ['src','tests','plots','experiments']:
    for p in (root/folder).rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ['__pycache__','.pytest_cache','toolchain']):continue
        if p.name=='Makefile' or p.suffix.lower() in {'.cu','.h','.cpp','.py','.ipynb','.png','.csv','.json','.log','.txt','.html','.md'}:
            files.append(p)
out=root.parent/'output';out.mkdir(exist_ok=True)
dest=out/'hw3_code.zip'
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(set(files)):
        assert p.read_bytes()[:4]!=b'\x7fELF',p
        z.write(p,p.relative_to(root).as_posix())
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
    names=set(z.namelist())
    required=['src/Makefile','src/parta/vecaddKernel01.cu','src/parta/matmultKernel01.cu','src/partb/q1_cpu.cpp','src/partb/q2_malloc.cu','src/partb/q3_unified.cu','src/partc/c1_naive.cu','src/partc/c2_tiled.cu','src/partc/c3_cublas.cu','src/partc/c4_triton.ipynb','results.json','plots/partb_with_unified.png','plots/partb_without_unified.png']
    assert all(p in names for p in required)
    assert not any(p.endswith(('.o','.exe','.dll','.so','.pyc')) for p in names)
manifest={'archive':str(dest),'files':len(files),'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'gpu_mode':result['meta']['mode'],'required_paths_verified':required,'report':str(root.parent/'output/pdf/report.pdf')}
(out/'submission_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
