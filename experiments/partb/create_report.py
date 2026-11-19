"""Build the Part B report from measured results and genuine UI captures."""
from pathlib import Path
import json, sys, math
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = json.loads((HERE / 'results.json').read_text())
ROWS = DATA['rows']
KS = [1, 5, 10, 50, 100]
def time(prog, k, s=None):
    return float(next(r['time_ms'] for r in ROWS if r['program']==prog and int(r['K'])==k and (s is None or int(r['scenario'])==s)))

def evidence():
    folder = HERE / 'screenshots'
    folder.mkdir(exist_ok=True)
    for k in KS:
        n = k * 1000000
        lines = [f'PART B - saved measured output - K={k}, N={n:,}',
                 'Commands: each program ran with arguments 1 5 10 50 100.',
                 'Launch configuration (derived from source):']
        if k<=10:
            lines.append(f'S1: {n:,} launches; 1 block/launch; 1 thread/block.')
        else:
            lines.append('S1: skipped above K=10, as required by the handout.')
        lines += [f'S2: {(n+255)//256:,} launches; 1 block/launch; 256 threads/block.',
                  f'S3: 1 launch; {(n+255)//256:,} blocks; 256 threads/block.',
                  'Final-block writes are guarded by i < N.', '']
        for prog in ['q1_cpu','q2_malloc','q3_unified']:
            lines.append(prog + ' - original stdout lines:')
            lines += [line for line in (HERE/(prog+'.log')).read_text().splitlines() if line.startswith(f'K={k} ')]
            lines.append('')
        (folder / f'K{k}_output.txt').write_text('\n'.join(lines), encoding='utf-8')

if '--evidence' in sys.argv:
    evidence()
    print('Wrote five output excerpts with launch configuration; no measurements altered.')
    sys.exit(0)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Preformatted
from PIL import Image as PILImage
import pypdfium2 as pdfium
from pypdf import PdfReader

OUT = ROOT.parent / 'output/pdf'
OUT.mkdir(parents=True, exist_ok=True)
pdfpath = OUT / 'part_b_report.pdf'
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=9))
styles.add(ParagraphStyle(name='SmallCustom',fontName='Helvetica',fontSize=8.5,leading=11,spaceAfter=7))
styles['Title'].textColor=colors.HexColor('#173652')
styles['Heading2'].textColor=colors.HexColor('#173652')
story=[]
def p(text, small=False): story.append(Paragraph(text,styles['SmallCustom' if small else 'BodyCustom']))
def h(text): story.append(Paragraph(text,styles['Heading2']))
def page(): story.append(PageBreak())
def table(rows, widths):
    t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#173652')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTNAME',(0,1),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eef3f7'),colors.white]),('LINEBELOW',(0,-1),(-1,-1),0.5,colors.HexColor('#bac7d1')),('ALIGN',(1,1),(-1,-1),'RIGHT')]))
    story.append(t); story.append(Spacer(1,10))
def picture(path, width=504, maxheight=480):
    with PILImage.open(path) as im: w,h=im.size
    factor=min(width/w,maxheight/h)
    story.append(Image(str(path),width=w*factor,height=h*factor))

story.append(Paragraph('Part B: CUDA Unified Memory', styles['Title']))
p('HPML Homework 3 | Report section draft | Measurements collected November 18, 2026 (America/New_York).')
h('Experiment setup and timing')
p(f'The task computes z[i] = x[i] + y[i] for N = K x 1,000,000 single-precision elements. Inputs are x=1 and y=2. Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB), Intel Core Ultra 9 185H. Software: Ubuntu 24.04.1 under Windows WSL2, Linux kernel 5.15.153.1, NVIDIA driver {DATA["environment"]["gpu"].split(",")[1].strip()}, CUDA nvcc 12.6.85, and g++ 13.3. CPU code uses -O2; GPU code uses -O3 -std=c++14 -arch=sm_89. The supplied Makefile was used with toolchain path overrides.')
p('Each binary ran once with arguments 1 5 10 50 100. All reported times are single observations, not means. CUDA_MODULE_LOADING=EAGER was set consistently. The CPU driver times only addArrays with its supplied host timer. GPU timing uses the supplied CUDA start/stop events enclosing <b>all launches</b> of each scenario and synchronizes the stop event. This includes GPU timeline gaps while the host submits many tiny kernels; it is not a sum of isolated kernel durations.')
p('Q2 performs allocations and explicit host/device copies outside the event interval. Q3 uses fresh managed allocations for every scenario, initialized on the CPU, with no prefetch or data warm-up. Thus Q3 measures first GPU use. Allocation, input generation, checksum validation, and cleanup are outside the reported intervals. These are not end-to-end application speedups.')
h('Q4: predictions written before measurement')
p('The original predictions were saved in predictions.txt before the benchmark started; the runner recorded their timestamp and SHA-256. CPU time was expected to grow roughly linearly with N, with cache and first-write effects. S1 was predicted slowest because it submits N single-thread kernels. S2 was expected faster than S1 but still launch-bound. S3 was predicted fastest because one grid exposes all work to the GPU.')
p('On log-log axes, approximately linear growth should have slope near one. A fixed launch cost could flatten S3 at small sizes. The CPU was expected to beat S1 and S2; explicit-memory S3 was expected to beat the CPU at large N. Managed memory was expected to preserve the GPU ordering, with possible first-use residency overhead and no guaranteed fixed slowdown relative to Q2.')
h('Validation and scope')
p('All 31 required timing records passed the supplied checksum (3N). Separate tests exercised both GPU implementations at N=1, 255, 256, 257, and 513, in all three scenarios: all 30 cases passed element-by-element comparisons with varied inputs and guards before/after the output. This validates partial-block handling beyond aggregate checksums. The supplied kernels, drivers outside the permitted functions, Makefiles, timer, and plot script were retained.')

page(); h('Q1: CPU baseline')
p('addArrays performs a simple loop over N entries and writes x[i]+y[i] to z[i]. The supplied driver releases all host arrays with free().')
table([['K (millions)','Time (ms)','Checksum']]+[[str(k),f'{time("q1_cpu",k):.3f}',f'{3*k*1000000:,}'] for k in KS],[125,150,229])
p(f'The CPU time increases from {time("q1_cpu",1):.3f} to {time("q1_cpu",100):.3f} ms as N grows 100-fold ({time("q1_cpu",100)/time("q1_cpu",1):.1f}x time). The trend is increasing but not perfectly proportional: K=10 costs {time("q1_cpu",10)/time("q1_cpu",5):.2f}x K=5. Cache capacity, first writes to output pages, clock variation, and system activity are possible contributors; a single sweep does not identify their individual effects.')
h('Q2: explicit cudaMalloc / cudaMemcpy')
p('S1 launches one block of one thread for every element. S2 launches one block of 256 threads for each consecutive chunk. S3 launches one grid of ceil(N/256) blocks of 256 threads. The given kernel checks i&lt;N. For K=1, 5, 10, and 50 the last block is partial; K=100 divides evenly by 256. The supplied driver copies inputs before timing, copies output afterward, and frees host/device allocations.')
table([['K','S1 time (ms)','S2 time (ms)','S3 time (ms)']]+[[str(k),f'{time("q2_malloc",k,1):.3f}' if k<=10 else 'Not required',f'{time("q2_malloc",k,2):.3f}',f'{time("q2_malloc",k,3):.3f}'] for k in KS],[55,155,147,147])
p(f'At K=10, S1 takes {time("q2_malloc",10,1)/1000:.3f} seconds for ten million launches, while S3 takes {time("q2_malloc",10,3):.3f} ms for one launch: a {time("q2_malloc",10,1)/time("q2_malloc",10,3):,.0f}x ratio. S2 still takes {time("q2_malloc",10,2):.3f} ms because 39,063 separate launches repeatedly expose just one block. S3 can schedule many blocks across the GPU and amortize submission overhead.')
p(f'At K=100, the CPU/S3 timing ratio is {time("q1_cpu",100)/time("q2_malloc",100,3):.2f}x ({time("q1_cpu",100):.3f}/{time("q2_malloc",100,3):.3f}). For the same size, S2 takes {time("q2_malloc",100,2)/time("q1_cpu",100):.2f}x as long as the CPU. Merely moving this operation to CUDA is insufficient: the launch pattern determines whether the parallel hardware is used effectively.')

page(); h('Q3: cudaMallocManaged')
p('The same launch patterns operate directly on pointers returned by cudaMallocManaged. No cudaMemcpy calls or prefetch passes were added. The driver waits for GPU completion before the CPU checks z and releases the arrays with cudaFree().')
table([['K','S1 time (ms)','S2 time (ms)','S3 time (ms)']]+[[str(k),f'{time("q3_unified",k,1):.3f}' if k<=10 else 'Not required',f'{time("q3_unified",k,2):.3f}',f'{time("q3_unified",k,3):.3f}'] for k in KS],[55,155,147,147])
p(f'At K=100, managed-memory S3 takes {time("q3_unified",100,3):.3f} ms, versus {time("q2_malloc",100,3):.3f} ms for explicit-memory S3 ({time("q3_unified",100,3)/time("q2_malloc",100,3):.2f}x as long). The CPU/managed-S3 timing ratio is {time("q1_cpu",100)/time("q3_unified",100,3):.2f}x. These ratios describe the provided timed regions, not the total cost of the two memory-management strategies.')
h('Q4: comparison with predictions')
p('For every tested K, both memory modes follow the predicted ordering: S1 is slowest, then S2, then S3. CPU is faster than S1 and S2 and slower than S3 in this sweep. S1 stops at K=10 by design, so the shorter curve is expected. Both S1 and S2 grow broadly with input size because their launch counts grow with N. S3 scales more efficiently by keeping the launch count at one.')
p(f'From K=10 to K=100, explicit S3 grows {time("q2_malloc",100,3)/time("q2_malloc",10,3):.2f}x for 10x the data, consistent with a near-linear large-array trend. Managed S3 grows {time("q3_unified",100,3)/time("q3_unified",10,3):.2f}x over the same interval. The curves need not be exact straight lines: first-use costs and runtime/system variability matter, especially for short intervals. The measurements do not establish a steady-state managed-memory penalty.')
h('Timing fairness and WSL limitation')
p('Explicit copies are excluded from Q2. Q3 starts from CPU-initialized managed allocations, so memory residency work can affect first GPU use differently. This platform reports managedMemory=1, concurrentManagedAccess=0, and pageableMemoryAccess=0. NVIDIA documents that WSL lacks full managed-memory support and concurrent CPU/GPU access [1]. Therefore these results should not be interpreted as native-Linux demand-paging measurements or as proof that explicit allocation is always faster. The event interval does not separately quantify transfer, residency, host submission, and kernel costs.')
p('Only one complete sweep was measured, as allowed by the handout. No confidence intervals are claimed. Laptop clocks, power management, and background work were not controlled; causal attribution of small differences would require repeated runs or profiling.')
p('[1] NVIDIA, CUDA on WSL User Guide 12.8, section 4.1: <link href="https://docs.nvidia.com/cuda/archive/12.8.2/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications" color="#245b8a">Known limitations for Linux CUDA applications</link>.',True)

page(); h('Q4: explicit-memory time versus K')
picture(ROOT/'plots/partb_without_unified.png',maxheight=260)
p('Generated by the supplied q4_plot.py without modification, from partb_measurements.csv. Both axes are logarithmic. Each point is one measured scenario interval. S1 includes N launches, S2 includes ceil(N/256) launches, and S3 includes one launch.')
h('Q4: managed-memory time versus K')
picture(ROOT/'plots/partb_with_unified.png',maxheight=260)
p('Fresh managed allocations and first GPU use for each point. Compare the relative launch-pattern ordering; Q2 and Q3 do not have identical memory-transfer timing scopes.',True)

for index,k in enumerate(KS):
    if index % 2 == 0: page()
    h(f'Output transcript: K = {k} million')
    if index == 0:
        p('<b>Screenshots still required:</b> computer use was stopped before capture. The following are text transcripts of actual saved output, not screenshots. Capture each K_output.txt file in an ordinary viewer and add those images before submission.',True)
    text=(HERE/f'screenshots/K{k}_output.txt').read_text()
    story.append(Preformatted(text,ParagraphStyle(name='Log',fontName='Courier',fontSize=7.6,leading=10)))
    story.append(Spacer(1,10))

page(); h('Reproducibility and AI-use disclosure')
p('The experiment directory contains run_experiment.py, check_scenarios.cu, predictions.txt, build logs, original program stdout, environment.json, results.json, partb_measurements.csv, and per-K text excerpts prepared for screenshots. results.json records exact source hashes, toolchain paths, commands, timestamps, hardware information, and all 31 results. Source hashes were checked again after the sweep and were unchanged.')
p('To repeat this local environment, run experiments/partb/run_experiment.py under WSL. It builds through the supplied Makefile, checks small cases, and runs each binary with arguments 1 5 10 50 100. To regenerate only the plots without rerunning the lengthy scenarios, run the supplied src/partb/q4_plot.py with --csv plots/partb_measurements.csv --out-dir plots from the assignment root.')
p('AI-use disclosure: OpenAI Codex assisted with the Part B implementations, correctness tests, benchmark execution, plots, report drafting, and preparation of logs for screenshots. All reported timings and outputs came from actual local runs; the predictions were recorded before collecting these measurements.')
p('This is the Part B section, not the complete assignment submission. Combine it with the other parts and add the student name and UNI to the final report before submission.',True)

def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#c7d1da')); canvas.line(54,42,558,42)
    canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#536677'))
    canvas.drawString(54,29,'HPML HW3 | Part B | Actual local measurements')
    canvas.drawRightString(558,29,str(doc.page))

SimpleDocTemplate(str(pdfpath),pagesize=letter,rightMargin=54,leftMargin=54,topMargin=42,bottomMargin=54,title='HPML HW3 - Part B: CUDA Unified Memory').build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(pdfpath)
assert all(page.extract_text().strip() for page in reader.pages)
render=ROOT.parent/'tmp/pdfs/partb'; render.mkdir(parents=True,exist_ok=True)
pdf=pdfium.PdfDocument(str(pdfpath))
for i in range(len(pdf)):
    pdf[i].render(scale=1.4).to_pil().save(render/f'page_{i+1:02}.png')
print(f'Created {pdfpath} ({len(pdf)} pages), rendered to {render}')
