"""Assemble the report exclusively from saved measurements and output captures."""
from pathlib import Path
import json,shutil,sys
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,Preformatted
from PIL import Image as PILImage
import pypdfium2 as pdfium
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1];E=ROOT/'experiments'
def read(name):return json.loads((E/name).read_text(encoding='utf-8-sig'))
A1=read('parta_q1/results.json');A2=read('parta_q2/results.json');A3=read('parta_q3/eager_results.json');A4=read('parta_q4/results.json')
B=read('partb/results.json');C=read('partc_final/results.json');S=C['summary']
W=read('partc_final/warm_results.json');WS=W['summary']
T=read('partc_c4/results.json')
preview='--layout-preview' in sys.argv
if not preview:
    GRADER=json.loads((ROOT/'results.json').read_text())
    assert GRADER['meta']['mode']=='gpu'
    assert all(GRADER['part_c']['correctness'].values())
    for part in ['part_a','part_b','part_c']:
        assert all(v['status']=='ok' for v in GRADER[part].values() if isinstance(v,dict) and 'status' in v)
OUT=ROOT.parent/'output/pdf';OUT.mkdir(parents=True,exist_ok=True)
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyLab',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=9))
styles.add(ParagraphStyle(name='SmallLab',fontName='Helvetica',fontSize=8.5,leading=11,spaceAfter=7))
styles.add(ParagraphStyle(name='CellLab',fontName='Helvetica',fontSize=8.5,leading=11))
styles.add(ParagraphStyle(name='CellHead',fontName='Helvetica-Bold',fontSize=8.5,leading=11,textColor=colors.white))
styles['Title'].textColor=styles['Heading2'].textColor=colors.HexColor('#173652')
story=[]
def p(text,small=False):story.append(Paragraph(text,styles['SmallLab' if small else 'BodyLab']))
def h(text):story.append(Paragraph(text,styles['Heading2']))
def page():story.append(PageBreak())
def table(rows,widths):
    cells=[[Paragraph(escape(str(v)),styles['CellHead' if i==0 else 'CellLab']) for v in row] for i,row in enumerate(rows)]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#173652')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eef3f7'),colors.white]),('LINEBELOW',(0,-1),(-1,-1),0.5,colors.HexColor('#bac7d1'))]))
    story.extend([t,Spacer(1,10)])
def picture(path,maxh=550):
    with PILImage.open(path) as im:w,hh=im.size
    f=min(504/w,maxh/hh);story.append(Image(str(path),width=w*f,height=hh*f))
def code(text):story.append(Preformatted(text,ParagraphStyle(name='MonoLab',fontName='Courier',fontSize=9,leading=13,spaceAfter=10)))
def bt(prog,k,s=None):return float(next(r['time_ms'] for r in B['rows'] if r['program']==prog and int(r['K'])==k and (s is None or int(r['scenario'])==s)))
fmt=lambda v:f'{v:.3f}'

story.append(Paragraph('High Performance Machine Learning<br/>Homework Assignment 3',styles['Title']))
p('<b>Srikar Kovvuri | UNI: srk2211</b>')
if preview:p('<b>LAYOUT PREVIEW - final GPU grader is still running.</b>')
p('Parts A, B, and C1-C4 | November 21, 2026 | All times in milliseconds unless stated otherwise.')
h('Setup and measurement conventions')
p('Experiments used an NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB) and Intel Core Ultra 9 185H, with Ubuntu 24.04.1 in WSL2 on Windows. The toolkit was CUDA nvcc 12.6.85 with GCC 13.3; cuBLAS was 12.6.4.1 for Part C. Part A records report driver 566.26; Part B and the current Part C records report driver 621.14. Comparisons use measurements within the same experiment, not timings across driver sessions.')
p('GPU builds used -O3 -std=c++14 -arch=sm_89 and the shared CUDA runtime. The CPU baseline used -O2. Supplied Makefiles, common helpers, and required output formats were preserved. Local compiler/library paths were provided through build overrides or an nvcc wrapper rather than changing the assignment Makefiles.')
p('CUDA C++ intervals use CUDA events with synchronization before elapsed time is read. Input generation, allocations, explicit copies, validation, and cleanup are outside those intervals. Multi-launch Part B intervals include all launches and the gaps while the host submits them. Part C3 includes first-GEMM setup within the supplied event interval; it is explicitly distinguished from steady-state compute below.')
p('Part A uses five measured fresh-process runs per size after one preliminary run, with shuffled ordering. Part B uses one full sweep, as allowed by the handout. Part C uses five interleaved fresh-process runs per implementation after one preliminary process per implementation. A preliminary process does not warm data or cuBLAS state inside the subsequent processes. Medians summarize repeated measurements; no slow trials were discarded.')
h('Correctness and evidence')
p('Part A used the supplied output checks; Part B additionally passed 30 element-wise/guard tests. All three Part C implementations passed nine element-wise CPU-reference cases each. C2 also passed five direct-kernel tests, including full-size tensor validation, rising tile maxima, large logits, uniform weights, and output guards. '+('The final GPU grader is still running.' if preview else 'The unmodified local grader completed in GPU mode with all required checks passing; results.json is preserved. All 15 public tests also passed under WSL.'))
p('The screenshot appendix contains actual browser captures of a local saved-log viewer, with original output and source-log paths. These are screenshots of recorded stdout, not live terminal sessions. Part A screenshots show the observations equal to the reported medians, Part B includes every required K and launch scenario, and Part C includes the current median observations. The original user-captured C1 screenshot is also retained.')
h('AI-use disclosure')
p('OpenAI Codex assisted with build setup, tests, measurements, Part B code, C2/C3/C4 implementations, and report preparation. The student supplied the Part A optimized kernels and C1 implementation; reported outputs and screenshots are based on actual local runs.')

page();h('Part A Q1: vector addition baseline (3 points)')
p('vecadd00 assigns a consecutive segment to each thread: index = tid*v + j. At a fixed loop iteration, adjacent threads access locations v floats apart. With 240 blocks of 512 threads, there are 122,880 threads and N=122,880v elements. Inputs are 1 and 2; every output is checked for exactly 3.')
table([['Values/thread','Elements','Median ms','Mean +/- SD ms']]+[[r['values_per_thread'],f"{r['elements']:,}",fmt(r['median_kernel_ms']),f"{r['mean_kernel_ms']:.3f} +/- {r['stdev_kernel_ms']:.3f}"] for r in A1['summary']],[95,120,100,189])
p('The median increases from 44.814 to 172.280 ms for a fourfold increase in data (3.84x time), approximately linear with work. Each thread performs more iterations while the grid remains fixed. Adjacent-lane address strides are 2,000, 4,000, and 8,000 bytes. Scattered warp accesses waste memory transactions, although per-thread locality can still benefit caches. The 293.279 ms run at v=2000 was retained; this explains the large standard deviation.')
h('Part A Q2: coalesced vector addition (8 points)')
p('vecadd01 uses index = tid + j*T, where T is the total number of threads. At each iteration, neighboring lanes access neighboring floats. The operation and total element count are unchanged. Both versions were rebuilt and measured in the same experiment; Q2 speedups use this fresh baseline rather than the earlier Q1 numbers.')
table([['Values/thread','vecadd00 median','vecadd01 median','Speedup']]+[[r['values_per_thread'],fmt(r['vecadd00']['median_kernel_ms']),fmt(r['vecadd01']['median_kernel_ms']),f"{r['median_speedup']:.2f}x"] for r in A2['summary']],[95,145,145,119])
p('At 2000 values/thread, coalescing reduces the median from 172.162 to 18.947 ms (9.09x). Useful bandwidth, 12N/time for two FP32 reads and one write, rises from 17.13 to 155.65 GB/s. This is useful traffic, not a hardware-counter measurement of physical DRAM traffic. The measured 5.97-9.09x speedups support the importance of warp-wide access patterns. The same 122,880 threads and per-thread work counts were used, so this experiment does not isolate a benefit from changing thread count.')
p('All 18 Q1 runs and all 36 Q2 runs, including preliminary runs, passed. GPU clocks and power state were not fixed; five repetitions describe observed variation without establishing an exact scaling law.',True)

page();h('Part A Q3: shared-memory matrix multiplication (6 points)')
p('matmult00 uses 16x16 thread blocks and a 16x16 output footprint, one result per thread. Each reduction step loads A/B tiles into shared memory, synchronizes, accumulates, and synchronizes before reusing the tiles. Arguments 16, 32, and 64 produce matrix sides 256, 512, and 1024. Controlled Q3 results below use CUDA_MODULE_LOADING=EAGER; the original inherited-environment series remains in the raw records.')
table([['Matrix side','CLI argument','Grid','Median ms','Mean +/- SD ms']]+[[r['size'],r['argument'],f"{r['size']//16} x {r['size']//16}",fmt(r['median_kernel_ms']),f"{r['mean_kernel_ms']:.3f} +/- {r['stdev_kernel_ms']:.3f}"] for r in A3['summary']],[80,85,90,90,159])
p('Doubling the side multiplies arithmetic work by eight. From 512 to 1024 the median rises 7.58x (1.225 to 9.285 ms), close to that work scaling. Each shared A/B element is reused across 16 output multiply-accumulates. All tested dimensions fit the tile exactly; no padding is required.')
h('Part A Q4: four outputs per thread (10 points)')
p('matmult01 keeps 16x16 threads but expands the footprint to 32x32. Each thread accumulates a 2x2 quadrant subtile in four registers. Cooperative quadrant loads preserve adjacent-column accesses across threadIdx.x, barriers protect tile use/reuse, and the inner reduction loop is unrolled. Arguments 8, 16, and 32 preserve the required matrix sizes.')
table([['Side','00 / 01 arguments','00 median ms','01 median ms','Speedup']]+[[r['size'],f"{r['matmult00']['argument']} / {r['matmult01']['argument']}",fmt(r['matmult00']['median_kernel_ms']),fmt(r['matmult01']['median_kernel_ms']),f"{r['median_speedup']:.2f}x"] for r in A4['summary']],[65,120,110,110,99])
p('These paired Q4 measurements were collected in a later session; the baseline is rerun beside the optimized version. The sizable Q3/Q4 baseline difference is a session difference of unmeasured cause, not a kernel change or an optimization speedup. Only the paired Q4 ratios are used to evaluate matmult01.')
p('At footprint F, the requested A/B input loads are 2n^3/F floats, so F=32 halves those loads relative to F=16 and doubles per-tile reuse. Shared tile storage grows from 2 to 8 KiB and each thread keeps four accumulators. The grid has one quarter as many blocks; extra resource use and less block-level parallelism can offset reuse. At 1024, the measured speedup is 1.68x; at 256, 0.442 to 0.453 ms shows no benefit. Both versions already contain an unroll directive, so no unrolling-only speedup is claimed.')
p('Every matrix output passed the CPU reference with tolerance 0.01; observed maximum absolute error was zero. A 7.980 ms Q4 baseline run at 1024 was retained. Ratios of medians limit its influence without hiding it.',True)

page();h('Part A Q5: rules of thumb and an unsuccessful change (3 points)')
p('<b>1. Make adjacent lanes touch adjacent global-memory addresses.</b> Coalescing vecadd at 2000 values/thread reduced 172.162 to 18.947 ms, a 9.09x speedup. Think about addresses across a warp at the same instruction, not just the sequence one thread follows.')
p('<b>2. Reuse data in shared memory and registers when the workload is large enough.</b> Increasing the matmul footprint from 16 to 32 doubles tile reuse and retains four output accumulators per thread. At 1024x1024 this changed 2.640 to 1.567 ms (1.68x). The load reduction is a source-level count; the experiment does not separately measure the effects of reuse, register pressure, and occupancy.')
p('<b>3. Balance work per thread against the number of blocks available to schedule.</b> At 256x256, moving from one to four outputs/thread reduced the grid from 256 to 64 blocks, but time changed from 0.442 to 0.453 ms (0.98x speedup). More per-thread work is not automatically faster, even when it increases data reuse.')
p('<b>Change that did not improve performance:</b> the 32x32 register-blocked footprint at 256x256 was about 2.5% slower in the measured medians. Given variability, the supported conclusion is that it did not demonstrate a benefit at this size, not that it must always be slower.')
h('Part A variability and interpretation')
table([['Matrix side','Q4 baseline mean +/- SD','Q4 optimized mean +/- SD']]+[[r['size'],f"{r['matmult00']['mean_kernel_ms']:.3f} +/- {r['matmult00']['stdev_kernel_ms']:.3f}",f"{r['matmult01']['mean_kernel_ms']:.3f} +/- {r['matmult01']['stdev_kernel_ms']:.3f}"] for r in A4['summary']],[100,202,202])
p('All timings measure the supplied event regions, not complete process duration. The environment is a laptop/WSL desktop without clock locking. Startup effects, scheduling, and clock/power variation can affect short kernels; no hardware profiling was performed to attribute a timing change to one cause. Part A conclusions therefore use paired comparisons and explicitly report the negative small-matrix result.')

page();h('Part B Q1-Q3: CPU, explicit memory, managed memory (12 points)')
p('Each array contains N=K x 1,000,000 FP32 elements. Q1 performs a CPU loop. Q2 uses cudaMalloc and cudaMemcpy; Q3 uses fresh cudaMallocManaged allocations per scenario, initialized on the CPU, without prefetch or a data-warming pass. All 31 required observations passed checksum=3N. Thirty additional small tests checked every element and output guards at N=1,255,256,257,513.')
table([['K (million)','Q1 CPU ms','Checksum']]+[[k,fmt(bt('q1_cpu',k)),f'{3*k*1000000:,}'] for k in [1,5,10,50,100]],[110,170,224])
p('S1 submits N launches with one block and one thread each. S2 submits ceil(N/256) launches, each with one 256-thread block. S3 submits one launch with ceil(N/256) blocks of 256 threads. The i&lt;N guard protects partial blocks. S1 is intentionally capped at K=10.')
for prog,label in [('q2_malloc','Q2: explicit memory'),('q3_unified','Q3: managed memory, first GPU use')]:
    p('<b>'+label+'</b>',True)
    table([['K','S1 ms','S2 ms','S3 ms']]+[[k,fmt(bt(prog,k,1)) if k<=10 else 'Not required',fmt(bt(prog,k,2)),fmt(bt(prog,k,3))] for k in [1,5,10,50,100]],[55,155,147,147])
p('Allocations and explicit copies are outside Q2 timing. First GPU use of managed allocations can incur residency work differently in Q3. The measured device reports managedMemory=1, concurrentManagedAccess=0, pageableMemoryAccess=0; WSL lacks full managed-memory support [2]. These columns are not a direct end-to-end comparison of memory strategies.',True)

page();h('Part B Q4: predictions, plots, and explanation (3 points)')
p('Predictions were saved before measurement in predictions.txt, with timestamp/hash recorded by the runner. CPU, S1, and S2 were expected to grow broadly linearly with N. S1 was predicted slowest, S2 intermediate, and S3 fastest; S3 could show a fixed-overhead floor at small N and near-linear bandwidth scaling at large N. CPU was expected to beat S1/S2; explicit S3 was expected to beat CPU at large N. First-use managed-memory overhead was expected to depend on the platform.')
picture(ROOT/'plots/partb_without_unified.png',205)
picture(ROOT/'plots/partb_with_unified.png',205)
p('Both charts were generated with the supplied q4_plot.py unchanged and use log-log axes. Every tested size follows the predicted S1>S2>S3 time ordering. At K=10, explicit S1 takes 149.405 seconds versus 0.571 ms for S3: submission overhead and poor parallel utilization dominate S1. At K=100, explicit S3 is 55.08x faster than the CPU within the measured scopes; S2 is 17.29x slower than the CPU.')
p('Managed S3 takes 95.726 ms at K=100 versus 5.932 ms explicit (16.14x as long), while remaining 3.41x faster than the CPU. This does not establish a steady-state or end-to-end penalty. From K=10 to 100, S3 increases 10.39x explicit and 10.56x managed for 10x data, consistent with near-linear large-input scaling. Smaller departures from straight lines can reflect fixed costs, caches, first writes, and uncontrolled runtime variation; one sweep cannot isolate those causes.',True)

page();h('Part C1: direct CUDA attention (15 points)')
p('Q, K, and V are Nxd FP64 matrices, with N=2048 and d=64 by default. C1 computes S=QK^T/sqrt(d), replaces S in place with a stable row softmax P, then computes O=PV. Each score/output is owned by one thread in a 16x16 block. Softmax assigns one row to a thread, subtracts its maximum, sums exponentials, and normalizes. The full N x N score allocation occupies 32 MiB.')
p('The student implementation was preserved. Its checksum sums every element of O. The supplied CUDA events enclose the three ordered launches and synchronize the stop event. The serial softmax row scans and repeated exponentials are limitations of this correct baseline.')
h('Part C2: tiled online attention (25 points)')
p('One 128-thread block owns a query row. It stages that query, 128 scores/weights, and reduction scratch in shared memory. It reads keys/values tile by tile from global memory; each thread accumulates its output columns in registers. No global N x N score or probability matrix is allocated. Dynamic shared memory is (d+256) doubles (2,560 bytes at d=64), plus three shared scalars; up to eight register accumulators per thread cover the specified d<=1024.')
p('For running maximum m, denominator l, and unnormalized output vector u, and a new tile with scores s_j and value vectors v_j, the update is:')
code("m_new = max(m, max_j(s_j))\na     = exp(m - m_new)\np_j   = exp(s_j - m_new)\nl_new = a*l + sum_j(p_j)\nu_new = a*u + sum_j(p_j*v_j)\nO     = u/l after the final tile")
p('Initially m=-infinity, l=0, and u=0. When m increases, every earlier exponential must be expressed relative to the new maximum: exp(s-m_new)=exp(m-m_new)*exp(s-m). Therefore both the denominator and output numerator must be rescaled. Rescaling only the denominator would over-weight the earlier numerator and produce the wrong normalized output; leaving both unchanged would mix incompatible normalization scales across tiles.')
p('Block reductions compute each tile maximum and exponential sum. Barriers protect query loading, reduction steps, shared state updates, and tile reuse. Invalid keys in the last tile receive zero weight and never read V. Adjacent output threads read adjacent V columns. This design removes score-matrix traffic but does not eliminate all K/V traffic or guarantee a speedup at every shape.')
p('C2 passed all nine small reference cases and five additional direct-kernel checks. Tests include maxima that rise across tiles, very large logits, uniform softmax, d=1024, partial tiles, guards, and the full 2048x64 tensor against an independent CPU computation. Detailed errors are recorded in check_tiled.log.',True)

page();h('Part C3: cuBLAS attention (15 points)')
p('Both matrix multiplies use cublasDgemm. The intermediate softmax retains C1\'s one-thread-per-row algorithm and 256-thread launch, so it remains a potential bottleneck. The array bytes are row-major, while cuBLAS interprets them as column-major [1]. A row-major R x C array is therefore viewed as its C x R transpose.')
table([
['Call/result','Column-major operands and operation','m,n,k','Flags A/B','lda,ldb,ldc'],
['Scores: S^T','A=K^T (d x N), op(A)=K (N x d); B=Q^T (d x N). S^T=scale*K*Q^T.','N,N,d','T / N','d,d,N'],
['Output: O^T','A=V^T (d x N); B=P^T (N x N). O^T=V^T*P^T.','d,N,N','N / N','d,N,d']],[55,225,75,65,84])
p('The first call passes dK then dQ, alpha=1/sqrt(d), beta=0, and writes column-major S^T into dS. Those same bytes are row-major S for the softmax. The second call passes dV then dS, alpha=1, beta=0, and writes column-major O^T, which is the row-major O consumed by the host. At the default size, the leading dimensions are 64/64/2048 and 64/2048/64 respectively.')
h('Supplied-driver first-invocation results: N=2048, d=64')
table([['Program','Checksum','Median ms','Range ms','C1/time']]+[[name,f"{s['checksum']:.3f}",fmt(s['median_ms']),f"{s['min_ms']:.3f}-{s['max_ms']:.3f}",f"{S['c1_naive']['median_ms']/s['median_ms']:.2f}x"] for name,s in S.items()],[100,115,85,115,89])
p(f"C2 takes {S['c2_tiled']['median_ms']:.3f} ms versus {S['c1_naive']['median_ms']:.3f} ms for C1, a {S['c1_naive']['median_ms']/S['c2_tiled']['median_ms']:.2f}x timing ratio. All three checksums agree to the reported precision; correctness was also tested element by element, rather than inferred from checksums.")
p('C3 timing includes the first GEMM invocation in each fresh process, despite EAGER module loading. cublasCreate occurs before the start event, but first-use library/kernel/workspace work can still delay the event interval. These are supplied-driver first-invocation measurements, not steady-state cuBLAS throughput. A separate warm process does not remove that cost in subsequent processes. A slow C3 result therefore does not establish that optimized GEMM is inherently slower than the handwritten kernels.',True)

page();h('Part C: steady-state kernel timing and required output')
p('To separate first-call overhead from repeated kernel execution, a supplemental harness includes the original C1, C2, and C3 source files without editing their kernels. It uses the same launch configurations and cuBLAS argument mapping, allocates/copies inputs and creates the cuBLAS handle before timing, performs three excluded warm-up pipelines per variant in the same process, then measures ten rounds with variant order rotated. CUDA events surround each complete pipeline and synchronize before reading elapsed time. Data stays resident on the GPU.')
table([['Program','Checksum','Median ms','Range ms','C1/time']]+[[n,f"{s['checksum']:.3f}",fmt(s['median_ms']),f"{s['min_ms']:.3f}-{s['max_ms']:.3f}",f"{WS['c1_naive']['median_ms']/s['median_ms']:.2f}x"] for n,s in WS.items()],[100,115,85,115,89])
p('Report output for C1, C2, and C3, in that order: checksum and median steady-state kernel milliseconds, formatted to three decimals:')
code('\n'.join(f"{WS[n]['checksum']:.3f},{WS[n]['median_ms']:.3f}" for n in ['c1_naive','c2_tiled','c3_cublas']))
p(f"C4 (FP32, different random inputs): checksum={T['checksum']:.6f}, kernel benchmark={T['triton_ms']:.6f} ms. Its precision and timing protocol differ from C1-C3; see the C4 section for the matched SDPA comparison.",True)
p(f"Under this matched resident-data protocol, C2 achieves {WS['c1_naive']['median_ms']/WS['c2_tiled']['median_ms']:.2f}x and C3 achieves {WS['c1_naive']['median_ms']/WS['c3_cublas']['median_ms']:.2f}x relative to C1. C3's warmed median is {WS['c3_cublas']['median_ms']:.3f} ms compared with {S['c3_cublas']['median_ms']:.3f} ms for the supplied-driver first-call median. This difference demonstrates why first-use setup must be separated from steady-state execution; it is not a precise decomposition of setup cost, because the two protocols also differ in residency and runtime state.")
p('The C3 pipeline still uses the serial-per-row softmax, so replacing the GEMMs alone does not optimize the entire attention computation. C2 removes the 32 MiB global score allocation and the corresponding score/probability reads and writes. These explanations follow the implementation; no per-kernel hardware-counter profile was collected to assign an exact fraction of time to each bottleneck.')
p('All ten observations were retained. The first three measured rounds were slower across all variants; clocks and power state were not locked, and the cause was not isolated. The ranges and raw screenshot records expose this variation rather than treating the median as a fixed hardware constant.',True)
p('All 39 warm-up/measured pipeline outputs were checked element by element against an independent CPU computation of the full 2048x64 tensor, with tolerance 1e-8. This validation and the device-to-host output copies occurred after the stop event. All passed. The median of ten observations can be the average of two middle timings and need not equal a single observed run.',True)
table([['Program','Maximum full-tensor absolute error']]+[[n,f"{s['max_abs_error']:.3g}"] for n,s in WS.items()],[180,324])

page();h('Part C raw timings and validation')
table([['Program','Run 1','Run 2','Run 3','Run 4','Run 5']]+[[n]+[fmt(v) for v in s['times_ms']] for n,s in S.items()],[104,80,80,80,80,80])
p('The three programs were run in interleaved order for five measured rounds. A preliminary run for each implementation was excluded. Each measured run allocates fresh inputs and executes one pipeline. Host/device transfers and handle creation are outside event timing; first-call work inside C3 is disclosed above. No outlier was removed.')
table([['Implementation','Reference cases','Largest observed printed-output error']]+[[n,9,f"{max(r['max_abs_error'] for r in C['correctness'] if r['program']==n):.3g}"] for n in S],[145,120,239])
p('The independent Python reference checks (N,d)=(1,1), (8,4), (17,19), (33,65), (63,64), (129,4), (257,8), (129,31), and (3,1024), with absolute tolerance 1e-6. Driver output is printed with nine significant digits, so these comparisons include text-rounding error. All 27 cases passed. C2\'s direct full-tensor and stability tests compare unrounded doubles with tolerance 1e-8.')
code(C['extra_checks'].strip())
p(('Final GPU grading is in progress. ' if preview else 'The original grade_local.py also ran in real GPU mode across all required parts. It rebuilt from the supplied Makefiles and independently validated each attention implementation at the public test sizes. ' )+'C1 and the provided common.h/Makefile were left unchanged; C2/C3 source hashes are recorded with the final measurements.')
h('References and reproducibility')
p('[1] NVIDIA cuBLAS documentation, cublas&lt;t&gt;gemm: <link href="https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm" color="#245b8a">column-major operands, transpose flags, and leading dimensions</link>.',True)
p('[2] NVIDIA CUDA on WSL User Guide 12.8, section 4.1: <link href="https://docs.nvidia.com/cuda/archive/12.8.2/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications" color="#245b8a">managed-memory limitations</link>.',True)
p('Assignment requirements follow the user-supplied updated HPML HW3 handout. Experiment directories parta_q1 through parta_q4, partb, partc_c1, and partc_final preserve raw logs, exact build commands, source hashes, run order, and statistics. plots/ contains the unchanged-script Part B plots and CSV. report_evidence/ contains the HTML log viewers, actual browser captures, source paths, and screenshot SHA-256 hashes.',True)

page();h('Part C4: tiled attention in Triton (bonus, 10 points)')
p('One Triton program owns 64 query rows. It loads a query tile, then loops over 64-key/value tiles. The running row maximum, normalization denominator, and output numerator remain local to the program. The same online-softmax rescaling derived for C2 updates both numerator and denominator before incorporating a new tile. Only the N x d output is allocated; there is no global N x N score or probability buffer.',True)
p('Both tile matrix products use tl.dot with input_precision="tf32x3" and FP32 accumulation. This uses the available tensor-core path while retaining more FP32 accuracy than a single TF32 product. At d=64 the launcher uses four warps and two stages per program, passes all tensor strides, and masks partial query/key tiles and padded head dimensions. For heads above 64 it reduces both tiles to at most 32 and uses one stage to fit shared memory; the supported head range is 1-128. The final result is the accumulated numerator divided by the denominator.',True)
p(f"The executed notebook uses PyTorch {T['torch']}, Triton {T['triton']}, CUDA {T['cuda']}, and {T['gpu']}. The supplied N=2048, d=64, seed-0 FP32 inputs are unchanged. PyTorch reference matrix multiplication has TF32 disabled. C1-C3 instead use FP64 and different input generation, so C4 checksum and timing are not directly compared against those versions.",True)
table([['Metric','Measured result'],['Triton checksum',f"{T['checksum']:.9f}"],['Reference checksum',f"{T['reference_checksum']:.9f}"],['Maximum absolute error',f"{T['max_abs_error']:.9g}"],['Triton latency (ms)',f"{T['triton_ms']:.6f}"],['PyTorch SDPA latency (ms)',f"{T['sdpa_ms']:.6f}"]],[250,254])
table([['Paired round','Triton mean ms','SDPA mean ms']]+[[i+1, f"{r['triton_ms']:.6f}", f"{r['sdpa_ms']:.6f}"] for i,r in enumerate(T['benchmark_rounds'])],[130,187,187])
winner='Triton' if T['triton_ms']<T['sdpa_ms'] else 'PyTorch SDPA'
ratio=max(T['triton_ms'],T['sdpa_ms'])/min(T['triton_ms'],T['sdpa_ms'])
p(f"{winner} was {ratio:.2f}x faster in this run. triton.testing.do_bench uses its default 25 ms warm-up budget, 100 ms repetition budget, and mean statistic. It times each function on the same resident inputs with CUDA events, excluding compilation and input generation. The helper clears L2 before each measured call; that clearing is outside its timed event pair. These are function-call GPU latency measurements, including any GPU work enqueued by the wrappers, rather than end-to-end data preparation time. Three paired rounds alternate the order of the two implementations. Reported latencies are the medians of those three do_bench means; every round is shown. Clock and power state were not locked. The first completed single-comparison run returned 7.561 ms for Triton and 0.570 ms for SDPA; it is preserved separately and prompted this repeated protocol, rather than being silently discarded.",True)
p('A separate CPU-operator profile, outside timing, observed '+', '.join(T['backend_ops'])+'. This identifies the selected SDPA implementation rather than assuming that FP32 SDPA uses FlashAttention. Backend-specific tiling, precision, and resource use can affect the comparison. This experiment does not isolate their individual contributions or establish performance at other shapes.',True)
p(f"The required error check passed at tolerance 1e-2. Seven additional checks cover N=1, partial tiles, a non-power-of-two head dimension, non-contiguous inputs, d=128, and large logits. Their largest absolute error was {max(x['max_abs_error'] for x in T['validation']):.6g}; all outputs were finite. SDPA also matched the same reference, with maximum absolute error {T['sdpa_max_abs_error']:.6g}.",True)
p('The completed src/partc/c4_triton.ipynb contains all executed cells and outputs. experiments/partc_c4 contains raw stdout, measurements, environment versions, source hash, and the execution script. The mandatory local grader does not grade this bonus notebook; its bonus status remains unchanged. C2 received a comment-only cleanup after the earlier recorded benchmarks.',True)
for q in [1,2,3,4]:
    page();h(f'Output screenshots: Part A Q{q}')
    p('Screenshots of recorded stdout; source-log paths are visible. Each image shows a measured observation equal to the corresponding reported median. Output text is unchanged.',True)
    sizes=[500,1000,2000] if q<=2 else [256,512,1024]
    for n in sizes:picture(E/f'report_evidence/a_q{q}_{n}.png',200)
for ks in [[1,5],[10,50],[100]]:
    page();h('Output screenshots: Part B / K='+', '.join(map(str,ks)))
    p('Each capture includes CPU, explicit-memory, and managed-memory output plus the derived launch configuration. All required K values and scenarios are covered; S1 above K=10 is intentionally omitted.',True)
    for k in ks:picture(E/f'report_evidence/b_k{k}.png',300 if len(ks)>1 else 570)
page();h('Output screenshots: Part C1-C3')
p('Current final experiment: actual median observations at N=2048, d=64. Each screenshot shows the required checksum/time line and the program\'s RESULT line.',True)
for n in ['c1_naive','c2_tiled','c3_cublas']:picture(E/f'report_evidence/{n}.png',145)
h('Earlier C1 check: original user screenshot')
picture(E/'partc_c1/c1_output_screenshot.png',105)
p('This earlier C1 result (44.766 ms) is preserved as historical evidence; the final comparison above uses the new interleaved measurements.',True)
for n in ['c1_naive','c2_tiled','c3_cublas']:
    page();h('Steady-state output screenshot: '+n)
    p('Actual browser capture of saved benchmark stdout. The ten measured records and the explicitly labeled computed median support the reported steady-state timing. Every record includes the independent full-tensor error.',True)
    picture(E/f'report_evidence/warm_{n}.png',580)

page();h('Output screenshot: C4 Triton and PyTorch SDPA')
p('Actual browser capture of the executed notebook stdout. The required correctness check and measured latencies are preserved verbatim.',True)
picture(E/'report_evidence/c4_triton.png',570)

def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#c7d1da'));canvas.line(54,42,558,42)
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#536677'))
    canvas.drawString(54,29,'Srikar Kovvuri | srk2211 | HPML Homework 3');canvas.drawRightString(558,29,str(doc.page))
dest=OUT/'report.pdf'
SimpleDocTemplate(str(dest),pagesize=letter,leftMargin=54,rightMargin=54,topMargin=40,bottomMargin=54,title='HPML Homework 3 - Srikar Kovvuri - srk2211',author='Srikar Kovvuri').build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(dest);assert all(p.extract_text().strip() for p in reader.pages)
shutil.copyfile(dest,ROOT/'report.pdf')
render=ROOT.parent/'tmp/pdfs/final';render.mkdir(parents=True,exist_ok=True)
pdf=pdfium.PdfDocument(str(dest))
for i in range(len(pdf)):pdf[i].render(scale=1.3).to_pil().save(render/f'page_{i+1:02}.png')
print(f'Created {dest}: {len(pdf)} pages. Rendered to {render}')
