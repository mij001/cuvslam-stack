# The Defense Primer — everything you need, from zero

*Read this top to bottom once, then use Parts V–VI as drill material. It
assumes you know what a CPU, RAM, an array, and matrix multiplication are —
nothing else. Every acronym is expanded on first use and again in the
glossary. Every number is one you can cite from a committed file in this
repository.*

---

# PART I — THE PROBLEM (why this research exists)

## 1. Physical AI and why a robot must know where it is

"Physical AI" means AI systems that act in the physical world — robots,
drones, autonomous cars, AR headsets. Unlike a chatbot, a robot has a hard
real-time job that never pauses: **it must know where it is and what the
world around it looks like, tens of times per second**, from its cameras and
sensors. That job is called **SLAM — Simultaneous Localization And Mapping**:
build a map of an unknown place *while* tracking your own position inside it.
It is "simultaneous" because the two halves feed each other — you locate
yourself against the map, and you extend the map from located positions.

The workload we study is **cuVSLAM**: NVIDIA's production, GPU-accelerated
visual SLAM library (it ships inside NVIDIA Isaac, their robotics stack).
"Production" matters for the defense: we did not characterize a toy from a
paper, we characterized the code that actually runs on shipping robots.

## 2. The SLAM pipeline in one walk-through

Think of it as three nested loops, fast to slow:

1. **Front-end (every frame, ~30–100× per second).** A camera image arrives.
   The pipeline converts it to grayscale, builds an **image pyramid** — the
   same image at multiple shrinking resolutions (full, half, quarter…), which
   lets you track both large and small motions — and computes **gradients**
   (edge maps). It finds corners worth tracking (**GFTT — "Good Features To
   Track"**, the Shi–Tomasi corner detector) and follows them frame-to-frame
   with **LK — Lucas–Kanade optical flow** (a small least-squares solve per
   feature that asks "where did this patch move to?").
2. **Visual odometry / back-end (every frame).** From the tracked features
   the system solves for camera motion. The heavy part is **BA — Bundle
   Adjustment**: a nonlinear least-squares problem over recent camera poses
   and 3-D landmark positions ("adjust the bundle of light rays so all
   observations agree"). Numerically it builds and solves a linear system
   each iteration (a Hessian matrix, reduced by a trick called the **Schur
   complement** that eliminates landmark variables first). In our data this
   is the structure tagged **`ba_linear_system`**.
3. **SLAM layer (occasionally).** Some frames are promoted to **keyframes**
   (snapshots worth remembering). Their feature descriptors go into a
   **keyframe database**. When the robot returns somewhere it has been, the
   system detects it by scanning that database — a **loop closure** — and
   corrects the accumulated drift of the whole trajectory. Loop closure is
   rare but crucial: without it, maps bend and drift without bound.

Modes: cuVSLAM can run **RGB-D** (color + depth camera, e.g. indoor robots),
**stereo** (two cameras, e.g. cars), mono, with or without an IMU (inertial
measurement unit). "Odometry mode" runs loops 1–2 only; "SLAM mode" adds
layer 3. Our campaign runs SLAM mode everywhere, which contains odometry as a
subset.

## 3. The memory wall, in plain numbers

Since the 1990s, processor arithmetic has gotten faster much quicker than
memory has. Today, moving a byte from DRAM (main memory) costs **orders of
magnitude more time and energy than computing with it**: a 64-bit
floating-point multiply costs roughly a picojoule-scale amount of energy; a
DRAM access costs hundreds of times that; going to storage costs thousands.
This is "the memory wall." For workloads that stream lots of data and do
modest math per byte — image pipelines are the canonical example — the
processor mostly *waits*.

The industry response is **memory-centric hardware**: instead of dragging
data to the compute, put compute where the data already is. Three flavors
matter to this thesis:

- **PiM / PnM — Processing in/near Memory.** Small compute units placed at or
  next to DRAM banks, so data is processed at full internal-memory bandwidth
  without crossing the memory bus. Real products exist: UPMEM's DPUs,
  Samsung's HBM-PIM, SK hynix AiM.
- **ISP — In-Storage Processing** (a.k.a. computational storage). Compute
  inside the SSD, so you can scan/filter a database where it lives instead
  of hauling it through the whole system (e.g. Samsung SmartSSD).
- **Near-sensor processing.** Compute stacked on or next to the image sensor
  itself, so per-frame pixel work happens before the data ever reaches DRAM.

The architecture community's standing question: **which data of which
workload belongs on which of these substrates?** Answering that requires a
*characterization* — a careful measurement of how a workload actually uses
memory. That is this thesis.

## 4. The thesis statement, in plain words

> cuVSLAM's performance is limited by data movement, not arithmetic. Its
> data structures fall into three classes by *lifetime and touch pattern* —
> streaming (per-frame images), hot-persistent (per-frame solver state), and
> cold-persistent (the session-long keyframe database) — and each class maps
> to a different memory-centric substrate: near-sensor SRAM, DRAM-PiM, and
> ISP respectively. We prove the classification by direct measurement, at
> the granularity of individual data structures, across 27 sequences from 4
> public datasets, and the classification is stable across all of them.

The novelty is not "SLAM uses memory a lot" (folklore). It is: (a) the
**per-data-structure evidence** — nobody before measured *which allocation*
each GPU kernel's memory traffic lands in for a production SLAM system; (b)
the **taxonomy validated by clustering rather than asserted**; (c) two
documented **self-corrections** ending with independent methods agreeing —
which is what makes the numbers trustworthy.

---

# PART II — THE MACHINERY (concepts you must be fluent in)

## 5. GPU 101

A **GPU** (graphics processing unit) is a throughput machine: thousands of
small arithmetic units organized into **SMs (Streaming Multiprocessors)** —
our RTX 2000 Ada GPU (NVIDIA's "Ada Lovelace" generation; compute capability
**sm_89**) has dozens of SMs. You program it by writing **kernels**:
functions launched over a grid of thousands–millions of **threads**. Each
individual execution of a kernel is a **launch** (our full TUM run has
~200,000 launches; we index them with a **grid-launch id** — a simple
counter, the "clock" our tooling uses).

Threads are executed in fixed groups of 32 called **warps**. The 32 slots in
a warp are **lanes**. All 32 lanes execute the same instruction at the same
time. Two warp-level health metrics matter here:

- **Divergence**: if some lanes are switched off (e.g. by an `if`), you waste
  hardware. `active_lanes = 32` means fully converged (no waste). *Every
  cuVSLAM kernel we measured runs at 32.0 — memory, not divergence, is the
  problem.*
- **Coalescing** — see §7.

## 6. GPU memory spaces — the most important section in this document

A GPU thread can address several distinct memory spaces. Confusing them
destroyed one of our own early results, so know them cold:

| space | where it physically lives | what it's for | in our traces |
|---|---|---|---|
| **global** | DRAM (the GPU's main memory) | the actual data: arrays you `cudaMalloc` | `LDG`/`STG` instructions |
| **shared** | on-chip SRAM inside each SM | a program-managed scratchpad; tiles for convolutions, sort staging, reduction trees | `LDS`/`STS` |
| **local** | DRAM, but a special per-thread region | **register spills**: when a thread needs more variables than it has registers, the compiler silently stores the overflow here; also thread-local arrays | `LDL`/`STL` |
| constant | DRAM + special cache | small read-only parameters | (excluded by the tool) |
| **texture** | DRAM, read via dedicated texture units | images read with hardware interpolation | **invisible to our tracer** (a disclosed limitation) |

Three facts to internalize:

1. **Shared memory is not DRAM.** Traffic to it says nothing about main
   memory. A convolution kernel that does 95 % of its accesses to shared
   tiles is *not* memory-hungry in the DRAM sense.
2. **Local memory IS DRAM**, but it is *compiler scratch*, not a data
   structure. Its addresses are interleaved so that lane 0..31 hitting "their
   own slot" produces adjacent addresses — **spill traffic is perfectly
   coalesced by construction**. This is the mechanism behind our biggest
   correction (§17).
3. An address tracer that records *all* spaces will show you a blended
   stream where spill and tile traffic can be 88–98 % of records. Any
   locality or attribution analysis must **filter by space first**.

## 7. Coalescing, sectors, cache lines

GPU DRAM and the L2 cache move data in **32-byte units called sectors**
(4 sectors = one 128-byte cache line). When a warp's 32 lanes issue a load
together, the hardware merges their addresses:

- If the lanes read 32 *consecutive* 4-byte floats → they fit in 4 sectors →
  **coalesced** (efficient: few sectors per request).
- If each lane reads a random location → up to **32 sectors** for one
  instruction → **scattered** (each lane drags in a whole sector to use a
  few bytes).

So **"sectors per warp access"** is the scatter meter: ~2–4 = streaming;
~30 = full scatter. Memorize two of ours: front-end kernels ≈ **4.0**
(streaming); the loop-closure scan's *global* accesses ≈ **23–30**
(scattered gather).

## 8. Caches and reuse distance (the architecture-independent trick)

A cache keeps recently used data close. Whether a cache of size S helps
depends on your **reuse distance**: for each access, how many *distinct*
things were touched since the last time you touched this one (also called
LRU stack distance — "LRU" = least-recently-used, the standard eviction
policy). If your reuse distance is smaller than the cache capacity, that
access would hit.

The beautiful property: reuse distance is a property of the **address
stream**, not of any particular cache. From one measured trace you can
predict the hit rate of *any* cache size — that's our
"hit-rate-vs-capacity **CDF**" (cumulative distribution function: what
fraction of accesses have reuse distance ≤ x, plotted over x).

Reading the curve: a **flat CDF** from 64 KiB to 48 MiB means "no realistic
cache size changes anything" — the misses are **compulsory** (first-touch,
cold): the data simply arrives, is used, and never returns. That is the
signature of streaming, and it's what all cuVSLAM front-end kernels show.
No cache fixes it; **not bringing the data to DRAM at all** (near-sensor
processing) fixes it.

## 9. The measurement tools, and what each one can and cannot see

- **nsys (NVIDIA Nsight Systems).** A timeline profiler: when did each kernel
  and copy run, on which stream. Cheap, whole-run. With **NVTX** (NVIDIA
  Tools Extension — named code ranges), the timeline shows *which pipeline
  stage* each kernel belongs to. cuVSLAM ships with NVTX annotations built
  in but disabled at two levels; we switched them on (a two-line change +
  build flag).
- **ncu (NVIDIA Nsight Compute).** Per-kernel **hardware counters** — the
  chip's built-in event tallies (bytes read, cache hit rates, sectors per
  request, stall reasons). Precise about *how much*, but it's an aggregate:
  it cannot tell you *which addresses* or *which data structure*. Counter
  metrics are therefore **proxies** — they correlate with behavior but need
  interpretation.
- **NVBit (NVIDIA Binary Instrumentation Tool).** A research tool that
  rewrites the GPU binary at load time and can call your code on **every
  memory instruction**, giving the actual **address trace** — ground truth,
  at 5–50× slowdown and gigabytes-to-terabytes of output. We patched its
  `mem_trace` example twice: (1) *windowing* (only instrument a chosen range
  of launch ids, or only kernels whose name matches a filter) to make traces
  feasible; (2) an *allocation sidecar* (log every memory allocation/free
  with the current launch id) to make traces attributable.
- **Our TaggedAllocator layer** (source-level): every GPU allocation in
  cuVSLAM logs its pointer, size, and a **host backtrace** (the chain of
  function calls that requested it). Resolving those with debug symbols
  (`addr2line` reading **DWARF**, the standard debug-info format) tells us
  *which data structure* owns every buffer. Three independent layers —
  source journal, driver-level sidecar, address trace — cross-check each
  other; that redundancy is how we caught our own bugs.

The epistemic ladder to recite when challenged: **counters tell you how
much; traces tell you which addresses; the allocator journal tells you which
data structure; NVTX tells you which pipeline stage.** Claims in the thesis
always cite the highest rung available, and where two rungs disagreed we
kept digging until they agreed (§17).

## 10. The datasets (why 27 sequences, and what they are)

- **KITTI** (odometry benchmark, seqs 00–10): a car driving through city
  streets; stereo cameras; kilometers-long; some sequences contain big loop
  closures (00, 02, 05, 06, 07, 09), some none (01, 04) — the loop-free
  ones are a deliberate *control group*.
- **EuRoC MAV** (11 sequences): a drone in a machine hall and rooms with
  motion-capture ground truth; stereo + IMU; aggressive motion.
- **TUM RGB-D fr3** (4 sequences): a handheld RGB-D camera indoors. Its
  `long_office_household` sequence orbits a desk island repeatedly — loop
  closures fire densely and predictably, which is why it is our canonical
  deep-dive workload. The "nostructure/notexture" variants are stress
  ablations.
- **TUM-VI**: visual-inertial sequences (corridor).

Four datasets × indoor/outdoor × RGB-D/stereo × loop/no-loop = the
generalization matrix. 27 sequences, zero failures.

---

# PART III — THE RESEARCH METHODOLOGY (defending the "how")

## 11. What "workload characterization" research is

An architecture paper says "build hardware X." A **characterization paper**
says "here is how workload W actually behaves, measured properly — and here
is what that implies for hardware." Characterization is publishable on its
own (venues below) because everyone downstream — hardware architects,
compiler writers — depends on it being done right. The bar is: rigorous
measurement, honest limitations, reproducibility, and *insight* (numbers
that change what a designer would do).

**Venues you'll name:** ISPASS (IEEE Int'l Symposium on Performance Analysis
of Systems and Software) and IISWC (IEEE Int'l Symposium on Workload
Characterization) for characterization; MICRO, ISCA, ASPLOS, HPCA — the four
top architecture conferences — for the follow-on "build it" paper. The
standard two-paper arc: characterize (ISPASS/IISWC) → design + simulate
(MICRO/ASPLOS class). We are at the end of stage 1.

**DAMOV** (Oliveira et al., 2021) is the landmark data-movement
characterization study: it defined a workflow — profile workloads with a
standard toolchain, extract memory metrics, cluster them into *classes*, and
map classes to memory-centric hardware. We adapted DAMOV's CPU-oriented
workflow to GPUs (different tools, warp-level metrics, a
divergence/coalescing axis CPUs don't have) and combined it with an
NCU-based **roofline** method (Cao et al., 2023).

**Roofline model** (60 seconds): plot performance (FLOP/s) against
**arithmetic intensity** (FLOPs performed per byte moved). Two ceilings bound
you: the flat compute roof (peak FLOP/s) and the slanted memory roof (peak
bandwidth × intensity). A kernel under the slanted roof is
**memory-bound** — more ALUs won't help it; only moving fewer bytes or
moving them cheaper will. Most of cuVSLAM lives under the slanted roof.

## 12. Experimental hygiene (what makes the numbers believable)

- **Locked clocks.** GPUs constantly change frequency (thermals, power).
  Two runs at different clocks aren't comparable. We pin GPU core and
  memory clocks (1620 MHz / 7001 MHz). Effect, measured: run-to-run
  variation (CoV) dropped from **49.6 %** (laptop, unlocked) to **0.14 %**
  (workstation, locked). **CoV — coefficient of variation** = standard
  deviation ÷ mean; 0.14 % means repeats agree to about a thousandth.
- **Measured ceilings.** Rooflines need the *achievable* peaks, not
  datasheet marketing: we measured **205.0 GB/s** DRAM bandwidth and
  **5445 GFLOP/s** FP32 on our locked card and use those.
- **Repeats & medians**: ×5 (or ×3) runs, medians reported; medians resist
  outliers.
- **Sensitivity analysis**: every classification threshold was perturbed by
  ±25 % and the class labels re-derived — labels that survive perturbation
  are robust, and we report which ones flip (they flip at a physically
  meaningful boundary: whether a working set fits L2).
- **Cold/warm bracketing**: ncu flushes caches between its replay passes
  (pessimistic); we bracket results between cold and warmed regimes rather
  than pretending one number is truth.
- **Determinism**: SLAM runs in synchronous mode for capture, so kernel
  order is reproducible; the allocator journal came out **identical (240
  allocations) across every run and even between a 30-frame and 2500-frame
  run** — that determinism is itself a finding (§16).
- **Traces are never used for timing.** Instrumentation slows execution
  5–50×; we use traces only for *addresses*, never for durations. Timing
  comes from uninstrumented locked-clock runs.

## 13. The statistics, each in one breath

- **Median**: middle value; robust to outliers.
- **CoV**: std ÷ mean — dimensionless "how noisy are repeats."
- **CDF**: for each x, the fraction of samples ≤ x. Our hit-vs-capacity
  curves are CDFs of reuse distance.
- **k-means**: unsupervised clustering; pick k centers, assign points to the
  nearest, move centers to the mean, repeat. Needs k given — so you sweep k
  and score each.
- **Silhouette score**: for each point, (distance to nearest other cluster −
  distance within own cluster), normalized to [−1, 1]; higher = better
  separated. Our sweep peaks at **k = 7–8**, matching the 7 behavior classes
  we had defined independently — that is the "the classes are in the data,
  not in our heads" argument.
- **Purity**: fraction of each cluster belonging to its majority label
  (ours: 0.68 vs our tree labels).
- **ARI — Adjusted Rand Index**: agreement between two partitions of the
  same items, corrected for chance (0 = random, 1 = identical; ours 0.30 —
  moderate agreement, expected since the tree uses thresholds and k-means
  uses geometry).
- **Jaccard similarity**: |A∩B| / |A∪B| for two sets — we use it for "how
  much of the loop-closure scan's working set is the same as last scan"
  (0.67 room-scale → 0.90 street-scale; i.e. 10–33 % turnover per scan).
- **Modal / agreement %**: mode = most common value. "48/49 kernels have a
  unanimous top data-structure tag" means for 48 kernels, *every* sequence
  votes for the same answer.

## 14. The attribution pipeline (our main methodological contribution)

Question it answers: *"Kernel K is memory-bound — but on WHICH data?"*

1. **Layer 1 — source journal.** Every `cudaMalloc`-family call inside
   cuVSLAM's allocation wrappers also logs `(pointer, size, backtrace)` to a
   file, gated by an environment variable (zero overhead when off; and it
   fires on *allocation* paths only, which happen at startup, not per pixel).
   The backtrace is resolved to function names/lines offline; the innermost
   frame that isn't allocator plumbing names the **owning data structure**
   (e.g. `SchurComplementBundlerGpu::Impl` → tag `ba_linear_system`). A fixed
   **tag vocabulary** (keyframe_descriptors, pyramid_levels, images_raw,
   feature_tracks, ba_linear_system, …) keeps names consistent.
2. **Layer 2 — driver sidecar.** The NVBit tool independently logs every
   allocation/free the *driver* sees, stamped with the current launch id.
   This catches allocations from libraries cuVSLAM links (CUB — CUDA
   building-block library used for sorting; cuSOLVER — dense linear algebra)
   that Layer 1 can't see, and puts allocation *lifetimes on the same clock
   as the trace* (essential when memory is freed and its address reused by a
   different structure — our tests cover exactly that).
3. **The join.** Stream the address trace in launch order, maintain the set
   of live allocations, and for each *global-space* access look up which
   allocation contains it → its tag. Shared-space accesses are bucketed
   `shared_onchip`; local-space accesses `local_spill` (see §6 — this
   bucketing is the correctness linchpin). Output: per-kernel × per-tag byte
   tables. Anything not matching is `unmapped` — our sanity meter (≤7 %,
   mostly <1 %, and the exceptions are explained — §18).
4. **Coverage audit.** Because traces are *windows* (slices of the run), a
   sparse kernel could be missed. We diff each sequence's full launch map
   against its attributed kernels and iteratively capture gap-fill windows
   until **zero kernels are missing** — with a planner that anchors windows
   on *dense clusters* of the missing kernel's launches (an isolated launch
   drifts between runs; a cluster doesn't).

Why three layers instead of one? **Independent signals cross-check.** Layer 1
knows names but only sees what it wraps; Layer 2 sees everything but knows no
names; the trace shows what kernels actually touch. When they disagree,
you've found a bug *before* a reviewer does — which is exactly what happened,
twice, and both times the disagreement became a finding.

## 15. Reproducibility & artifact evaluation

**Artifact evaluation (AE)**: top venues let reviewers run your code and
stamp the paper with badges (available / functional / results reproduced).
Our posture: every table regenerates from committed CSVs with
standard-library Python (no GPU needed for the analysis layer), 18 GPU-free
tests, all capture scripts committed and resumable. The one administrative
blocker is choosing a repository LICENSE.

---

# PART IV — THE FINDINGS, RETOLD SLOWLY

## 16. The GPU memory budget is *static* (and why that's a big deal)

We journaled every GPU allocation over a full ~2500-frame loop-closing run:
**240 allocations, 108.65 MB, all made at startup, none freed mid-run — and
the identical 240 for a 30-frame run.** cuVSLAM pre-sizes everything.

Budget by structure: images 39.97 MB, pyramids 31.15 MB, BA linear system
24.26 MB, **keyframe descriptors 6.73 MB (fixed!)**, feature tracks 4.27 MB,
depth pyramid 2.21 MB, ICP state 0.07 MB (+ 17.14 MB pinned host memory,
15.41 of which mirrors the BA system).

Why it matters: the keyframe *database* — the thing that grows for the whole
session as the robot maps more space — does **not** grow on the GPU. The GPU
holds a fixed 6.7 MB working buffer; the growing store lives on the **host**
in **LMDB** (Lightning Memory-mapped Database — an embedded key-value store).
Therefore the **ISP (in-storage) opportunity is at the host storage system**,
and we know this from the allocator itself, not from an inference.

## 17. The self-correction arcs — own this story, it will come up

**Arc 1 (counters vs trace, round one).** ncu counters read the loop-closure
scan (`st_track_with_cache`) as a scattered gather (18–30 sectors/request).
Our first NVBit trace analysis said the opposite: 2.1 sectors/warp, 99.4 %
coalesced — and we published internally "the trace overturns the counter
proxy."

**Arc 2 (the attribution join finds the truth).** When we joined traces to
allocations, 88–98 % of most kernels' accesses matched *no allocation at
all*. Impossible — unless the accesses weren't global memory. They weren't:
the tool records **every memory space** (§6). Filtering by opcode showed the
loop-closure scan's traffic is **94 % register-spill** (local space) and its
*global* accesses are **23–30 sectors/warp, 2–6 % coalesced — a scattered
gather, exactly what the counters said.** The earlier "coalesced" reading was
the spill stream (coalesced by construction) drowning the data accesses. We
re-derived every locality table space-filtered, published a dated correction
section, and marked the superseded claims in place.

How to present this to a hostile panel — verbatim if you like:

> "We made a measurement error, our own pipeline caught it, we corrected it
> in a dated, in-place correction, and the corrected measurement now agrees
> with an independent method. Two methods agreeing after a documented
> reconciliation is *stronger* evidence than either alone — and the failure
> mode we found (space-blended GPU traces) is itself a finding that other
> groups using NVBit-style tracing need to know about."

Never be defensive about this. Self-correction on the record is what
distinguishes measurement science from advocacy.

## 18. What each kernel actually touches (the attribution results)

Steady-state numbers (TUM, 28.1 GiB of traced addresses), all reproduced
across the full matrix:

- **`st_track_with_cache`** (the loop-closure scan): 94.2 % local_spill,
  4.9 % keyframe_descriptors, 0.8 % unmapped. Its DRAM *volume* is spill;
  its *data* access is a scattered gather over the descriptor buffer.
- **`st_build_cache`** (keyframe ingest): 93 % keyframe_descriptors.
- **`sba::reduced_system_stage_2`** (solver core): 96.9 % ba_linear_system —
  one named, pre-sized, contiguous structure carrying essentially all of the
  solver's DRAM traffic.
- **Front-end compute kernels** (convolutions, GFTT, sorts): 83–98 % of
  accesses go to **shared-memory tiles** (on-chip, no DRAM); their global
  residue lands exactly on pyramid/image/track structures.
- **NVTX stage map** (measured, not guessed from names):
  `st_track_with_cache` runs inside the "SLAM: LC & optimization" stage —
  it *is* the loop-closure kernel.

At full scale: **48 of 49 kernels get the same top data-structure tag in
every sequence they appear in** (27 sequences, 4 datasets, indoor + outdoor,
RGB-D + stereo). The one exception isn't noise — it's a *constant* ~40 %
unmapped slice in one BA kernel, identical everywhere, consistent with
**static module memory** (`__device__` globals — GPU memory baked into the
binary and allocated by the module loader, invisible to both journal layers
by design) — plus the LK tracker whose image reads go through **texture
units** our tracer cannot see (disclosed lower bound). Bounded, explained,
and namable with one more instrumentation layer if a reviewer insists.

## 19. The spill finding (the new axis nobody expected)

The loop-closure scan spends most of its DRAM bandwidth on **register
spills** — the compiler ran out of registers for the 9-dimensional patch
descriptors and silently shuttles them to DRAM-backed local memory. That is:
*the dominant memory consumer of the most architecturally interesting kernel
is compiler scratch, not the data structure.* Hardware implication: a larger
register file or a dedicated spill-SRAM attacks the volume; near-memory
gather attacks the data-side latency. Methodological implication (for the
community): unfiltered GPU address traces are dominated by spill/tile records
and will poison locality analyses that don't filter by space.

## 20. The taxonomy with its hardware asks (the punchline table)

| class | structures (measured tags) | evidence | substrate ask |
|---|---|---|---|
| **streaming** | images_raw 40 MB, pyramid_levels 31 MB, gradients | flat reuse CDFs 64 KiB→48 MiB (compulsory misses); 1.68 MB/frame H2D upload; 41 % of kernel time is copies | **near-sensor SRAM**: consume pixels before DRAM; caches provably can't help |
| **hot-persistent** | ba_linear_system 24.3 MB (+15.4 MB pinned mirror) | 96–100 % of solver global traffic on one pre-sized structure, every frame | **DRAM-PiM**: bank-level streaming compute over a fixed-size resident structure |
| **cold-persistent** | keyframe DB: 6.7 MB fixed on GPU; growing store host-side (LMDB) | scan = scattered gather, footprint grows 0.46→1.09 MB room→street, 10–33 % turnover/scan; DB growth measured host-side | **ISP** at the host store; on-GPU: spill-SRAM (volume) + near-memory gather (latency) |

---

# PART V — FACING THE PANEL

## 21. How to answer a hostile question (the technique)

1. **Agree with the true part first.** "You're right that we measure one
   workload family." Never contest a fact that is a fact.
2. **Bound it.** Say precisely how far the concern reaches and where it
   stops. "That limits external validity across SLAM systems, not the
   internal validity of any number in the paper."
3. **Point at evidence.** Cite the specific measured number and where it
   lives. You have a repo where every table regenerates from committed data;
   say so.
4. **Show the plan or the disclosure.** Either it's in the limitations
   section, or it's the next experiment with a concrete design.
5. Keep answers under ~45 seconds. If they want more, they'll ask.

Words that serve you: *"measured," "bounded," "disclosed," "reproduces
from committed data."* Words to avoid: *"obviously," "we believe,"
"probably," "trust me."*

## 22. The attack drill — 18 likely strikes and strong answers

**Q1. "Your own report first said the loop-closure kernel was coalesced,
then you reversed it. Why should we trust anything here?"**
A: Because the reversal is the strongest part of the record. The counter
method and the trace method initially disagreed; our attribution join
localized why — the trace blended memory spaces, and 94 % of that kernel's
records were register spills, which are coalesced by construction. After
space filtering, the trace says 23–30 sectors/warp — matching the counters'
18–30 independently. The correction is dated, in place, and both methods now
agree. A pipeline that can't catch its own errors is the one you shouldn't
trust.

**Q2. "One workload. Is cuVSLAM representative of SLAM, let alone Physical
AI?"**
A: cuVSLAM is a production system shipping in NVIDIA Isaac — not a research
prototype — and we characterize it across four public datasets, two camera
modalities, indoor and outdoor, loop and loop-free, 27 sequences with 0
failures. Within that space, per-kernel class consistency is 91 % and
data-structure attribution is unanimous for 48/49 kernels — the behavior is
a property of the algorithms, not the input. Claims beyond visual SLAM are
framed as hypotheses, and the pipeline structure we exploit (streaming
front-end / iterative solver / growing database) is shared by most SLAM
designs.

**Q3. "Everything is on one desktop GPU. Robots use edge devices."**
A: Correct, and disclosed. The characterization is architecture-*aware* but
the core claims are architecture-*independent* by construction: reuse-
distance CDFs and address-stream attribution don't depend on cache sizes of
the machine measured on. The hardware-parameterized harness has a Jetson
Orin descriptor ready; that re-run is scheduled work, and unified memory on
Orin will make the transfer findings more interesting, not less.

**Q4. "Isn't the spill dominance just a compiler artifact? Recompile with
more registers per thread and your headline disappears."**
A: The spill *is* a compiler decision — that's the point of reporting it: it
is invisible to counter-level studies and it dominates the DRAM traffic of a
key kernel in a shipped binary. Raising the register budget trades occupancy
(parallelism) for spill; whether that trade wins is a scheduling question we
can now pose precisely because we quantified the spill stream separately
from the data stream. Either way the *data-side* result — a scattered gather
over the descriptor buffer — is unaffected.

**Q5. "Your tracer can't see texture fetches. How much traffic are you
missing?"**
A: Disclosed as a lower bound on image-structure traffic. Bounded two ways:
ncu's texture-unit counters (a different, complete mechanism) cover those
kernels' totals, and the affected kernels are the front-end streaming class,
where the near-sensor argument rests on the flat reuse CDFs and the measured
1.68 MB/frame upload — neither of which the TEX path changes.

**Q6. "40 % of one BA kernel's traffic is 'unmapped.' That's a hole."**
A: It's the same ~40 % in all 27 sequences — a systematic residual, not
noise. It is consistent with static module memory: `__device__` globals
allocated by the module loader, invisible to both a source-level journal and
a `cuMemAlloc` hook *by construction*. It's bounded, it's explained, and one
more layer (kernel-argument correlation via `cuModuleGetGlobal`) names it if
required. Total unmapped elsewhere is under 1–7 %.

**Q7. "You traced windows, not whole runs. What did you miss?"**
A: We audited exactly that: every sequence's full launch map (cheap,
uninstrumented pass) diffed against the attributed kernels, then gap-fill
windows captured iteratively until **zero kernels were missing in all 27
sequences**. The audit and planner are committed tools, and the coverage
matrix is a committed CSV.

**Q8. "Windowed traces bias toward steady-state behavior."**
A: The window *set* is stratified by design: cold-start windows, mid-run
steady state, late loop-closure-dense windows, plus name-filtered captures
of every keyframe-database scan. Rare-path kernels are covered by
construction, and the no-loop KITTI sequences bracket the other extreme.

**Q9. "k-means with your own features finding your own class count is
circular."**
A: The decision tree and the clustering use the same measurements but
independent structure: the tree encodes thresholds from the DAMOV
methodology; k-means knows nothing about them. That an unsupervised sweep
prefers k = 7–8 — with silhouette, not our labels, doing the choosing — is
the non-circular part. Purity 0.68 / ARI 0.30 is reported honestly as
moderate agreement: the clusters recover the classes' geometry, not our
labels verbatim.

**Q10. "61 % of GPU time 'carries PiM affinity' — that sounds like
marketing."**
A: It's a defined, reproducible quantity: the share of locked-clock GPU time
in kernels whose class is strong- or conditional-PiM under the G-taxonomy,
with the ±25 % threshold sensitivity published. The conditional share (40 %)
is separated from the strong share (21 %) precisely so nobody reads it as a
speedup promise. It sizes the *opportunity*, the substrate evaluation prices
it — that's the next paper.

**Q11. "No energy numbers, yet PiM's selling point is energy."**
A: Correct and disclosed. Whole-run wall power via NVML is being added
(cheap); per-kernel energy needs AccelWattch and belongs to the simulation
phase, which this characterization feeds. Note the thesis claims are about
*traffic placement*, which is measured; the energy delta is the follow-on
paper's dependent variable.

**Q12. "Your ISP claim is about host storage, but you measured the GPU."**
A: The GPU measurement is what *establishes* it's a host-storage problem:
the allocator journal proves GPU-side keyframe state is a fixed 6.7 MB while
the session-scale store grows in LMDB on the host. The host-side I/O
characterization (DB growth rate, bytes per loop-closure scan) is the
identified next measurement — a few days of work with standard OS tools,
already scoped.

**Q13. "Instrumentation changes the system you measure (Heisenberg)."**
A: Three mitigations, all verifiable: the journal is env-gated and fires
only on allocation paths (which happen at startup — zero per-frame work);
timing numbers come exclusively from uninstrumented locked-clock runs;
and the instrumented build's allocation structure is bit-identical across
runs and run lengths, so we can A/B against the baseline wheel, which we
kept.

**Q14. "How do you know your address-to-structure mapping is even
correct?"**
A: Three independent signals reconcile: 274/274 source-journal allocations
matched driver-sidecar events; unmapped global traffic after the join is
under 1 % for most kernels; and the per-kernel results are unanimous across
27 sequences. Plus unit tests fabricate address-reuse edge cases (same
address, different structure over time) and the join resolves them via
launch-id lifetimes.

**Q15. "Why didn't you compare against ORB-SLAM3 / another SLAM?"**
A: Scope discipline. A second system would dilute depth for breadth without
changing the thesis: the taxonomy is defined by lifetime/touch-pattern
classes that any keyframe-based SLAM has. We position that as external-
validity discussion, not as measured claim.

**Q16. "The keyframe descriptor buffer is 6.7 MB — that fits in L2. Why
does it need any special hardware?"**
A: Per-scan, yes — and we say so: each scan is nearly L2-resident. The ask
comes from the *session* axis: the working set grows with map size (0.46 →
1.09 MB per scan from room to street already) and turns over 10–33 % per
scan; the union over a deployment is the whole database, which lives in
host storage. The GPU-side asks are the spill volume and gather latency;
the capacity story is at the store — three different mechanisms, each
matched to its evidence.

**Q17. "Your windows were picked from one run and applied to another —
runs differ."**
A: Yes — launch-id drift between runs is real and we hit it: sparse kernels
shifted out of planned windows. That's why the coverage audit exists and
why the gap-fill planner anchors on dense launch *clusters* rather than
isolated occurrences. The audit-to-zero-missing loop is the systematic
answer, and it's committed and rerunnable.

**Q18. "What exactly is novel here? Profilers exist."**
A: Four things, in order: (1) first per-*data-structure* memory attribution
of a production SLAM system on GPU — from allocation backtraces joined to
address traces, not from kernel names; (2) the measured persistence taxonomy
with clustering validation across 27 sequences; (3) the space-filtering
result — spill/tile contamination of GPU address traces — which corrects a
methodology others are using right now; (4) the toolchain itself, portable
and artifact-ready. "Profilers exist" is precisely the point: profilers
answered *how much*; none of them answered *which data*.

## 23. Numbers to know cold (flashcards)

| number | what it is |
|---|---|
| **27 / 4 / 0** | sequences / datasets / failures in the campaign |
| **91 %** | cross-sequence kernel class consistency (42/49 ≥ 80 %, 24/49 unanimous) |
| **48/49** | kernels with unanimous top data-structure tag |
| **k = 7–8** | k-means silhouette optimum ≈ number of G-classes |
| **61 %** | GPU time with PiM affinity (21 strong + 40 conditional) |
| **0.14 % vs 49.6 %** | locked vs unlocked run-to-run CoV |
| **205.0 GB/s / 5445 GFLOP/s** | measured DRAM / FP32 ceilings at lock |
| **240 / 108.65 MB** | GPU allocations / total, static for whole run |
| **6.73 MB** | fixed GPU keyframe-descriptor buffer |
| **94 % / 5 %** | st_track: spill share / descriptor share |
| **23–30 vs ~2** | sectors/warp: st_track global (scatter) vs spill (coalesced) |
| **0.46 → 1.09 MB, J 0.67 → 0.90** | scan footprint & Jaccard, room → street |
| **1.68 MB/frame, 41 %** | H2D sensor upload; copy share of kernel time |
| **88–98 % → <1–7 %** | unmapped before/after memory-space filtering |
| **28.1 GiB** | addresses in the TUM steady-state window trace |

## 24. What is honestly not done (say it before they do)

1. Substrate-side evaluation (simulation deltas, energy) — the architecture
   paper's job; every input artifact exists.
2. Host-storage characterization of the LMDB database — scoped, days of
   work, arms the ISP leg with direct I/O numbers.
3. Edge-device (Jetson Orin) re-run — harness ready.
4. Naming the static-memory residual — optional Layer-3 refinement.
5. Energy — NVML whole-run imminent; per-kernel with AccelWattch later.
6. LICENSE for artifact evaluation — administrative, pending.

---

# PART VI — GLOSSARY (A–Z)

- **AccelWattch** — GPU power model integrated with Accel-Sim; gives
  per-kernel energy estimates in simulation.
- **Accel-Sim** — the standard academic GPU simulator; replays real traces
  through configurable GPU models (we'll use it for the PiM/NDP deltas).
- **addr2line** — tool converting a code address into `function, file:line`
  using debug info.
- **Ada (Lovelace)** — NVIDIA GPU generation of our RTX 2000 Ada (sm_89).
- **AE — Artifact Evaluation** — reviewers reproduce your results from your
  code/data; badges awarded.
- **ARI — Adjusted Rand Index** — chance-corrected agreement between two
  clusterings (0 random, 1 identical).
- **ASLR** — address-space layout randomization; OS loads code at random
  addresses per run; we log memory maps so backtraces stay resolvable.
- **BA — Bundle Adjustment** — nonlinear least-squares over poses and
  landmarks; SLAM's numerical core.
- **backtrace** — the chain of function calls active at a point in
  execution.
- **CDF** — cumulative distribution function; fraction of samples ≤ x.
- **coalescing** — hardware merging of a warp's 32 addresses into few
  memory sectors; the efficiency axis of GPU memory access.
- **CoV** — coefficient of variation, std/mean.
- **CUB** — NVIDIA's CUDA building-block library (sorting etc.); allocates
  its own scratch (caught by Layer 2).
- **cuSOLVER** — NVIDIA dense linear-algebra library (the small pose solves).
- **DAMOV** — 2021 landmark data-movement characterization methodology we
  adapted from CPU to GPU.
- **DRAM** — main memory (GDDR6 on our GPU).
- **DWARF** — debug-information format inside binaries; what addr2line
  reads; kept by building `RelWithDebInfo`.
- **EuRoC** — drone (micro-aerial-vehicle) stereo+IMU dataset.
- **GFTT** — Good Features To Track (Shi–Tomasi corners).
- **global memory** — GPU DRAM address space for data (`LDG`/`STG`).
- **grid-launch id** — running counter of kernel launches; our trace clock.
- **H2D / D2H** — host-to-device / device-to-host copies over PCIe.
- **HPCA / ISCA / MICRO / ASPLOS** — the four top computer-architecture
  conferences.
- **IISWC / ISPASS** — the two main workload-characterization /
  performance-analysis venues.
- **IMU** — inertial measurement unit (gyro + accelerometer).
- **ISP** — in-storage processing (compute inside the SSD). *(In camera
  literature ISP also means "image signal processor" — different thing;
  ours is storage.)*
- **Jaccard** — set overlap |A∩B|/|A∪B|.
- **keyframe** — frame promoted into the long-term map/database.
- **KITTI** — automotive stereo benchmark suite.
- **k-means / silhouette / purity** — see §13.
- **LDG/STG, LDS/STS, LDL/STL** — GPU load/store opcodes for
  global / shared / local space; how we filter spaces.
- **LK** — Lucas–Kanade optical flow (feature tracking).
- **LMDB** — Lightning Memory-mapped Database; host-side store where the
  landmark/keyframe database grows.
- **local memory** — DRAM-backed per-thread space used for register spills;
  coalesced by construction; tagged `local_spill`.
- **loop closure** — recognizing a previously visited place and correcting
  accumulated drift.
- **LRU / stack distance** — least-recently-used; reuse distance measured in
  distinct items, predicts hits for any capacity.
- **ncu (Nsight Compute)** — per-kernel hardware-counter profiler.
- **NDP** — near-data processing (umbrella for PiM/PnM).
- **nsys (Nsight Systems)** — timeline profiler.
- **NVBit** — NVIDIA Binary Instrumentation Tool; per-instruction hooks →
  address traces.
- **NVML** — NVIDIA Management Library (power/clock queries; our whole-run
  energy source).
- **NVTX** — NVIDIA Tools Extension; named ranges visible in nsys; cuVSLAM's
  own annotations, which we enabled.
- **PiM / PnM** — processing in / near memory (compute at DRAM banks).
- **pinned (page-locked) memory** — host RAM the GPU can DMA directly;
  device-visible under UVA; tagged `:host` in our tables.
- **pyramid** — multi-resolution image stack for coarse-to-fine tracking.
- **RGB-D** — color + depth camera.
- **roofline** — performance bound model: min(compute roof, bandwidth ×
  arithmetic intensity).
- **Schur complement** — matrix elimination trick making BA tractable.
- **sector** — 32-byte memory access granule; 4 per 128-B cache line.
- **shared memory** — on-chip per-SM scratchpad (`shared_onchip` tag; not
  DRAM).
- **SLAM** — simultaneous localization and mapping.
- **SM** — streaming multiprocessor, the GPU's core unit.
- **spill (register spill)** — compiler moving excess thread variables to
  local memory.
- **TUM RGB-D / TUM-VI** — handheld indoor RGB-D / visual-inertial datasets.
- **UVA** — unified virtual addressing; one address space spanning host and
  device (why pinned host memory is kernel-visible).
- **warp / lane** — 32 threads executing in lockstep / one slot within.
- **zstd** — fast compression; our traces stream through it.

---

*Final advice: the panel is not the enemy of the thesis; sloppiness is. Every
question in §22 has a measured number behind it, and every number lives in a
file a reviewer can regenerate. Answer from the evidence, concede the
boundaries, and let the self-correction story do what it does — prove that
when this pipeline is wrong, it finds out first.*
