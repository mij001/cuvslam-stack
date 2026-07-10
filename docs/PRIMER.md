# The cuVSLAM Memory-Characterization Project — a complete primer

**Read this one document and you will understand the whole project: the theory,
what was built, what was measured, what it means, and how to defend it.** It
assumes *no* prior knowledge. Every technical word is defined the first time it
appears. It builds up in order — earlier parts are needed for later ones — so
read it top to bottom the first time.

If you only remember one sentence: *we measured, kernel by kernel, exactly how a
production robot-vision program moves data on a GPU, and used that to say which
piece of computation would be better run on a different kind of hardware (memory
that can compute, a camera chip, or the CPU) — and we proved the measurements
are trustworthy.*

---

## Table of contents

- Part 0 — The thesis in one paragraph (the destination)
- Part 1 — Foundations: computers, memory, and why data movement is the enemy
- Part 2 — The problem: robot vision, and the idea of "heterogeneous" hardware
- Part 3 — The intellectual backbone: DAMOV, adapted to GPUs
- Part 4 — The tools: how you actually measure a GPU
- Part 5 — The metrics: every number, its formula, and what it means
- Part 6 — The method: the six-step pipeline, and the decision tree
- Part 7 — Trust: how we proved the numbers are real
- Part 8 — The findings, one by one (what we actually learned)
- Part 9 — What was built (the software)
- Part 10 — Limits, roadmap, and what's next
- Part 11 — Defending it: the questions you will be asked, with answers
- Glossary — every term in one place

---

## Part 0 — The thesis in one paragraph (the destination)

Modern robots ("**Physical AI**" — machines that sense and act in the real
world) run their vision software on a **GPU** (Graphics Processing Unit — a chip
with thousands of small compute cores, originally for graphics, now used for any
massively parallel work). The specific program we study is **cuVSLAM**, NVIDIA's
production **Visual SLAM** system (SLAM = *Simultaneous Localization And
Mapping* — figuring out where the camera is *and* building a map, at the same
time, from camera images). The deep observation behind this project is that on
modern hardware the bottleneck is usually **not** doing arithmetic — it is
**moving data** between memory and the compute cores. That movement costs time
and, even more, **energy** (a robot runs on a battery). One proposed fix is to
stop moving the data so far: put small compute engines *inside or next to the
memory* — **Processing-in-Memory (PiM)** — or *inside the camera/image chip* —
**Image Signal Processor (ISP)** near-sensor computing. But you cannot redesign
hardware for a program you have not measured. So this project **characterizes**
(carefully measures and classifies) cuVSLAM's memory behaviour, per **kernel**
(one GPU function launch), across many datasets, and produces a per-kernel
verdict: *keep this on the GPU, or move it to PiM / ISP / CPU — and here is the
evidence*. Everything below builds the vocabulary and the method to make that
paragraph fully meaningful and defensible.

---

## Part 1 — Foundations: computers, memory, and why data movement is the enemy

### 1.1 CPU vs GPU

A **CPU** (Central Processing Unit) is the general-purpose "brain" of a computer:
a few very fast, very clever cores that do one thing after another very quickly.
A **GPU** is the opposite trade-off: **thousands** of simpler, slower cores that
all do work **at the same time** (in **parallel**). If you must add one million
pairs of numbers, a CPU does them almost one at a time (fast, but sequential); a
GPU does thousands at once. Camera images are grids of millions of pixels, and
you often do the *same* operation to every pixel — a perfect fit for a GPU. This
is why robot vision runs on GPUs.

NVIDIA GPUs are programmed with **CUDA** (Compute Unified Device Architecture —
NVIDIA's system for writing GPU programs). A **kernel** is one function you
launch on the GPU; when it runs, thousands of copies execute in parallel.

### 1.2 Threads, warps, blocks, SMs

- A **thread** is one instance of the kernel's work (e.g. "process pixel #5031").
- Threads run in fixed groups of **32 called a warp**. The 32 threads in a warp
  execute the *same instruction at the same time* (on different data). This
  matters a lot later.
- Warps are grouped into **thread blocks**, and blocks run on a hardware unit
  called an **SM** (Streaming Multiprocessor). A GPU has many SMs (the desktop
  GPU we used, an NVIDIA RTX 2000 Ada, has **22 SMs**).

### 1.3 Memory: the hierarchy, from tiny-and-fast to huge-and-slow

Computers store data in a **hierarchy**. Small memories are fast but tiny; big
memories are slow but huge. Data lives in the big slow memory and is copied up
to the small fast ones to be worked on. From fastest/smallest to slowest/biggest,
on a GPU:

1. **Registers** — a handful of values *inside* each thread. Fastest possible.
2. **Shared memory / L1 cache** — a small scratchpad *inside each SM* (on our
   GPU, 100 KB per SM). "**Cache**" = an automatic fast copy of recently-used
   data. "L1" = level-1, the first cache. **Shared memory** is a programmer-
   controlled part of the same on-chip storage.
3. **L2 cache** — a bigger cache (25 MB on our GPU) shared by *all* SMs.
4. **DRAM** (Dynamic Random-Access Memory), also called **VRAM** or "global
   memory" — the GPU's main memory (16 GB on our GPU). Big, but ~100× slower to
   reach than registers. **This is the memory whose traffic we care about most.**
5. **Host RAM** — the CPU's main memory, off the GPU chip entirely. Reaching it
   from the GPU means crossing the **PCIe** bus (the cable-like link between CPU
   and GPU), which is slow.
6. **Storage** (SSD/disk) — permanent, huge, slowest.

Key idea: **the further down this list data must travel, the more time and
energy it costs.** A number in a register is essentially free; the same number
fetched from DRAM costs orders of magnitude more energy. Fetching from host RAM
or storage is worse still.

### 1.4 Bandwidth vs latency (two different "slow"s)

- **Latency** = how long you wait for *one* piece of data to arrive (like the
  delivery time for a single parcel).
- **Bandwidth** = how much data you can move *per second* once it is flowing
  (like the width of the highway). Measured in **GB/s** (gigabytes per second).

A program can be limited by either. GPUs hide **latency** by having thousands of
warps ready: while one warp waits for data, the SM runs another (this is what
**occupancy**, §5.6, measures). But no trick hides a **bandwidth** limit — if
the DRAM highway is full, you simply wait.

### 1.5 The memory wall and the "von Neumann bottleneck"

Over decades, compute got fast much faster than memory got fast. The result is
the **memory wall**: on modern chips, many programs spend most of their time
*waiting for data*, not computing. The classic name for "compute is separate
from memory, so everything must shuttle across a narrow link" is the **von
Neumann bottleneck** (after the computer architecture nearly all machines use).

This is the single fact the whole project rests on: **if data movement, not
arithmetic, is the bottleneck, then the way to go faster and use less energy is
to move data less — which may mean changing the hardware, not the software.**

### 1.6 Coalescing (why *how* a warp touches memory matters)

Recall a warp is 32 threads acting together (§1.2). When those 32 threads read
memory, the hardware fetches data in fixed 32-byte chunks called **sectors**. If
the 32 threads read 32 neighbouring locations, the hardware satisfies them with
a few sectors — efficient. This is **coalesced** access. If the 32 threads read
32 *scattered* locations, the hardware must fetch many sectors, most of whose
bytes are thrown away — a **scattered gather**, very wasteful of bandwidth. The
number **sectors per request** measures this (4 ≈ perfectly coalesced; 32 ≈ fully
scattered). Coalescing is a *GPU-specific* concern that does not exist on CPUs,
and it turns out to matter enormously here.

### 1.7 Register spill (a subtle but crucial idea)

Each thread has only a limited number of registers (§1.3). If a kernel needs more
temporary values than there are registers, the compiler **spills** the extras to
a per-thread slice of DRAM called **local memory**. So "**register spill**" is
DRAM traffic that is *not the program's data at all* — it is compiler scratch
paper, a symptom of register pressure. Distinguishing spill traffic from real
data traffic is a key contribution of this project (see F6, F9).

---

## Part 2 — The problem: robot vision, and the idea of "heterogeneous" hardware

### 2.1 SLAM and cuVSLAM

A robot with a camera needs to answer two questions continuously: *where am I?*
(**localization**) and *what does the world look like?* (**mapping**). Doing both
at once, from images, is **Visual SLAM**. The pipeline, roughly:

1. **Front-end / feature detection** — find distinctive points ("features",
   e.g. corners) in each new image.
2. **Tracking** — match features between consecutive frames to estimate how the
   camera moved.
3. **Bundle adjustment (BA)** — a big math optimization that refines the camera
   poses and 3-D point positions so everything is geometrically consistent.
4. **Loop closure** — recognise a previously-seen place ("I've been here
   before") and correct accumulated drift; this needs a growing database of past
   keyframes.

**cuVSLAM** is NVIDIA's production, GPU-accelerated implementation of this,
shipped inside their Isaac robotics stack. We treat it as the **workload** — the
real program we measure. We do not modify its algorithm; we observe it.

### 2.2 Accuracy: how we know the vision is *correct*

A characterization is meaningless if the program was producing garbage
trajectories. So we check cuVSLAM's output against **ground truth** (the true
camera path, recorded by an external system). Two standard error measures:

- **APE / ATE** (Absolute Pose/Trajectory Error) — after best-aligning the
  estimated path to the true path, the average distance between them, in metres.
  "How far off is the whole trajectory."
- **RPE / RTE** (Relative Pose/Trajectory Error) — the error over short segments,
  as a percentage. "How much does it drift per metre travelled."

Lower is better. We compare to the numbers NVIDIA published in the cuVSLAM paper
(arXiv:2506.04359) to confirm our runs reproduce their accuracy (Part 7, F13).

### 2.3 Heterogeneous computing and the candidate substrates

"**Heterogeneous**" computing means using *different kinds* of hardware for
different jobs, instead of forcing everything onto one. The candidate places
("**substrates**") a piece of computation could live are:

- **GPU** — the default; great when there is real arithmetic and reuse.
- **CPU** — better for tiny, irregular, or barely-parallel work not worth a GPU
  launch.
- **PiM (Processing-in-Memory)** — small compute engines built *into the DRAM
  chips themselves*, so bandwidth-hungry work runs next to the data instead of
  shuttling it to the SMs. Two flavours we distinguish:
  - **PiM near-bank** — for *streaming, bandwidth-bound* work (read a lot,
    compute a little). A DRAM chip is organised into **banks**; putting compute
    at each bank gives enormous aggregate internal bandwidth.
  - **PiM scatter** — for *irregular gather/scatter* access (§1.6) that defeats
    normal memory systems; a scatter-capable engine near memory helps.
- **ISP / near-storage** — an **Image Signal Processor** is the chip that
  processes raw camera output. "Near-sensor / near-storage" computing does work
  right where data is born (the camera) or stored (the keyframe database), so it
  never has to travel to the GPU at all.
- **Near-sensor SRAM** — a small fast on-camera memory to consume streaming
  image data before it ever reaches DRAM.

The project's central deliverable is: **for each kernel of cuVSLAM, which of
these is the best home, and what is the evidence?**

### 2.4 Why "per kernel" and why energy

Different kernels behave completely differently — one is compute-bound, another
streams data, another does a scattered database scan. A single verdict for "the
whole program" would be useless. And the reason to move work off the GPU is often
**energy**, not just speed: moving a byte from DRAM costs far more energy than
computing on it, and a battery-powered robot cares about joules. (A **joule** is
the unit of energy; **watts** are joules per second — power.)

---

## Part 3 — The intellectual backbone: DAMOV, adapted to GPUs

### 3.1 DAMOV (the origin, on CPUs)

**DAMOV** ("Data Movement" — a well-known 2021 methodology by Oliveira et al.) is
a systematic way to answer, on **CPUs**, "which parts of a program are limited by
data movement, and would benefit from Processing-in-Memory?" Its recipe:

1. **Screen** — find the functions that actually matter (consume real time).
2. **Characterize** — measure each with hardware counters and, crucially, a
   *machine-independent* locality analysis (how much data reuse there is).
3. **Classify** — sort functions into a small number of **bottleneck classes**
   (e.g. "bandwidth-bound", "cache-bound", "compute-bound"), *derived from the
   data by clustering*, not asserted.

The core DAMOV insight is a metric called **LFMR** (Last-to-First Miss Ratio):
of the memory requests that miss the first cache, what fraction also miss the
*last* cache and go all the way to DRAM? If that fraction is near 1, the big
caches are useless for this code — a strong sign PiM would help.

### 3.2 Why it does not port to GPUs unchanged

GPUs differ from CPUs in three ways that break DAMOV's CPU assumptions:

1. **Two cache levels, not three** (L1/L2, no L3) — so LFMR is redefined as
   *L2-miss / L1-miss*, one level down.
2. **Latency is hidden by parallelism** — a CPU stalls on a cache miss; a GPU
   with high **occupancy** just runs another warp. So "memory-bound" on a GPU
   must account for occupancy, not just miss rates.
3. **Coalescing exists** (§1.6) — a uniquely GPU way to waste bandwidth, with no
   CPU analog.

This project's methodological contribution is **GPU-DAMOV**: DAMOV re-derived for
GPUs, with these three adaptations, *plus* an extension from "is it memory-bound"
to "**which substrate** — GPU / CPU / PiM / ISP — and if it stays on the GPU,
what is the fault to fix." Applying it to a real production V-SLAM stack (not a
toy benchmark) is the empirical contribution.

### 3.3 The GPU bottleneck taxonomy (G0–G7)

Every kernel is placed in one of eight classes. You will meet the exact rules in
Part 6; here is the vocabulary:

| class | one-line meaning | PiM verdict |
|---|---|---|
| **G0** no-signal | too small / no dominant bottleneck | n/a (maybe CPU) |
| **G1** bandwidth-bound | DRAM highway saturated | **PiM: strong** |
| **G2** coalescing-bound | scattered access wastes bandwidth | **PiM: conditional** |
| **G3** L2-reuse-bound | the L2 cache *is* helping | PiM: weak (keep the cache) |
| **G4** latency-bound | stalls at low occupancy waiting on memory | PiM: strong if cache-defeating |
| **G5** compute-bound | actually doing arithmetic | PiM: none (keep on GPU) |
| **G6** on-chip-bound | limited by shared-memory/on-chip units | PiM: none |
| **G7** dependency-bound | stalls on instruction dependencies, not memory | PiM: none (fix occupancy first) |

G7 was **not** in the original hypothesis — it emerged from the actual cuVSLAM
data (some kernels stall waiting on dependency chains at low occupancy, with
memory idle). That the taxonomy grew from the data is a feature, exactly as
DAMOV prescribes.

---

## Part 4 — The tools: how you actually measure a GPU

You cannot see inside a running GPU by eye. Three complementary tools give three
views; a fourth tags the data structures.

### 4.1 Nsight Systems ("nsys") — the timeline

**nsys** records a **timeline** of the whole run: when each kernel ran, for how
long, and when data was copied between CPU and GPU. From it we get each kernel's
**share of GPU time** — the "who matters" screen. It has near-zero overhead, so
it does not disturb the run. It also reads **NVTX ranges** (see §4.4).

### 4.2 Nsight Compute ("ncu") — the microscope

**ncu** measures **hardware performance counters** — thousands of tiny event
tallies inside the GPU (bytes read from DRAM, cache hits, warp stalls, etc.). To
read counters accurately ncu uses **kernel replay**: it runs a kernel many times,
collecting a few counters each time, because collecting all at once is
impossible. This makes ncu *slow*, so we:

- profile only a **bounded window** of launches (not the whole run), and
- use a **curated metric set** (~15–30 chosen counters), never `--set full`
  (which asks for everything and gets killed on a small GPU before writing a
  report).

Every metric in Part 5 comes from ncu counters.

### 4.3 NVBit — the ground truth of every memory address

**NVBit** (NVIDIA Binary Instrumentation Tool) rewrites the GPU machine code to
record **every single memory address** each warp touches. Our tool, **mem_trace**,
records global-memory loads/stores (the DRAM-visible data), giving the exact
addresses — the ground truth behind ncu's summary counters. It is extremely slow
(100–1000× the normal run), so it too is **windowed** (a patch bounds which
launches and which kernel it instruments). From the traces we compute *locality*
(how much data is reused, §5.7) and *attribution* (§4.4).

### 4.4 NVTX ranges and the TaggedAllocator — naming things

Two problems: (a) which kernel belongs to which *pipeline stage* (feature
detection vs tracking vs loop closure), and (b) which *data structure* (image,
keyframe database, BA matrix) a kernel's memory traffic actually touches.

- **NVTX** (NVIDIA Tools Extension) lets a program annotate its own code with
  named ranges ("we are now in loop closure"). We compiled cuVSLAM with its own
  NVTX annotations on; nsys then reports a **measured** kernel→stage table — so
  stage membership is *measured, not guessed from kernel names*.
- The **TaggedAllocator** is a small piece of instrumentation we injected that
  **journals every memory allocation** cuVSLAM makes: its size, a **backtrace**
  (the chain of function calls that requested it, so we can name it), and a data-
  structure **tag** (`keyframe_descriptors`, `ba_linear_system`, `images_raw`…).
  Joining the NVBit address trace against this journal tells us, for each byte of
  DRAM traffic, *which named data structure it belongs to*. This is what lets us
  say "92.6% of this kernel's traffic is register spill, the rest is keyframe
  descriptors" as a *measured* statement (F6, F9).

### 4.5 Why locked clocks, measured ceilings, and repeats

Measurements must be trustworthy:

- **Locked clocks** — a GPU normally changes its speed constantly (to save power/
  heat), which makes timing noisy. We lock the clocks (graphics 1620 MHz, memory
  7001 MHz) so every measurement is a *stable quantity*. Result: run-to-run
  variation (**CoV**, coefficient of variation = standard deviation ÷ mean) drops
  from **49.6% (unlocked) to 0.14% (locked)**.
- **Measured ceilings** — the "roofline" (§5.5) needs the GPU's true peak
  bandwidth and compute. We *measure* these (205 GB/s DRAM, 5445 GFLOP/s) rather
  than quote the marketing spec, and label which is which.
- **Repeats and sensitivity** — every classification is stress-tested (Part 6.4)
  and cache-cold vs cache-warm effects are bracketed, so no headline rests on a
  fragile number.

---

## Part 5 — The metrics: every number, its formula, and what it means

Each metric turns raw ncu counters into one interpretable feature. A rule used
everywhere: **sum extensive quantities (time, bytes, instructions) across a
kernel's launches first, then form ratios once** — never average ratios (that
would weight a tiny launch the same as a huge one).

### 5.1 Speed-of-Light (SoL): how "full" a resource is

**Speed-of-Light** = a resource's utilisation as a percentage of its theoretical
peak. Three of them:

- **Memory SoL** — how full the memory *pipeline* is.
- **Compute SoL** — how full the arithmetic units are.
- **DRAM SoL** — how full the DRAM *bandwidth* (the highway, §1.4) is.

Reading: DRAM SoL ≥ 50% means the DRAM highway is saturated → bandwidth-bound
(class G1). Compute SoL clearly higher than memory SoL → compute-bound (G5).

### 5.2 LFMR (the DAMOV metric, GPU version)

    LFMR = 1 − (L2 hit rate)

**L2 hit rate** = fraction of L2 accesses that found their data in the L2 cache.
So LFMR = the fraction that *missed* and went to DRAM. **LFMR ≈ 1** → the L2 is
not helping, the data is cache-defeating → PiM-favourable. **LFMR ≤ 0.35** → the
L2 is absorbing the reuse → keep it on the GPU (a bigger cache, not PiM, wins).

### 5.3 MPKI (memory intensity)

    MPKI = (DRAM bytes ÷ 32) ÷ (warp-instructions ÷ 1000)

"Misses Per Kilo-Instruction": how many DRAM sector-fetches (32-byte chunks) the
kernel causes per thousand instructions. High MPKI = memory-intensive.
(*Trap:* the instruction count is **warp**-instructions, 32× fewer than thread-
instructions; getting this wrong is a common error we avoid.)

### 5.4 Arithmetic Intensity (AI) — the key ratio

    AI = FLOPs ÷ DRAM bytes

A **FLOP** is one floating-point operation (an add or a multiply). We count them
the standard way: adds + multiplies + 2×(fused multiply-adds), because a fused
multiply-add does two operations at once. **AI = how much arithmetic you do per
byte you fetch from DRAM.** Low AI (few ops per byte) = memory-bound; high AI =
compute-bound. This single number places a kernel on the **roofline** (§5.5).

*(Generalization note: for non-cuVSLAM workloads the FLOP numerator can be
integer, half-precision, or double-precision ops instead of FP32; the tool picks
the dominant type automatically, so the same roofline works for a neural network
or a database kernel.)*

### 5.5 The Roofline model (a picture that decides bound-ness)

The **roofline** is a graph. The x-axis is AI (arithmetic intensity, §5.4); the
y-axis is achieved performance (GFLOP/s). The "roof" has two parts:

- a **slanted line** on the left = the DRAM bandwidth limit (performance you can
  possibly reach is AI × bandwidth), and
- a **flat line** on the right = the compute peak.

A kernel is a point under the roof. If it sits under the *slanted* part (low AI),
it is **memory-bound** — no amount of faster arithmetic helps; only more
bandwidth (or less traffic — i.e. PiM) does. If it sits under the *flat* part, it
is **compute-bound**. The roofline is the single clearest way to see, per kernel,
whether the memory wall (§1.5) is the limit. (The dashboard draws this
interactively, with the measured roof.)

### 5.6 Occupancy (how well latency is hidden)

**Occupancy** = the fraction of the GPU's maximum concurrent warps that are
actually active. It is the GPU's main way to hide memory **latency** (§1.4):
plenty of warps means the SM always has work while some warps wait for data.
**Low occupancy (< 25%)** means latency *cannot* be hidden — the kernel stalls.
Combined with the stall reason (§5.7), low occupancy separates "waiting on
memory" (class G4) from "waiting on instruction dependencies" (class G7).

### 5.7 The stall taxonomy (why a warp is not making progress)

When a warp is not issuing an instruction, the hardware records *why* (the
**stall reason**). The important ones:

- **long_scoreboard** — waiting on a slow memory (DRAM/L2) load → *memory* stall.
- **short_scoreboard** — waiting on fast shared memory → *on-chip* stall.
- **lg/mio/tex_throttle** — a queue to the memory/texture units is full → also
  *memory-side* pressure.
- **wait** — waiting on a fixed-latency instruction dependency (e.g. a
  multiply's result) → *dependency* stall, **not** memory.
- **barrier** — waiting at a thread-synchronisation point.

The *dominant* stall reason, together with occupancy and SoL, is what the
classifier (Part 6) uses to name the bottleneck. (E.g. "`wait` dominant at 11%
occupancy with DRAM idle" ⇒ G7 dependency-bound ⇒ *not* a memory problem, keep it
on the GPU and raise occupancy.)

### 5.8 Locality and the reuse-distance CDF (the DAMOV heart)

**Locality** = how much a program *reuses* data it already fetched. Measured from
the NVBit address trace (§4.3), machine-independently:

- **Reuse distance** — for each memory access, how many *distinct* other
  locations were touched since this one was last touched. Small reuse distance =
  the data was used again soon = a cache would catch it.
- The **reuse-distance CDF** (Cumulative Distribution Function — a curve showing
  "what fraction of accesses have reuse distance ≤ X") answers: *what hit rate
  would a cache of size X give?* If the curve is **flat** from a small cache to a
  huge one, then **no cache size helps** — the misses are **compulsory** (first-
  time touches of streaming data). That is the signature of cache-immune
  streaming (finding F5), and it argues for near-sensor consumption, not a bigger
  cache.

### 5.9 The memory-space split (data vs scratch)

Every memory access is in a **space**: **global** (real DRAM data), **shared**
(on-chip scratchpad, §1.3), or **local** (register spill, §1.7). A crucial
methodological point: locality and "DRAM traffic" claims must be about **global**
space only. Early in the project a kernel looked "coalesced and cache-friendly"
— until we filtered by space and found that signal was its **register-spill
stream** (local space), while its actual **global** data accesses were a
scattered gather. Separating these (via `--spaces global`) is what makes the
attribution honest (F6).

---

## Part 6 — The method: the six-step pipeline and the decision tree

The whole characterization is one pipeline. Each step consumes the previous one.

    Capture → Screen → Classify → Attribute → Validate → Verdict

### 6.1 Capture

Run the workload under the three profilers (§4) at locked clocks: nsys for the
timeline, ncu for the counters (bounded window, curated set), NVBit for the
address trace (bounded window). Also record **whole-run energy** (joules, by
sampling GPU power and integrating over time) and **host-side I/O** (bytes read
from storage, memory-mapped page-ins, peak host RAM). Every run also records its
**accuracy** vs ground truth (§2.2) so we know it was computing correctly.

### 6.2 Screen

From the nsys timeline, compute each kernel's **share of GPU time**. Attention
goes to the kernels that dominate; a kernel using 0.1% of the time is not worth
redesigning hardware for. This is DAMOV's "does it matter" filter.

### 6.3 Classify — the decision tree in detail

For each kernel, its metrics (§5) feed an **ordered decision tree** (first
matching rule wins). The thresholds are stated once and are *the same values the
code uses* (they are imported live into the documentation so they cannot drift):

| threshold | value | meaning |
|---|---|---|
| `sol_hi` | 40 | a SoL % counts as "high" (bound) |
| `sol_ratio` | 1.5 | how much compute must beat memory to be compute-bound |
| `dram_sat` | 50 | DRAM SoL % = saturated |
| `lfmr_hi` / `lfmr_lo` | 0.4 / 0.35 | L2 "not helping" / "earning its keep" |
| `sect_scatter` | 8 | sectors/request marking a scattered gather |
| `occ_low` / `occ_low_dep` | 25 / 30 | occupancy below which latency (G4) / dependency (G7) can't be hidden |

The rules, in order (this exact order is the classifier):

1. **G5 compute** — compute SoL ≥ 40% and clearly beats memory SoL.
2. **G6 on-chip** — dominated by a shared-memory/MIO stall while DRAM is *not*
   saturated.
3. **G1 bandwidth** — DRAM SoL ≥ 50% (the highway is full).
4. **G2 coalescing** — memory-limited *and* sectors/request ≥ 8 (scattered).
5. **G3 L2-reuse** — memory-limited *and* LFMR < 0.35 (the L2 is catching reuse).
6. **G4 latency** — a memory stall dominates at occupancy < 25% with DRAM not
   saturated (waiting on memory latency the GPU can't hide).
7. **G7 dependency** — a *dependency* stall (wait/barrier/…) dominates at low
   occupancy while *neither* memory nor compute is high (the wall is instruction
   dependencies, not data).
8. **G0** — nothing dominant (tiny/launch-overhead kernels).

### 6.4 Sensitivity and independent validation (why to trust the labels)

- **Sensitivity analysis** — every kernel is re-classified with all thresholds
  scaled ±25%. If its class *changes* under that wiggle, it is flagged
  "borderline" and cannot carry "high" confidence. Headline kernels are stable.
- **Clustering validation** — separately, an unsupervised **k-means** algorithm
  (which groups points by similarity with *no* labels) is run on the pooled
  feature cloud of all kernels across all sequences. It independently prefers
  **7–8 clusters**, matching the 7–8 hand-built G-classes. So the taxonomy is
  *discovered from the data*, not merely asserted — the decision tree is only the
  labelling; clustering is the check.

### 6.5 Attribute

Using the NVBit trace ⋈ TaggedAllocator journal join (§4.4), split each kernel's
DRAM traffic by **memory space** (global/shared/spill) and name the **data
structure** behind the global part. This is what turns "this kernel is memory-
bound" into "this kernel streams the keyframe-descriptor database" — the sentence
an architect needs.

### 6.6 Verdict — from evidence to a substrate

The class × the stage's **persistence** (is the data *streaming* through,
*hot-persistent* like the BA matrix reused within a frame, or *cold-persistent*
like the keyframe DB scanned occasionally) × the features give the substrate:

| class + condition | affinity | substrate |
|---|---|---|
| G1 + streaming | strong | **near-sensor SRAM** (consume before DRAM) |
| G1 + otherwise | strong | **DRAM-PiM** (bank-level bandwidth) |
| G2 (scatter) | conditional | **scatter-capable PiM**, or a data-layout fix first |
| G4 + cache-defeating | strong | **near-memory compute** |
| cold-persistent + big set | strong | **ISP / near-storage** (the keyframe DB) |
| G3 (L2 helps) | weak | keep on GPU (a bigger cache wins) |
| G5 / G6 / G7 | none | **keep on GPU** |
| tiny (occ<8%, <1 ms) | — | **CPU/host** |

And, when the verdict is "keep on GPU", the *same evidence* names the **current-
architecture fault** to fix first: register-spill-dominated traffic ⇒ a register-
pressure/compiler fault (it's scratch, not data); high sectors/request ⇒ a data-
layout fault; low occupancy + dependency stalls ⇒ a launch-configuration fault; a
flat reuse CDF ⇒ structurally cache-immune (needs near-sensor consumption, not a
bigger cache).

### 6.7 The placement model (turning verdicts into a number)

To estimate the payoff, a simple analytical model. If a kernel takes time `t` on
the GPU and a fraction `m` of that is memory-bound work, then on a PiM substrate
with internal bandwidth `k`× the GPU's and compute ratio `c`:

    t_pim = t·(1−m)/c + t·m/k

Offload the kernel only if its affinity is allowed in the scenario *and*
`t_pim < t`. Two scenarios: **conservative** (k=4, c=0.5, strong-affinity only)
and **moderate** (k=8, c=0.75, strong+conditional). An energy version uses the
same split. The rule (a standing project law): *simulated numbers are always
reported as **deltas** versus the measured GPU baseline, never as absolutes.*

---

## Part 7 — Trust: how we proved the numbers are real

A reviewer's first attack is "your measurements are noise / your instrumentation
changed the answer / you got lucky on one run." Each is pre-empted:

- **Accuracy is validated (F13).** We ran a **141-configuration matrix** across
  KITTI, EuRoC, TUM, ICL-NUIM datasets and every sensor mode (stereo, RGB-D,
  monocular, inertial) and pipeline mode. The profiled cuVSLAM **reproduces
  NVIDIA's published accuracy** (e.g. EuRoC stereo APE 0.114 m vs the paper's
  0.13; KITTI 500 m-drift 0.82% vs the leaderboard's 0.85%). So we measured a
  *correct* program.
- **Profiling is neutral.** We re-ran everything with and without each profiler:
  on deterministic modes nsys and ncu give **bit-identical** trajectories, and
  NVBit is within ~2 mm. The instrumentation does not change the computation, so
  the memory numbers describe cuVSLAM, not the profiler. (Across the 192-config
  campaign, 121 were bit-identical; every flagged difference was traced to a
  benign cause.)
- **Nondeterminism is understood (F13).** GPUs are slightly nondeterministic
  run-to-run (floating-point reductions can reorder). On long, large-loop
  sequences (kitti00) the trajectory can be *bimodal* — but a *plain* re-run
  scatters the same amount, proving it is the sequence, not the profiler.
- **Locked-clock rigor (F1).** CoV 0.14%; ceilings measured; thresholds
  sensitivity-tested; cache cold/warm bracketed. No headline rests on a kernel
  with high variance.
- **The classifier itself is validated, DAMOV-style.** Four checks mirroring
  the original paper's own robustness section (`docs/GPU_DAMOV_PARITY.md`):
  (1) *held-out ground truth* — eight small CUDA kernels each **designed by
  construction** to be one class (a streaming triad is bandwidth-bound, a
  register-resident polynomial is compute-bound, …) were classified blind with
  the frozen thresholds: **8/8 recovered** (DAMOV's analogous check: 97/100);
  (2) *real-hardware intervention* — locking the GPU's core clock and memory
  clock independently and watching which one each kernel's runtime tracks:
  **7/7 classes match their predicted response signature** (and the experiment
  itself taught us two GPU facts: a scattered gather is bounded by memory-
  request *concurrency*, not bus bandwidth, and load latency is mostly
  core-clock-domain L2/interconnect traversal); (3) *cross-hardware* — the same
  workload classified on two different GPUs keeps its class for **80%** of
  signal kernels; (4) *two independent clustering algorithms* (k-means and
  hierarchical) both reproduce the class structure. The taxonomy is not just
  asserted — it survives the same tests its CPU ancestor set for itself.

---

## Part 8 — The findings, one by one (what we actually learned)

These are the results a thesis reports. Each is a measured fact with a
consequence.

- **F1 — Publishable rigor.** Locked-clock RTX 2000 Ada: CoV 0.14%; measured
  ceilings 205 GB/s / 5445 GFLOP/s; ±25% classification sensitivity; cold/warm
  cache bracket. *Consequence:* every later number is a stable, defensible
  quantity.
- **F2 — It generalizes.** 27 sequences × 4 datasets, odometry and SLAM, **0
  failures**. A kernel's bottleneck class is the *same* across sequences 91% of
  the (time-weighted) time (24/49 kernels unanimous). *Consequence:* the
  bottleneck is a property of the *kernel*, not of one lucky dataset.
- **F3 — The taxonomy is discovered, not asserted.** k-means prefers k=7–8,
  matching the hand-built classes. Roughly **60–72% of GPU time carries PiM
  affinity** (≈21% strong + ≈40% conditional across the campaign; ~72% time-
  weighted across the deep studies). *Consequence:* the offload opportunity is
  large and independently validated.
- **F4 — Data movement is already the bottleneck at the system edge.** Explicit
  CPU↔GPU copies cost **41%** of kernel time; the camera-to-GPU upload is **1.68
  MB per frame**. *Consequence:* a direct, measured near-sensor opportunity.
- **F5 — The front-end is cache-immune streaming.** Its reuse-distance CDF is
  **flat from 64 KiB to 48 MiB** — no cache size helps; the misses are
  compulsory. *Consequence:* a bigger cache is the wrong fix; near-sensor
  consumption is the right one.
- **F6 — The loop-closure scan is a scattered gather (two methods agree).** After
  separating spill from data (§5.9), the loop-closure kernel `st_track`'s *global*
  accesses are 23–30 sectors/warp, only 2–6% coalesced — a scattered gather,
  matching the ncu counters. *Consequence:* a scatter-capable PiM (or a data-
  layout fix) target; the classifier label G2 stands, confirmed by two
  independent methods.
- **F7 — The scan's working set grows with the session.** `st_track`'s footprint
  grows 0.46 → 1.09 MB from room-scale to street-scale, and it migrates between
  scans. The union over a whole deployment — the entire keyframe database — is
  what no cache holds. *Consequence:* the ISP / near-storage case.
- **F8 — The GPU memory budget is static; the database grows on the host.** The
  TaggedAllocator journal shows the GPU allocation is **fixed** (240 allocations,
  108.65 MB; keyframe state a fixed 6.7 MB) even over a 2500-frame loop-closing
  run — cuVSLAM pre-sizes everything. *Consequence:* the session-scale database
  grows *host-side*, which is exactly what an ISP/near-storage substrate targets
  — shown from the allocator, not inferred.
- **F9 — Every byte has a name.** The three-layer attribution resolves **274/274
  allocations, 0 unknown**; **48/49 kernels** have a unanimous data-structure
  tag. `st_track` = 92.6% register spill + keyframe descriptors; the BA solver =
  97% the BA linear system. *Consequence:* verdicts are about named data
  structures, not anonymous traffic.
- **F13 — Accuracy validated (see Part 7).** *Consequence:* the whole
  characterization measures a correct program.
- **Energy (new).** Whole-run GPU energy is now measured: **34.67 J** on one
  sequence (12.54 W mean, 25.08 W peak). *Consequence:* the joule the PiM story
  is about is a measured baseline, not an assumption.
- **Host I/O + memory (new).** During a run: **66 MB** read from storage (the
  sensor-data ingestion feeding the H2D upload of F4) and **708 MB** peak host
  RAM (where the keyframe DB lives while the GPU allocation stays static per F8).
  *Consequence:* the characterization now has a *host* dimension, arming the ISP
  claim with a direct measurement.

The through-line: **cuVSLAM's memory behaviour splits cleanly** — a cache-immune
streaming *front-end* (near-sensor), a compute-heavy *bundle adjustment* (keep on
GPU), and a scattered, session-growing *loop-closure/keyframe-database* back-end
(PiM-scatter + ISP/near-storage) — each a *different* substrate, each with
measured evidence.

---

## Part 9 — What was built (the software)

The project is also a reusable toolchain, not just a study:

- **A single-TOML runner** — one text file fully describes a run (dataset, camera
  rig, every algorithm knob, the accuracy check). No per-dataset scripts.
- **A config regime** — humans own a small set of **base** configs; a mutator
  script derives every variant (odometry/SLAM/async/CPU, feature toggles) from
  them, so the whole matrix is defined once and reproducible byte-for-byte.
- **A profiling harness** — one entry point runs any config under nsys / ncu /
  NVBit and writes versioned results with provenance. It is **workload-agnostic**
  via *adapters*: cuVSLAM is one adapter; a second adapter runs *any* GPU program
  (a PyTorch model, a CUDA benchmark) through the identical pipeline. This is how
  the method generalizes beyond cuVSLAM.
- **The analysis library** — screen, classify, roofline, locality, attribution,
  substrate — all runnable from committed CSVs with no GPU.
- **A controller/target model** — the machine you sit at (the "controller") drives
  profiling on remote "targets" (a workstation, a Jetson Orin robot board) over
  ssh; an environment **doctor** checks each target and prints a one-line fix for
  every driver/CUDA/tool incompatibility.
- **The dashboard** — a web app with the whole story: **Findings** (each
  conclusion with its computed number and an interactive chart), **Explore** (one
  run's full reasoning chain: where time goes → which kernels dominate → the
  metrics vs their thresholds → the decision tree *fired* for that kernel → the
  verdict), **Methodology** (this document's content, with formulas stamped live
  from the code so the UI can never disagree with the analysis), **Profile**, and
  **Setup**. Every number traces back to a committed measurement artifact.

---

## Part 10 — Limits, roadmap, and what's next (be honest about these)

A defense rewards knowing your own gaps. The honest status:

**Done (the characterization paper's evidence is complete):** everything in
Part 8, plus energy and host-I/O.

**Deliberately *not* done, with reasons:**
- **No substrate-side simulation yet** — we prove *candidacy* (which kernel
  wants PiM/ISP) but have not simulated the actual speedup/energy *delta* on NDP
  hardware. That is the *architecture* paper (Phase 3), not the *characterization*
  paper. **Groundwork is laid:** a generator emits Accel-Sim NDP configs and
  per-kernel offload manifests from the verdicts (conservative offloads 16/49
  kernels, moderate 26/49); the remaining work is running the (gated) simulator.
- **The last ~40% of a few kernels' traffic is unmapped** (module globals,
  texture reads). **Groundwork is laid:** a script names the exact targets
  (`sba::build_full_system`, `lk_track` carry 83% of the unmapped traffic); the
  fix (a Layer-3 kernel-argument correlation) needs one more instrumentation tool.
- **One GPU (desktop Ada).** Some findings (the exact spill/shared/global split)
  are compiler-codegen-specific. The app already targets a **Jetson Orin** robot
  board; re-running there validates codegen-independence — *ready, needs the
  device*.
- **One workload family (cuVSLAM).** Argued as representative (production-grade,
  deployed in Isaac) rather than benchmarking a second SLAM system, which would
  add scope with weak marginal evidence.

**Dropped, with reasons:** using a simulator as the *primary* instrument (we use
real hardware + native tools), the original CPU DAMOV pipeline (it is the origin
we adapted *from*), multi-tenant GPU sharing (cuVSLAM is single-tenant), and a
config-level occupancy sweep (the single-point occupancy + the G4/G7 class
already answer the question).

The phased plan: **characterize (done) → quantify + lay groundwork (now) →
design + evaluate the NDP substrate (next, the architecture paper).**

---

## Part 11 — Defending it: questions you will be asked, with answers

**Q: "Isn't this just profiling? What's the contribution?"**
A: Three contributions. (1) *Method:* GPU-DAMOV — adapting a CPU data-movement
methodology to GPUs (two-level cache → LFMR redefined; occupancy-hidden latency;
coalescing), extended from "is it memory-bound" to "which substrate + which
on-GPU fault." (2) *Empirical:* applying it to a **production** V-SLAM stack (not
a benchmark) and finding a clean three-way substrate split with per-data-
structure evidence. (3) *Artifact:* a reproducible, workload-agnostic toolchain.

**Q: "How do you know the classification is right and not arbitrary?"**
A: Two independent checks. The thresholds are sensitivity-tested (±25%, borderline
kernels flagged); and unsupervised k-means, with no labels, independently finds
the same 7–8 clusters. The decision tree is the labelling; clustering is the
validation.

**Q: "Did your instrumentation change what you measured?"**
A: No — proven. On deterministic modes nsys and ncu give bit-identical
trajectories, NVBit within ~2 mm; the residual differences are the sequence's own
run-to-run nondeterminism, which a plain re-run reproduces.

**Q: "Why should I believe cuVSLAM was even working?"**
A: A 141-configuration accuracy matrix reproduces NVIDIA's published numbers
across four datasets and every sensor mode. We measured a correct program.

**Q: "You claim the keyframe database belongs near storage — did you see that, or
guess it?"**
A: Measured. The TaggedAllocator journal shows the GPU allocation is *static*
(fixed 6.7 MB keyframe buffer) while the session-scale database grows host-side;
attribution names the loop-closure kernel's traffic; host-memory sampling shows
708 MB peak host RAM. Three independent measurements, not inference.

**Q: "The scattered-gather finding contradicted your earlier trace — which is
right?"**
A: This is a strength, not a weakness. The counters said scattered; an unfiltered
trace briefly said coalesced; the attribution join revealed that "coalesced"
signal was the *register-spill* stream; filtering to global space showed the real
data accesses *are* a scattered gather, matching the counters. Two methods now
agree after a self-correction on record.

**Q: "Where's the speedup? You never show PiM being faster."**
A: Correct, and deliberate. This is the *characterization* — it establishes
candidacy and the offload opportunity (60–72% of GPU time) with an analytical
placement model. Demonstrating the actual speedup/energy delta needs an NDP
simulator (Accel-Sim + AccelWattch); that is the follow-on architecture paper,
and its config-generation groundwork is already built.

**Q: "What's Physical AI got to do with it?"**
A: Robots are battery-powered and latency-sensitive; the energy cost of data
movement (measured: 34.67 J/run baseline) is exactly what near-data substrates
attack. cuVSLAM is the vision core of NVIDIA's robotics stack, so it is a
representative Physical-AI workload.

---

## Glossary — every term in one place

- **AI (Arithmetic Intensity)** — FLOPs per DRAM byte; low = memory-bound.
- **Attribution** — naming which data structure a kernel's memory traffic touches.
- **Bandwidth** — data moved per second (GB/s); a highway's width.
- **Bank** — a subdivision of a DRAM chip; PiM puts compute at each bank.
- **BA (Bundle Adjustment)** — the big optimization refining poses and 3-D points.
- **Cache (L1/L2)** — automatic fast copies of recently-used data; L1 per-SM, L2 shared.
- **Coalescing** — a warp's 32 threads reading neighbouring memory (efficient) vs scattered (wasteful).
- **CoV (Coefficient of Variation)** — std-dev ÷ mean; our run-to-run noise (0.14% locked).
- **CUDA** — NVIDIA's GPU programming system.
- **cuVSLAM** — NVIDIA's production GPU Visual-SLAM program; the workload.
- **DAMOV** — the CPU data-movement-bottleneck methodology we adapt to GPUs.
- **DRAM / VRAM / global memory** — the GPU's big, slow main memory.
- **FLOP** — one floating-point operation (add/multiply).
- **GPU / SM / warp / thread / block** — the parallel chip and its work units (§1.1–1.2).
- **Ground truth** — the true camera path, for checking accuracy.
- **Heterogeneous computing** — using different hardware kinds for different jobs.
- **ISP (Image Signal Processor)** — the camera's processing chip; a near-sensor substrate.
- **Joule / Watt** — energy / power (energy per second).
- **Kernel** — one GPU function launch (thousands of parallel threads).
- **Latency** — wait time for one datum; hidden by occupancy.
- **LFMR** — 1 − L2-hit-rate; ≈1 means caches don't help (PiM-favourable).
- **Locality / reuse distance / reuse CDF** — how much data is reused; flat CDF = cache-immune.
- **Memory wall** — compute outran memory, so programs wait on data.
- **MPKI** — DRAM sector-fetches per thousand warp-instructions (memory intensity).
- **ncu (Nsight Compute)** — the counter microscope (kernel replay).
- **nsys (Nsight Systems)** — the timeline profiler.
- **NVBit** — binary instrumentation recording every memory address.
- **NVTX** — in-code annotations giving measured kernel→stage mapping.
- **Occupancy** — fraction of max concurrent warps active; hides latency.
- **PCIe** — the CPU↔GPU link; crossing it is slow.
- **Physical AI** — machines that sense and act in the real world (robots).
- **PiM (Processing-in-Memory)** — compute engines inside/next to DRAM (near-bank / scatter).
- **Register spill / local memory** — compiler scratch pushed to DRAM; not real data.
- **Roofline** — the AI-vs-performance graph that shows memory- vs compute-bound.
- **Sector** — a 32-byte memory fetch chunk; "sectors/request" measures coalescing.
- **SLAM** — Simultaneous Localization and Mapping (from camera images = Visual SLAM).
- **SoL (Speed-of-Light)** — a resource's utilisation as % of peak (memory/compute/DRAM).
- **Stall reason** — why a warp isn't issuing (long_scoreboard=memory, wait=dependency, …).
- **Substrate** — where a computation runs: GPU / CPU / PiM / ISP / near-sensor.
- **TaggedAllocator** — our instrumentation journaling every allocation + its data-structure tag.
- **Taxonomy (G0–G7)** — the eight GPU bottleneck classes.
- **von Neumann bottleneck** — compute separated from memory, so data must shuttle across a narrow link.

---

*This document is the single source you need. The measured artifacts behind every
number live in `reports/` and `profiling/reports/`; the exact formulas and
thresholds live in `profiling/analysis/` and are shown, stamped from that code,
in the dashboard's Methodology tab. If a number here ever disagrees with the
code, the code is right — regenerate the docs.*
