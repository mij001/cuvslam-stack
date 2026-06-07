# Adapting DAMOV to GPU Compute — A Research Feasibility Study

*How to port the DAMOV data-movement-bottleneck methodology from CPUs to GPUs, using NVBit, Nsight Compute (NCU), Nsight Systems (Nsys), Accel-Sim/GPGPU-Sim, and AccelWattch. Written as a follow-on to the DAMOV walkthrough; assumes you've read that guide.*

---

## 0. Verdict up front

**It is feasible, and the GPU ecosystem already contains a near-perfect analog of every single component DAMOV used.** The original DAMOV-SIM is `Pin → ZSim (cores+caches) + Ramulator (HMC DRAM)`, screened by Intel VTune and costed by a CACTI-style energy model. The GPU world hands you a one-to-one replacement stack: `NVBit → Accel-Sim/GPGPU-Sim 4.x (SMs + L1 + L2 + HBM) + AccelWattch (power)`, screened by Nsight Compute + Nsight Systems. NVBit is explicitly the GPU equivalent of Pin/DynamoRIO, and Accel-Sim's frontend is *already* built to consume NVBit traces — exactly the relationship Pin has with ZSim.

**But the *methodology* does not port unchanged.** Two of DAMOV's three steps transfer almost directly (the locality analysis is literally architecture-independent), while the third — the six-class bottleneck taxonomy — must be **re-derived**, because GPUs relate to memory in fundamentally different ways than CPUs:

1. GPUs hide memory latency with *massive thread parallelism* (thousands of warps), not out-of-order execution and deep caches. This largely **dissolves DAMOV's "DRAM latency-bound" class**.
2. GPUs have a **two-level** cache hierarchy (per-SM L1, shared L2, then DRAM) — no L3 — so DAMOV's star metric **LFMR must be redefined** and the L1/L2/L3-capacity classes restructure.
3. GPU memory performance is dominated by **coalescing and control-flow divergence**, first-order effects with *no CPU analog*, which must become new metrics.
4. GPUs already use **HBM (~TB/s)**, so the bandwidth headroom that near-data processing offers over a GPU is much smaller than over a CPU. The interesting GPU-NDP win cases shift toward *capacity-bound* and *energy* arguments rather than raw bandwidth.

The rest of this document is the detailed case: the tool-for-tool mapping, how each DAMOV step changes, a proposed revised taxonomy, a concrete "GPU-DAMOV" pipeline you could build, and the real pitfalls.

---

## 1. The five ingredients of DAMOV we must port

Recall what DAMOV is actually made of (from the walkthrough). To adapt it, we port each piece:

| # | DAMOV ingredient | Its job | What it needs from a GPU version |
|---|---|---|---|
| 1 | **VTune top-down** (Step 1) | Screen for memory-bound functions (`Memory Bound > 30%`) | A GPU profiler that flags memory-bound kernels |
| 2 | **Locality analysis** (`locality.cpp`, Step 2) | Architecture-independent temporal/spatial locality from the address stream | A way to capture a GPU kernel's memory address stream |
| 3 | **DAMOV-SIM** = ZSim + Ramulator (Step 3) | Controlled cycle-accurate sim: vary caches/cores, measure AI/MPKI/LFMR | A configurable, validated GPU simulator |
| 4 | **Energy model** (CACTI-derived pJ/access) | Per-config energy for the host-vs-NDP comparison | A GPU power model |
| 5 | **The metrics + 6-class tree** | The diagnosis itself (LFMR, MPKI, AI, locality) | A re-derived GPU metric set + taxonomy |

Ingredients 1–4 are *tooling* (largely solved on GPUs). Ingredient 5 is *intellectual* (the actual research contribution of a "GPU-DAMOV").

---

## 2. The tool-for-tool mapping (CPU → GPU)

This is the headline result of the research. Every DAMOV building block has a mature, validated GPU counterpart:

| DAMOV (CPU) | GPU replacement | Notes |
|---|---|---|
| **Intel Pin** (binary instrumentation) | **NVBit** | NVBit is a dynamic binary instrumentation library for NVIDIA GPUs, explicitly designed as the GPU analog of Pin/DynamoRIO. It instruments SASS (the native GPU machine ISA) without recompiling, injecting callbacks before/after instructions. |
| **ZSim hooks** (`xchg %rcx`) for ROI | NVBit instruction/range selection + **NVTX ranges**; Accel-Sim kernel selection | You mark which kernel(s)/launches to trace rather than a magic instruction. |
| **ZSim** (cores + cache hierarchy + coherence) | **GPGPU-Sim 4.x** inside **Accel-Sim** | Models SMs (with sub-cores, warp schedulers), per-SM L1/scratchpad, shared L2, interconnect. Cycle-level, validated against real GPUs. |
| **Pin→ZSim** trace flow | **NVBit→Accel-Sim** trace flow | Accel-Sim ships an *"Accel-Sim Tracer"* (an NVBit tool) that generates SASS traces; its *SASS frontend* feeds them to the performance model. This is the *same* instrument→trace→simulate pattern DAMOV uses. |
| **Ramulator** with **HMC** config | GPGPU-Sim's built-in DRAM model (GDDR/HBM); or plug in **DRAMsim3 / Ramulator2** for a custom PIM-aware memory | GPGPU-Sim already models GDDR5/HBM; HBM-PIM research typically swaps in an extended DRAMsim3 for bank-level compute. Ramulator already had HBM/GDDR5 device configs in the DAMOV repo itself. |
| **VTune top-down** (Step 1 screen) | **Nsight Compute (NCU)** + **Nsight Systems (Nsys)** | Nsys = system-level timeline (which kernels matter, data transfers, concurrency). NCU = per-kernel microarchitectural deep-dive (Speed-of-Light, stalls, memory workload, roofline). |
| **CACTI-style energy** (pJ/access in Table 1) | **AccelWattch** | A validated GPU power model integrated with Accel-Sim/GPGPU-Sim 4.2+; models SASS, divergence, DVFS; can be driven by the simulator *or* by real hardware counters. |
| **`get_stats_per_app.py`** (parse stats → MPKI/LFMR/IPC) | Accel-Sim stats + a custom parser; **Accel-Sim Correlator** | The Correlator even matches simulator stats against real-hardware NCU stats — built-in validation DAMOV did by hand. |
| **k-means / hierarchical clustering** | Same (scikit-learn) | The clustering step is tool-agnostic and ports verbatim. |

The practical punchline: a "GPU-DAMOV" is not a from-scratch build. You'd be **gluing existing, validated, open-source tools** (NVBit + Accel-Sim + AccelWattch + NCU/Nsys) in the same topology DAMOV glued ZSim + Ramulator + VTune — and the GPU community already wires NVBit→Accel-Sim routinely.

> **AMD note.** If you target AMD GPUs instead, the analogs are: profiling = **ROCprof / Omniperf**; simulation = **Accel-Sim's AMD-GCN trace support** or **MGPUSim**. The conceptual story is identical; tool names change.

---

## 3. Step 1 on GPUs: finding memory-bound kernels (Nsys + NCU)

DAMOV Step 1 = "use VTune top-down to keep functions with Memory Bound > 30% and ≥ 3% of cycles." On GPUs this splits cleanly across two tools.

**Nsys (Nsight Systems) — the app-level / "which kernel matters" pass.** Nsys gives a *system-wide timeline*: CPU threads, GPU kernels, CUDA API calls, host↔device memory copies (H2D/D2H), stream concurrency, and NVTX-annotated regions. Use it the way DAMOV uses the "≥ 3% of cycles" filter — to find the kernels that dominate runtime and to spot *system-level* data-movement bottlenecks (e.g., the program is actually bottlenecked by PCIe H2D copies, not kernel compute). This catches a class of data movement DAMOV's function-level CPU view doesn't emphasize: **host-device transfer**.

**NCU (Nsight Compute) — the per-kernel "is it memory-bound, and why" pass.** NCU is the real VTune analog. Its relevant sections:

- **GPU Speed of Light (SoL).** Reports achieved performance as a percentage of peak for two top-level domains: **Compute (SM) %** and **Memory %**. This is a *direct* analog of the roofline compute-vs-memory split and of VTune's top-down. A kernel with high Memory% and low SM% is your "memory-bound" candidate. (Example phrasing you'll see in practice: "Speed-of-Light score: 18% compute, 70% memory.")
- **Warp State Statistics (stall reasons).** This is the gold mine and arguably *better* than VTune's single "Memory Bound %," because NCU tells you *which kind* of memory problem:
  - **Stall "Long Scoreboard"** = warps waiting for data from L1TEX/cache/VRAM → **latency-bound**.
  - **"LG Throttle"** (local/global throttle) = the load/store queue is full of in-flight memory requests → **bandwidth-bound / uncoalesced** (too many requests, often because accesses aren't coalesced so each thread issues its own transaction).
  - **"MIO Throttle"** = memory-input/output pipe saturated (often shared-memory/special-function pressure).
  - NVIDIA's own heuristic: if the issue-active SoL is below ~80% *and* Long Scoreboard is the top stall, the kernel is memory-latency limited.
- **Memory Workload Analysis.** A memory chart with per-unit transfer sizes, **L1 and L2 hit rates**, and **requested vs. achieved** memory throughput. This directly yields the cache and bandwidth numbers you need.
- **Source Counters / warp-stall sampling.** Periodically samples the warp PC and scheduler state, correlating stalls down to individual SASS/PTX/CUDA-C source lines (compile with `-lineinfo`). This is the GPU analog of finding the "hot" code, and it directly enables a *fine-grained-offloading* study like DAMOV's Case Study 4.
- **Roofline.** NCU computes arithmetic intensity and plots the kernel against the device's compute and memory roofs — the exact picture DAMOV §3.2 describes, available for free per-kernel.

**A subtle and important difference from CPU-DAMOV.** On a CPU, separating the *bandwidth-bound* (Class 1a) from the *latency-bound* (Class 1b) case required the *simulation* (LFMR + MPKI + core-count sweep). On a GPU, **NCU already gives you a coarse bandwidth-vs-latency split at profiling time** (Long Scoreboard vs LG Throttle, plus achieved-vs-peak bandwidth %). So a chunk of DAMOV's intellectual work is partially handed to you by the profiler. The simulation's job shifts from "diagnose the bottleneck" toward "predict what an NDP/PIM substrate or a different cache config would do about it."

**Proposed GPU Step-1 screening rule** (the analog of "Memory Bound > 30%"): keep a kernel if it consumes a meaningful fraction of Nsys timeline *and* NCU shows Memory% ≫ SM% (e.g., Memory% > 60% and SM% < 50%) **or** a memory stall reason (Long Scoreboard / LG Throttle / MIO Throttle) is the dominant stall. The exact thresholds should be calibrated empirically, exactly as DAMOV calibrated 30% by checking when fixes stopped mattering.

---

## 4. Step 2 on GPUs: locality (the part that ports almost unchanged)

DAMOV's Step 2 is its most portable piece, because `locality.cpp` is **architecture-independent by construction** — it consumes a stream of (address, size) pairs and computes temporal/spatial locality with no notion of cores or caches. You can feed it a GPU memory trace verbatim.

**Capturing the GPU address stream: NVBit `mem_trace`.** NVBit ships a `mem_trace` tool that records the memory reference address of every global load/store by injecting an instrumentation function that reads the registers/immediate used to compute the address, and ships them GPU→CPU over a channel. That stream is exactly the input `locality.cpp` wants. (Caveat: `mem_trace` overhead is ~100–1000×, vs ~2–5× for basic-block counting — see §9.)

**But you must make four GPU-specific decisions**, because a GPU "address stream" is not a single sequential thing like a CPU thread's:

1. **Granularity / interleaving.** A CPU trace is one thread's sequential accesses. A GPU kernel has thousands of threads in warps, all interleaved. You must decide whether to compute locality **per-thread**, **per-warp**, or **per-SM**. Per-thread reuse-distance measures algorithmic reuse; per-warp/per-SM captures what the shared caches actually see. A defensible choice: compute per-warp (the hardware's unit of memory issue) and also a per-thread variant, and report both. This is a genuine extension of DAMOV's metric, not a copy.

2. **Spatial locality ≈ coalescing-friendliness.** DAMOV's spatial-locality metric (minimum stride over a window) is, on a GPU, essentially measuring whether a warp's 32 threads touch nearby addresses — i.e., whether the access will **coalesce** into one wide transaction or scatter into 32 small ones. So DAMOV's spatial metric is *more* important on GPUs, but it should be computed *within a warp at an instant* (intra-warp spatial locality) rather than only along one thread's timeline.

3. **Stack-removal analog.** `locality.cpp` discards CPU stack addresses (anything touched ≥ 2²¹ times) so they don't fake locality. The GPU analog: you typically trace only **global** memory (NVBit's `mem_trace` already targets global loads/stores), excluding registers and **shared memory/scratchpad**. But shared-memory usage is itself a first-order GPU signal (it's a *programmer-managed* cache), so you may want to *separately* measure shared-memory traffic rather than just drop it.

4. **Word vs sector granularity.** DAMOV shifts addresses by `log2(size)` to stay cache-line-independent. GPUs use **sectored caches** (e.g., 32-byte sectors within 128-byte lines), so you'd likely analyze at 32-byte sector granularity to match how GPU caches actually move data.

**New architecture-independent metrics to add (GPU-specific).** Beyond temporal/spatial locality, a GPU study should add:
- **Coalescing efficiency** = (ideal transactions) / (actual transactions per memory instruction). NVBit can compute this directly from the per-warp addresses; NCU also reports it (sectors-per-request).
- **Branch/memory divergence** = fraction of warp lanes active per instruction. NVBit exposes the active mask per instrumented instruction, so this is directly measurable.

These two have *no CPU counterpart* and will be central to the GPU taxonomy (§6).

---

## 5. Step 3 on GPUs: the controlled experiment + redefined metrics

DAMOV Step 3 runs each function on three configs (Host / Host+prefetch / NDP) across a core-count sweep {1,4,16,64,256}, collecting AI/MPKI/LFMR. Here's how each piece maps.

### 5.1 The simulator and configs

Use **Accel-Sim / GPGPU-Sim 4.x** as the cycle-level model. Define (at least) two configurations, mirroring DAMOV's Host-vs-NDP contrast:

- **"Baseline GPU"** — a standard modern GPU: per-SM L1 (adaptive, shared with scratchpad), a large shared L2, GDDR/HBM. (This is the default Accel-Sim config for, say, an A100/H100.) This is the analog of DAMOV's "Host CPU."
- **"NDP/PIM GPU"** — the harder, *modeling* part. There is no single right answer (this is itself a research decision, just as DAMOV's NDP config was). Reasonable options:
  - *Logic-layer near-memory cores*: GPU-like SMs (or simpler units) placed near HBM with **reduced or no L2** and higher internal bandwidth + lower latency — the closest analog to DAMOV's "delete L2/L3, turn on pimMode." You'd reduce the L2, shorten DRAM latency, and raise effective bandwidth in the memory model.
  - *Bank-level / in-DRAM PIM (HBM-PIM style)*: offload only simple ops (e.g., GEMV, element-wise) to compute units inside DRAM banks — modeled with an extended DRAMsim3 (as current HBM-PIM works do).
  - *Offload-to-near-memory-GPU-cores (TOM/Pattnaik style)*: model the existing GPU-NDP proposals.

Optionally add a **prefetcher** config (GPUs have hardware prefetchers and software prefetch), mirroring DAMOV's middle config, though prefetching plays a different role on GPUs (TLP often substitutes for it).

### 5.2 Redefining LFMR for a two-level hierarchy

DAMOV's LFMR = (L3 misses) / (L1 misses) — "what fraction of first-level misses reach DRAM," i.e., are the *deep* caches earning their keep. GPUs have no L3, so:

```
LFMR_gpu = (L2 misses) / (L1 misses)
```

Same intuition, one level down: **LFMR_gpu ≈ 1** means the shared L2 is useless (nearly every L1 miss reaches HBM) → an NDP-friendly fingerprint; **LFMR_gpu ≈ 0** means the L2 catches the spillover → the kernel benefits from the on-chip hierarchy and NDP (which removes L2) would hurt. This is a clean, defensible port of the single most important DAMOV metric.

### 5.3 Redefining MPKI and AI

- **MPKI → L2 MPKI**, but you must fix the **instruction-counting unit**. On a GPU, do you count *warp instructions*, *thread instructions*, or *SASS instructions*? They differ by up to 32×. The cleanest analog to DAMOV is **L2 misses per kilo *warp*-instruction** (the warp is the issue unit), but you should state the choice explicitly and be consistent — this is a classic GPU-metrics pitfall.
- **AI** — arithmetic intensity (FLOPs or ops per byte) ports directly and is *already standard* on GPUs via the NCU roofline. Use ops per byte of HBM traffic, or ops per L2 sector to mirror DAMOV's "ops per cache line."

### 5.4 The sweep: occupancy/SMs/CTAs instead of core count

DAMOV sweeps CPU core count 1→256 to (a) increase memory pressure, (b) reveal memory-level parallelism (MLP), and (c) grow the aggregate cache so it can tell "uncacheable" from "needs a bigger cache." The GPU equivalents of these knobs are different and richer:

- **Occupancy / number of concurrent warps per SM** — the GPU's *primary* latency-hiding knob, and the true analog of "MLP." Sweeping occupancy shows how much the kernel relies on parallelism to hide memory latency. (You can throttle warps in the simulator, à la cache-conscious warp scheduling.)
- **Number of SMs / CTAs (thread blocks)** — analog of "more cores → more pressure on shared L2 and HBM." Sweeping this exposes **L2 contention** and **HBM bandwidth saturation**, the GPU analogs of DAMOV's L3-contention and bandwidth-saturation behaviors.
- **L1/scratchpad partitioning** — modern GPUs split the unified on-chip memory between L1 cache and programmer-managed shared memory per kernel; sweeping this is a GPU-only axis with no CPU analog.

The "aggregate cache grows with the sweep" trick still applies for L1 (more SMs → more total L1), so you can still distinguish "L1-capacity-bound at low occupancy" from "fundamentally uncacheable" by watching how LFMR_gpu trends as you add SMs — exactly DAMOV's logic for separating its Class 1c from 1a/1b.

### 5.5 Energy via AccelWattch

DAMOV's host-vs-NDP energy comparison (its big argument for NDP) ports to **AccelWattch**, which gives cycle-level GPU power broken down by component (SM, caches, interconnect, DRAM), accounts for divergence, and is validated against real hardware. For an NDP-GPU config you'd extend AccelWattch's component model to charge less for the eliminated L2/off-chip traffic and more for in-memory compute — the GPU version of DAMOV's "NDP saves L2/L3 + off-chip-link energy."

### 5.6 Free validation: the Accel-Sim Correlator

DAMOV validated its simulator informally. Accel-Sim ships a **Correlator** that automatically matches simulator statistics against **real-hardware statistics from profiling tools (NCU)**. So you can validate your "Baseline GPU" config against the same NCU runs you used for Step 1 — closing the loop in a way the original DAMOV had to do by hand.

---

## 6. The hard part: why the six-class taxonomy must change

This is where a "GPU-DAMOV" becomes genuine research rather than a porting exercise. DAMOV's six classes assume a CPU's memory relationship. On a GPU, the axes shift:

**(a) The "DRAM latency-bound" class (DAMOV 1b) largely collapses.** A CPU stalls on a single long DRAM access because it has limited ways to hide it; that's why 1b exists. A GPU is *architected* to tolerate latency by switching among thousands of warps. If a kernel has enough occupancy, pure DRAM latency is hidden and the kernel becomes either bandwidth-bound or fine. So on GPUs, "latency-bound" mostly means **"latency-bound *because occupancy is too low to hide it*"** — a new, GPU-specific failure mode (low-occupancy latency exposure) rather than DAMOV's fundamental-latency class. (Note also: since Turing, GPUs handle some L1 misses out-of-order, further eroding pure latency stalls.)

**(b) No L3 ⇒ DAMOV's 1c (L1/L2 capacity) and 2a (L3 contention) restructure.** The GPU's shared **L2 is the contention battleground** (all SMs hammer one L2), so an **"L2 capacity/contention-bound"** class is real and central. The per-SM **L1 is tiny and adaptive** (shared with scratchpad), so an **"L1/coalescing-bound"** class is also real. The mapping isn't one-to-one; you'd likely merge and rename.

**(c) Coalescing and divergence become first-class bottleneck axes.** A kernel can be "memory-bound" purely because its accesses don't coalesce (each warp issues 32 tiny transactions, saturating the request queue → LG Throttle), even with perfect locality. There is *no CPU analog*; DAMOV has no class for it. This deserves its own class.

**(d) The NDP value proposition is different.** A CPU has ~100 GB/s; near-data processing offered DAMOV ~3.7× more bandwidth — a big prize. A GPU *already* has HBM at ~1–3 TB/s. So in-/near-memory processing offers a *smaller bandwidth multiplier* over a GPU. The compelling GPU-NDP cases are therefore:
   - **Capacity-bound** workloads where the dataset doesn't fit (LLM inference KV-cache, giant embeddings) — PIM adds *effective bandwidth at capacity* (this is exactly why Samsung HBM-PIM and recent LLM-PIM works exist).
   - **Energy** — skipping data movement still saves joules even when raw bandwidth isn't the limiter.
   - **Sparse / indirect / graph** access where the GPU's coalescing breaks down (recent processing-near-HBM SpGEMM/SpMV works target exactly this).

**A sketch of a revised GPU bottleneck taxonomy** (to be validated by clustering, not asserted):

| Proposed GPU class | Fingerprint (GPU metrics) | NDP/PIM verdict |
|---|---|---|
| **G1 — HBM bandwidth-bound** | high L2 MPKI, high LFMR_gpu, achieved BW ≈ peak, LG Throttle | ✅ candidate (esp. capacity-amplification + energy) |
| **G2 — Coalescing/divergence-bound** | low coalescing efficiency, high divergence, LG Throttle, high transaction count | ⚠️ helps only if PIM serves the scatter cheaply; often a *software* fix first |
| **G3 — L2 capacity/contention-bound** | LFMR_gpu rises with #SMs, high temporal locality | ⚠️ NDP relieves L2 pressure at scale (analog of DAMOV 2a) |
| **G4 — L1/occupancy-latency-bound** | Long Scoreboard dominant at low occupancy, LFMR_gpu falls with occupancy | ⚠️ often fixed by raising occupancy, not NDP |
| **G5 — Compute/Tensor-core-bound** | high AI, high SM%/Tensor%, low Memory% | ❌ keep on GPU (analog of DAMOV 2c) |
| **G6 — Scratchpad/shared-memory-bound** | high shared-mem traffic, MIO Throttle | ❌ GPU-specific; on-chip, not a data-movement-to-DRAM problem |

This is a *hypothesis* for what k-means/hierarchical clustering on the GPU metric set would reveal — the actual classes should fall out of the data, exactly as DAMOV's six did.

---

## 7. The NDP/PIM substrate question for GPUs

A reasonable objection: "the GPU already sits next to HBM — what does 'near-data processing' even mean here?" Three concrete substrates from the literature, each a valid target for a GPU-DAMOV "NDP config":

1. **In-DRAM / bank-level PIM (HBM-PIM).** Compute units inside DRAM banks do simple ops (GEMV, element-wise, reductions) at full internal bandwidth. This is *commercial* (Samsung HBM-PIM/Aquabolt) and the hot substrate for LLM inference; modeled with extended DRAMsim3 supporting bank-level execution and PIM-aware scheduling.
2. **Logic-layer near-memory accelerators / cores** in a 3D stack — the closest analog to DAMOV's HMC-logic-layer NDP.
3. **Offload-to-GPU-cores-near-memory** — the classic GPU-NDP research line: **TOM** (Transparent Offloading and Mapping, ISCA 2016), **TOP-PIM** (HPDC 2014), and **scheduling techniques for GPU PIM** (Pattnaik, PACT 2016), plus graph-specific work like **GraphPIM**. These are precisely the GPU-NDP works the original DAMOV paper *cites* but does not model.

There is also direct precedent for the *platform-independent NDP analysis* idea outside DAMOV: e.g., Corda et al.'s "platform-independent software analysis for near-memory computing" and Awan et al.'s NDP-for-Spark study reuse the same Ramulator + locality lineage. A GPU-DAMOV sits naturally in this line.

**Memory-model choice for the sim:** GPGPU-Sim's native DRAM model is adequate for a first cut, but for a serious PIM study you'd integrate **DRAMsim3** (used by current HBM-PIM works) or **Ramulator2** with bank-level-compute and PIM-scheduling extensions, replacing GPGPU-Sim's memory partition back-end. This mirrors DAMOV's choice to pair ZSim with Ramulator rather than ZSim's built-in memory.

---

## 8. A concrete proposed pipeline: "GPU-DAMOV"

Putting it together, here is an end-to-end recipe a research group could implement, step-by-step, with the exact tools:

**Workloads.** CUDA ports of the same suites DAMOV used (Rodinia, PolyBench-GPU, Parboil, Chai-GPU) plus GPU-relevant additions (DeepBench / cuDNN kernels, Tango, GNN kernels, sparse kernels, an LLM-inference kernel set for the capacity-bound story).

1. **Compile** the CUDA workloads (`-lineinfo` for source correlation).
2. **Nsys** → app-level timeline. Identify the kernels that dominate runtime and any host↔device transfer bottlenecks. (Analog of DAMOV's "≥ 3% of cycles" filter, plus a transfer-bottleneck check DAMOV lacked.)
3. **NCU** on the surviving kernels → Speed-of-Light (Memory% vs SM%), Warp State (Long Scoreboard / LG Throttle / MIO Throttle), Memory Workload Analysis (L1/L2 hit rates, achieved vs peak BW), and Roofline. Apply the **GPU Step-1 screening rule** (§3) to keep memory-bound kernels. Record a *first-cut* bandwidth-vs-latency-vs-coalescing label straight from NCU.
4. **NVBit `mem_trace`** on the kept kernels → per-warp global address streams. Port **`locality.cpp`** to consume them → temporal & spatial locality (per-warp and per-thread), plus the new **coalescing-efficiency** and **divergence** metrics (§4). *Reuse DAMOV's actual code here* — it's architecture-independent.
5. **NVBit (Accel-Sim Tracer)** → SASS traces of the kept kernels.
6. **Accel-Sim / GPGPU-Sim 4.x** → run each kernel on **Baseline-GPU** and **NDP-GPU** configs across the **occupancy / #SM / #CTA** sweep (§5.4). Collect L1/L2 misses, IPC, achieved BW. Compute **LFMR_gpu = L2miss/L1miss**, **L2 MPKI**, **AI** (§5.2–5.3).
7. **AccelWattch** → per-config energy (§5.5) for the host-vs-NDP energy argument.
8. **Accel-Sim Correlator** → validate Baseline-GPU sim stats against the Step-3 NCU runs (§5.6).
9. **Cluster** (k-means + hierarchical, scikit-learn) on the GPU metric set {temporal locality, coalescing efficiency, divergence, AI, L2 MPKI, LFMR_gpu, LFMR_gpu-vs-occupancy trend} → **derive the GPU bottleneck classes empirically** and check them against the §6 hypothesis. Validate on held-out kernels exactly as DAMOV validated on its 100 held-out functions.
10. **Release** a GPU benchmark suite of representative kernels per class + the toolchain — a "GPU-DAMOV."

---

## 9. Challenges, pitfalls, and open problems

Honest obstacles, so you go in clear-eyed:

- **Trace-size explosion.** NVBit `mem_trace` runs ~100–1000× slower than native and emits enormous traces (a GPU executes billions of memory accesses per kernel). DAMOV capped CPU runs at ~200 M instructions; you'll need aggressive **kernel/region selection and sampling** for GPUs, plus lots of disk. Basic-block instruction counting (`instr_count_bb`, ~2–5×) is cheap; full memory tracing is the expensive part — budget for it.
- **NVBit and NCU/Nsys are mutually exclusive.** NVBit uses the same underlying mechanism as nvprof/Nsight, so you cannot run NVBit instrumentation and NCU profiling in the *same* execution. You run them in separate passes (fine, but plan the workflow).
- **Instruction-counting semantics.** Warp vs thread vs SASS instruction counts differ by up to 32×; pick one (warp-instruction is the cleanest analog) and be consistent, or your MPKI/AI numbers won't mean what you think.
- **Simulator fidelity lags the newest GPUs.** Accel-Sim/GPGPU-Sim are validated mainly through Volta/Turing/Ampere; **Hopper/Blackwell** features (4th/5th-gen Tensor Cores, TMA, thread-block clusters / distributed shared memory, FP8) are partially or not modeled. If your workloads lean on those, simulator accuracy drops. (Accel-Sim v1.3.0, Feb 2025, targets H100-class but bleeding-edge features remain a gap.)
- **Defining the "NDP-GPU" config is a modeling judgment, not a fact.** Just as DAMOV's NDP config was a deliberate, debatable choice (same cores, delete L2/L3, pimMode), your GPU-NDP config encodes assumptions about bandwidth, latency, and what compute lives near memory. State them; run sensitivity analyses.
- **Inter-kernel data movement is worse on GPUs.** DAMOV's function-level limitation (it ignores data moved *between* functions) is amplified on GPUs, where the natural unit is a *kernel*, data persists in global memory across kernels, and host↔device transfers matter. A faithful GPU study should treat *kernel-to-kernel* and *host-to-device* data movement explicitly (Nsys helps here).
- **The NDP upside is smaller and more conditional on GPUs.** Because HBM already delivers ~TB/s, don't expect DAMOV's clean "1a → big NDP win" story to dominate. Expect the value to concentrate in **capacity-bound** (LLM/embedding), **sparse/irregular** (coalescing-broken), and **energy** regimes. This reshapes the *conclusions*, not just the metrics.
- **Energy-model extension effort.** AccelWattch is validated for *conventional* GPUs; modeling the energy of an in-/near-memory substrate requires extending its component model, which is real work and a source of uncertainty (the same caveat DAMOV's energy model carries).

---

## 10. What you'd reuse vs. rebuild — summary

| DAMOV piece | Reuse / Adapt / Rebuild for GPU |
|---|---|
| Locality analysis (`locality.cpp`) | **Reuse** (architecture-independent) — add granularity choice + coalescing/divergence |
| The 3-step *methodology shape* | **Reuse** (screen → cluster by locality → simulate & classify) |
| Clustering (k-means/hierarchical) | **Reuse** verbatim |
| LFMR, MPKI, AI definitions | **Adapt** (L2-based LFMR; warp-instruction MPKI; AI from roofline) |
| Step-1 profiler | **Replace** VTune → NCU + Nsys (and you get bandwidth-vs-latency for free) |
| Simulator | **Replace** ZSim+Ramulator → Accel-Sim/GPGPU-Sim 4.x (+ DRAMsim3 for PIM) |
| Energy model | **Replace** CACTI pJ → AccelWattch (then extend for PIM) |
| Trace capture | **Replace** Pin → NVBit (Accel-Sim Tracer + mem_trace) |
| The six-class taxonomy | **Rebuild** — re-derive from GPU data (latency class collapses; coalescing/L2-contention/occupancy classes emerge) |
| The NDP value argument | **Rebuild** — shift from bandwidth to capacity + energy + irregular access |

---

## 11. A minimal first project (to de-risk the idea)

If you want a tractable starting point rather than the full pipeline, do this scoped study first:

1. Pick **~10 kernels** spanning obvious extremes: a few dense GEMM/conv (expected compute-bound), a few STREAM-like/SpMV/graph kernels (expected bandwidth/irregular), one low-occupancy latency-exposed kernel.
2. Profile all with **NCU** (SoL, stalls, memory workload, roofline) — get the first-cut labels and the real-hardware ground truth.
3. **NVBit `mem_trace`** + ported `locality.cpp` on the same 10 → locality + coalescing + divergence.
4. **Accel-Sim baseline** run + **Correlator** to confirm the sim matches NCU on these 10.
5. Add *one* simple **NDP-GPU config** (reduced L2 + faster/wider HBM model) and measure the per-kernel delta.
6. Plot the 10 kernels on {LFMR_gpu, AI, coalescing efficiency} and see whether they separate the way §6 predicts.

If those 10 cluster sensibly and the simulator correlates with NCU, you've validated the whole approach and can scale to a full benchmark suite.

---

## References and tools (with links)

**The DAMOV original**
- Oliveira et al., "DAMOV: A New Methodology and Benchmark Suite for Evaluating Data Movement Bottlenecks," IEEE Access, 2021. Code: https://github.com/CMU-SAFARI/DAMOV

**GPU instrumentation (Pin analog)**
- NVBit (NVIDIA Binary Instrumentation Tool), NVlabs: https://github.com/NVlabs/NVBit — Villa et al., "NVBit: A Dynamic Binary Instrumentation Framework for NVIDIA GPUs," MICRO 2019.
- NVBit tutorial (overheads, mem_trace, instr_count): https://eunomia.dev/others/nvbit-tutorial/

**GPU simulation (ZSim+Ramulator analog)**
- Accel-Sim framework: https://accel-sim.github.io/ — Khairy et al., "Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling," ISCA 2020. (Tracer = NVBit tool; SASS frontend; Correlator; Tuner. v1.3.0, Feb 2025; AMD-GCN support.)
- GPGPU-Sim 4.x: https://github.com/accel-sim/gpgpu-sim_distribution (cycle-level; runs NVBit SASS traces; AccelWattch energy in 4.2+).

**GPU power model (energy-model analog)**
- AccelWattch: https://accel-sim.github.io/accelwattch.html — Kandiah et al., "AccelWattch: A Power Modeling Framework for Modern GPUs," MICRO 2021 (validated vs Volta; SASS+PTX; divergence; DVFS).

**GPU profiling (VTune analog)**
- Nsight Compute (per-kernel: Speed-of-Light, Warp State/stalls, Memory Workload Analysis, Roofline, Source Counters): https://developer.nvidia.com/nsight-compute and Profiling Guide: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- Nsight Systems (system-level timeline): https://developer.nvidia.com/nsight-systems
- NVIDIA, "The Peak-Performance-Percentage Analysis Method for Optimizing Any GPU Workload" (the SoL%/stall heuristic): NVIDIA Developer Blog.

**Memory model for PIM (Ramulator analog)**
- DRAMsim3 (used by HBM-PIM works for bank-level compute); Ramulator2.

**GPU near-data / processing-in-memory (the NDP targets DAMOV cites but doesn't model)**
- Hsieh et al., "Transparent Offloading and Mapping (TOM): Enabling Programmer-Transparent Near-Data Processing in GPU Systems," ISCA 2016.
- Zhang et al., "TOP-PIM: Throughput-Oriented Programmable Processing in Memory," HPDC 2014.
- Pattnaik et al., "Scheduling Techniques for GPU Architectures with Processing-in-Memory Capabilities," PACT 2016.
- Samsung HBM-PIM (Aquabolt-XL) and recent LLM-inference-on-HBM-PIM works (e.g., HPIM, 2025) for the capacity-bound case.
- Corda et al., "Platform-Independent Software Analysis for Near-Memory Computing," arXiv:1906.10037 — precedent for platform-independent NDP characterization in the DAMOV lineage.

**AMD path (alternative ecosystem)**
- Profiling: ROCprof / Omniperf. Simulation: Accel-Sim AMD-GCN traces, or MGPUSim.

---

*Bottom line: porting DAMOV to GPUs is a "glue mature tools in the same topology" problem for the infrastructure (NVBit + Accel-Sim + AccelWattch + NCU/Nsys are a drop-in stack), and a "re-derive the bottleneck science" problem for the taxonomy — because GPUs hide latency with parallelism, cache in two levels, live or die on coalescing, and already own HBM. The locality analysis ports for free; the six classes do not.*
