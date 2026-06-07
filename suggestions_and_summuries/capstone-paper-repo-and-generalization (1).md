# Capstone: Paper × Repo × `ncu` — Two Hard Questions, a Verified Cross-Map, and a Reusable Profiling Playbook

*This document complements the two earlier files (`gpudb-measurement-report.md` — the full lecture with all 110 metrics; `calculated-metrics-walkthrough.md` — the derived-value maths). Here I (A) answer your two profiling-accuracy questions properly, (B) tie every result in the paper to the exact repo code and `ncu` counters that produced it, with honest notes where paper and code disagree, and (C) generalise the whole method so you can point it at a completely different workload. Written for an EEE undergraduate; no prior GPU knowledge assumed, but I won't re-teach what the earlier files cover.*

---

## PART A — Two profiling-accuracy questions

### A1. "Caches are dumped each round, so how can the L2 metrics be accurate?"

You've spotted the single most important subtlety in GPU profiling. Let me build the answer from the mechanism up.

**Why there are "rounds" at all (kernel replay).** The GPU has only a limited number of physical counter registers, and some counters interfere with others if read simultaneously. So `ncu` cannot collect all 110 metrics in one execution of a kernel. Instead it uses **kernel replay**: it runs the *same* kernel many times ("passes"/"rounds"), each pass measuring a different subset of metrics, then stitches the subsets into one report. For this to be valid, every pass must do *identical* work and start from an *identical* state — otherwise you'd be gluing together measurements of different things.

**What `ncu` does to the caches between rounds.** To force that identical starting state, `ncu` by default **flushes (invalidates) the L1 and L2 caches before every pass.** This default is the option `--cache-control all`. I checked the repo's command in `utility/ncu_profiler.py`:
```
ncu ... --metrics <list> --apply-rules yes --clock-control none -f -o gpudb-perf <bin>
```
It sets `--clock-control none` (don't touch the clock — it's pinned externally) but it **never sets `--cache-control`**, so the **default flush-everything behaviour is in effect.** So your premise is exactly right: every profiled kernel begins from a cold cache, every round.

**So what is actually being measured — and what isn't.** Flushing before each kernel removes one specific thing: **inter-kernel reuse** — data that a *previous* kernel left warm in L2 that the *current* kernel could have reused. It does **not** remove **intra-kernel reuse** — data the kernel itself loads and then re-reads during its own run. A cache that starts cold still warms up *during* the kernel and serves later accesses within that same kernel. So the reported hit rate reflects **"this kernel's own locality, measured from a clean slate."**

**Why that is still accurate — and even desirable — for this study.** Four reasons, in order of importance:

1. **The dominant kernel is huge and long-running, so the cold start is negligible.** Table 4 shows that for Crystal and HeavyDB, **>90% of the entire query runs in a single fused kernel** (Crystal's `probe_ht` is 99.35% of Q21, ≈3.96 ms ≈ 5.6 million cycles at 1.41 GHz). Filling a 40 MB L2 once at the start is a tiny transient compared with millions of cycles of steady-state operation. The measured hit rate is therefore dominated by that kernel's *intrinsic* intra-kernel behaviour, which a cold start captures correctly. Inter-kernel reuse simply isn't where the time is.
2. **The roofline doesn't use the hit *rate* — it uses the request *count*.** The L2 roofline's byte estimate is `lts__t_requests_srcunit_tex_op_read.sum × 128`. A *count of requests issued* is an additive, extensive quantity that is far less sensitive to warm-vs-cold state than a hit *ratio*. (Flushing can push a few extra requests down to L2 at the very start, but again that's amortised over a kernel running millions of cycles.)
3. **Flushing makes the measurement fair and reproducible — which is the whole point of a comparison study.** Because *every* kernel of *every* system is measured from the identical cold slate, no system is accidentally advantaged by leftover cache state from a previous step. Fairness across Crystal / HeavyDB / BlazingSQL / TQP matters more here than reproducing one system's absolute warm-cache number.
4. **The numbers themselves prove the measurement is stable.** The paper reports the same Crystal Q21 L2 hit rate as **51.38%** in Table 6 and **51.49%** in Table 7 (two independent profiling runs), and L1 requests as **19 M** vs **18 M**. A 0.11-percentage-point and ~5% spread is run-to-run *noise*, not cache-state chaos. If flushing were corrupting the measurement, you'd see large variance between runs; instead you see near-determinism — *because* flushing removes the uncontrolled variable (prior cache state). Your worry ("it must be inaccurate") would predict instability; the data shows the opposite.

**When flushing *would* bias you (so you know the limits).** If your workload were *many short kernels that each rely on a hot table a predecessor left in L2* (e.g., an iterative pipeline of tiny kernels over one resident lookup table), per-kernel flushing would **understate** the true hit rate and **overstate** DRAM traffic, because the real run keeps that table warm across kernels while the profiler keeps wiping it. SSB avoids this trap (few, long kernels), but a different workload might not — see Part C5.

**The knobs, and why they weren't used.** You can tell `ncu` *not* to flush, with `--cache-control none`, or to re-run the whole application per pass with `--replay-mode application` (which preserves real cross-kernel cache state). Both trade away determinism: with no flush, each pass sees a different cache state, so the stitched metrics no longer describe a single consistent execution, and you must average many runs to denoise. The authors chose the default (flush) because for SSB it costs almost nothing in accuracy (reasons 1–4) and buys clean, comparable, reproducible numbers.

**Bottom line.** The L2 metrics are accurate *as defined*: they measure each kernel's own locality from a cold start, consistently and reproducibly. For this workload that equals the real behaviour to within run-to-run noise, because the time lives in one long kernel whose steady state swamps the cold transient — and because the roofline leans on robust request *counts*, not fragile hit *rates*. For optimization validation (Crystal vs Crystal-Opt) the conclusion is even safer: both builds are flushed identically, so the *delta* (L1 0.47%→41.31%, L2 51%→67%) is valid even if an absolute number were slightly conservative.

---

### A2. "Are all the profiled metrics actually used in the calculations?"

**No.** Of the 110 counters in `utility/counter_config.py`, only a minority drive any number that appears in the paper. They fall into three buckets.

**Bucket 1 — Used directly in the derived calculations (~12 counters, in `stats/flush_ncu_csv.py`).** These are the only ones the Python actually does arithmetic on:

| Counter | Drives which calculated value |
|---|---|
| `gpu__time_duration.sum` | total time; throughput; top-kernel share |
| `smsp__cycles_elapsed.avg.per_second` | total integer ops |
| `smsp__sass_thread_inst_executed_op_integer_pred_on.sum.per_cycle_elapsed` | total integer ops (Fig 8 top) |
| `dram__bytes_read.sum` | total DRAM bytes (Fig 8 bottom); AI_dram |
| `lts__t_requests_srcunit_tex_op_read.sum` | L2 bytes (×128); AI_l2 |
| `smsp__inst_executed.sum` | per-kernel weight for stall averaging |
| 5 stall ratios: `long_scoreboard`, `wait`, `lg_throttle`, `drain`, `barrier` | the stall pie (Fig 9) |

**Bucket 2 — Read by hand for tables and ceilings (~10 counters).** Not processed by the Python, but reported in the paper after manual inspection of the human-readable export (`stats/ncu_export.py`):

| Counter(s) | Where it appears |
|---|---|
| `smsp__warps_active / warps_eligible / issue_active .avg.per_cycle_active` | Table 5 (Active / Eligible / Issued; Stalled = Active − Eligible) |
| `launch__registers_per_thread` | Table 6/7 (registers per thread) |
| `l1tex__t_sector_hit_rate.pct`, `lts__t_sector_hit_rate.pct` | Table 6/7 (L1/L2 hit %) |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | Table 6/7 (L1 request count) |
| `sm__sass_thread_inst_executed_op_integer_pred_on.sum.peak_sustained` | the compute ceiling β |
| (A100 datasheet DRAM bandwidth; not a counter) | the memory ceiling α |

**Bucket 3 — Collected but not in any reported number (~88 counters).** This is most of the list: the entire FP32/FP64 achieved-and-peak set; nearly all Speed-of-Light percentages; every per-pipe utilisation (ALU/FMA/LSU/ADU/CBU/XU/Uniform/Tex/FP16/FP64/Tensor×3); occupancy and block-limit metrics; all launch metrics except registers; **13 of the 18 stall reasons** (only 5 are plotted); the memory-width histogram (8/16/32/64/128-bit); local load/store traffic; L2 writes, L2 lookup hit/miss, DRAM writes; and the warp/instruction ratio extras.

**Why collect 88 things you won't put in a chart?** Three deliberate reasons:
1. **`--apply-rules yes` consumes them.** Nsight Compute's built-in "rules" engine turns raw counters into automatic guidance ("uncoalesced access detected", "low occupancy due to registers"). Those rules need the broad set, even if the paper's *figures* don't.
2. **One-shot diagnosis.** Re-profiling is expensive (you must re-run the database, re-warm, etc.); collecting broadly once means any follow-up question ("was it register-bound? was shared memory the issue?") is answered from the existing report without re-running.
3. **Confirming negatives.** The FP and tensor counters being ≈0 is *evidence* — it's how the authors justify using an **integer-only** roofline ("those functional units are not needed", §4.2.1). A negative result you can point to is worth collecting.

**The cost, and the tradeoff.** More distinct counters can mean more replay passes, which slows profiling. The authors accepted that because (a) database kernels are deterministic so replay is reliable, and (b) breadth-now is cheaper than re-profiling-later. The lesson for *your* future profiling: collect a wide net once, but know that only ~20 of the counters carry the headline story.

*(A nice tie-in to Part C: the FP32/FP64/tensor/shared-memory counters that are "wasted" for SSB are precisely the ones you would **promote into Bucket 1** for an ML or HPC workload. The harness is already workload-general; you just change which counters you do arithmetic on.)*

---

## PART B — The consolidated map: every paper result ↔ repo code ↔ `ncu` metrics

This is the "in conjunction with the paper and repo" view. Read each row as: *the paper shows X → the repo computes it here → from these counters → by this formula.*

| Paper artifact | What it shows | Repo code | Raw `ncu` inputs | How it's derived |
|---|---|---|---|---|
| **Fig 4** (cold breakdown: Compute/HtoD/DtoH/Compile/Other) | where cold time goes; justifies warm focus | **Nsight *Systems*** timeline (not the released `ncu` pipeline; see discrepancies) | CUDA API + memcpy trace | timeline segmentation |
| **Fig 5** (warm end-to-end + GPU time) | Crystal fastest; others 8–30× slower | each system's own timer (`run_query.py`); GPU time = Σ kernel durations | `gpu__time_duration.sum` (GPU bars); app timers (end-to-end) | hatched = end-to-end − GPU |
| **Table 3** (5 roofline metrics) | the metric definitions | `utility/counter_config.py` (`metric_roofline`) | the 5 listed counters | — |
| **Fig 6** (DRAM roofline) | most queries under-use DRAM BW | `flush_roofline` → `roofline_dram_*.txt` | IntRate, Clock, Duration, `dram__bytes_read.sum` | x = ops/bytes; y = ops/time/1e9 |
| **Fig 7** (L2 roofline) | optimised systems saturate L2 | `flush_roofline` → `roofline_l2_*.txt` | same ops; `lts__t_requests_srcunit_tex_op_read.sum` | L2 bytes = req×128; x = ops/L2bytes |
| **Fig 8** (total ops; total bytes) | BlazingSQL/TQP do far more work | `flush_inst`→`inst.txt`; `flush_bytes`→`bytes.txt` | IntRate·Clock·Duration; `dram__bytes_read.sum` | sum over kernels |
| **Table 4** (top-3 kernels, Q21) | kernel-fusion evidence (4 vs 172/190 kernels) | `flush_top_kernel`→`top_kernel_q21.txt` | `gpu__time_duration.sum` per kernel name | share = kernel_time/total_time |
| **Table 5** (scheduler warps) | issue-bound (Eligible≈0.16) | manual read of `ncu_export.py` | `smsp__warps_active/eligible/issue_active.avg.per_cycle_active` | Stalled = Active − Eligible |
| **Fig 9** (stall breakdown, Q21) | mostly Mem-LD stalls (BlazingSQL = compute) | `flush_stall`→`stall_q21.txt` | `smsp__inst_executed.sum` + 5 stall ratios | instruction-weighted, normalised to 100% |
| **Table 6** (memory stats) | Crystal bypasses L1 (0.47%) | manual read | `launch__registers_per_thread`, L1/L2 hit %, L1/L2 request counts | direct |
| **Fig 10 / Table 7** (Crystal vs Crystal-Opt) | 1.9× speedup; bytes/ops/hits improve | re-run pipeline on both builds | same counters as Fig 8 + Table 6 | before/after comparison |
| **Fig 11** (roofline shift after opt) | AI rises, dots move toward ceilings | `flush_roofline` on both builds | same as Fig 6/7 | AI_dram 1.64→2.54; AI_l2 0.166→0.271 |
| **§7 model** (t′, Slowdown, unified) | predict QPS vs resource allocation | **not in repo** (patent) | uses AI_dram, AI_l2, #IntOps, β, α | `t′=max(t, #Ops/(AI·BW′))`; `Slowdown=max(DRAM,L2)`; compute = 1/AllocRatio |
| **Fig 14** (QPS vs DoC, MIG) | DoC=2 best for Crystal; 1.5× | `concurrency_script/*_part_mig.py` | wall-clock via `perf_counter` | QPS = DoC·iter·13 / time |
| **Fig 15** (MIG vs MPS) | MPS wins small data; MIG wins large | `*_part_mig.py` vs `*_part_mps.py` | wall-clock | QPS comparison |

**Verified numbers worth pinning (now taken straight from the paper):**
- **Measured peak compute β = 18 247 Gops/s** (full GPU). *Correction to my earlier walkthrough:* I had illustratively estimated β ≈ 9.7 Tera-ops/s from an ALU count; the paper's *measured* value (via `peak_sustained × clock`) is **≈18.25 Tera-ops/s** — almost 2× higher. This is the best possible argument for *measuring* the peak with the counter rather than hand-calculating it.
- With β = 18 247 Gops/s and DRAM α ≈ 1 555 GB/s, the DRAM crossover is **AI\* = β/α ≈ 11.7 ops/byte** (I earlier said ~6.3 using the wrong β). Crystal Q21 at AI_dram = 1.64 is still far left of it ⇒ firmly memory-bound — the conclusion is unchanged, only the crossover value is corrected.
- **Compute-bound example (paper §7):** BlazingSQL Q34 sits at **AI = 27.15 ops/byte, throughput = 2886.74 Gops/s**; 27.15 > 11.7 ⇒ compute-bound, matching the paper's treatment.
- **Model accuracy:** correlation 0.95; relative error p50 ≤ 0.11, p95 ≤ 0.46, mean 0.15.

**Honest paper↔repo discrepancies (so you're never caught out):**
1. **`dram__bytes.sum` (paper Table 3) vs `dram__bytes_read.sum` (repo code).** The paper lists the generic byte counter; the released code sums *read* bytes specifically. Practically identical for these read-dominated queries, but the code is the ground truth.
2. **Stall-category labels.** The paper's Fig 9 legend reads {Mem LD, Compute, Mem Queue Full, Branch, **Scheduler Full**}; the released `flush_stall` uses {long_scoreboard, wait, lg_throttle, **drain**, barrier} labelled {Mem LD, Compute, Mem Queue Full, **Mem WB**, Branch}. The fifth category differs (released code has no "Scheduler Full" / drain↔Mem WB relabel). The dominant slice (Mem LD) is identical either way, so the conclusion holds; just don't expect the legend to match the script byte-for-byte.
3. **Nsight Systems for Fig 4 only.** The cold-time breakdown needs `nsys`; the released characterization pipeline is `ncu`-only and the `nsys` hooks in `profiler.cuh` are commented out. Fig 4 was produced with `nsys` directly.
4. **§7 analytical model and TQP are not in the repo** (Microsoft patent; TQP closed-source). You can reproduce Crystal/HeavyDB/BlazingSQL characterization and the MIG/MPS *measurements*; the *predicted* curves and TQP numbers come from the paper.
5. **Tiny run-to-run deltas** (Table 6 vs Table 7: 51.38 vs 51.49% L2 hit; 19 vs 18 M L1 req) — measurement noise, and (per Part A1) evidence the method is stable.

---

## PART C — Generalising the method to a *different* workload

The paper studies SQL on GPUs, but the **methodology is a general recipe for "why is my GPU code slow, and how will it scale when I resize the GPU?"** Here is how to lift it off SSB and onto anything.

### C1. What is workload-specific vs universal

| Universal (reuse as-is) | Workload-specific (must adapt) |
|---|---|
| Lock the clock; separate warm/cold; flush-cache replay | Which **operation type** is the AI numerator (integer? FP32? tensor?) |
| The two-/multi-level **roofline** and AI = ops/bytes | Which **memory levels** matter (add shared mem? NVLink?) |
| **Sum-then-divide** aggregation across kernels | Which **stall reasons** you expect to dominate |
| Scheduler/stall/occupancy **bottleneck drill-down** | The **source-level fix** the data points to |
| Closed loop: measure → diagnose → fix → re-measure | MIG/MPS profile choices for *your* sharing pattern |
| The AI-invariance assumption for the allocation model | Whether short-kernel inter-kernel reuse breaks the flush assumption |

### C2. The 8-step reusable playbook

1. **Fix the controls.** `nvidia-smi -lgc <clock>` to pin the clock; pick a representative input size that fits in memory without spilling; decide warm vs cold and isolate (use the warm-up-then-`-s <N>` skip trick for server/multi-process apps).
2. **Tier-0 — where does wall-clock go?** Run `nsys` (timeline) or app timers. If transfer/compile/setup dominate, your problem is *system software*, not the kernel; fix that first. If compute dominates, proceed.
3. **Tier-1 — choose the AI numerator and build the roofline.** Pick the op-type that matches the workload (table in C3), profile its *achieved* rate + *peak* + clock + per-level bytes, then plot ops/byte vs ops/s against the measured ceilings. Use one roofline per memory level that carries meaningful traffic.
4. **Diagnose bound-ness.** Compute AI\* = β/α for each level; left of it ⇒ memory-bound at that level, right ⇒ compute-bound. Note how far the dot sits below its ceiling (the unrealised headroom).
5. **Tier-2 — drill into the bottleneck.** Scheduler states (issue-bound?), stall reasons (which resource?), occupancy + block limiters (what caps warps?), memory efficiency (hit rates, coalescing, register spill). Same counters as the paper; workload-agnostic.
6. **Aggregate correctly.** For multi-kernel apps (an ML training step can be thousands of kernels), use sum-then-divide for extensive quantities; instruction-weight any ratio you average.
7. **Tie to source and fix.** Map each microarch symptom to a code cause (e.g., long-scoreboard stalls + low L1 hit ⇒ enable L1 / improve locality; low active-threads-per-inst ⇒ reduce divergence / terminate dead threads). Re-measure with the *same* counters to prove the delta — exactly the Crystal→Crystal-Opt loop.
8. **Model the resource allocation.** Treat AI as implementation-invariant; predict slowdown under a smaller partition as `#Ops / (AI × BW′)` for the bound level, combine levels with `max(...)`, and use `1/AllocRatio` when compute-bound. Validate against a few measured MIG/MPS points before trusting it for scheduling.

### C3. The one knob that must change: the AI "operation"

The harness already collects every op-type's achieved+peak counters (Part A2, Bucket 3), so retargeting is mostly *choosing which to divide by*:

| Workload | AI numerator op-type | Counter already in `counter_config.py`? | Likely dominant bottleneck |
|---|---|---|---|
| OLAP / analytics / SQL (this paper) | **integer** | yes (`...op_integer...`) | memory / L2 bandwidth |
| Dense linear algebra, HPC FP32 | **FP32 FMA** | yes (`...op_ffma...`, `...op_fadd...`, `...op_fmul...`) | compute (FMA pipe) or DRAM |
| Scientific FP64 | **FP64 FMA** | yes (`...op_dfma/dadd/dmul...`) | FP64 pipe (scarce on consumer GPUs) |
| Deep-learning training/inference | **tensor (FP16/TF32/INT8)** | partially (per-pipe `tensor_op_hmma/imma/dmma` utilisation; add the tensor *throughput* metrics) | tensor cores or HBM bandwidth |
| FFT / stencil / image filtering (EEE-relevant) | **FP32** + heavy **shared-memory** traffic | FP32 yes; **add** `l1tex__data_pipe_lsu_wavefronts_mem_shared*` / shared-mem throughput | shared-memory bandwidth / `mio_throttle` |
| Cryptography / hashing | **integer** (+ XU for some ops) | yes | compute (ALU/XU) |

### C4. Worked retargeting example — profiling a CNN inference pipeline

Suppose you want to profile a ResNet-style inference run instead of SSB. Apply the playbook:

- **Step 1 controls:** same clock lock; "scale factor" becomes *batch size* (pick the largest that fits without spilling — the paper's SF-16 logic). Warm = weights resident + plan compiled (e.g., TorchInductor/TensorRT engine built); cold = includes engine build + HtoD weight copy. Use the warm-up-then-skip trick because the framework fires setup kernels.
- **Step 3 numerator:** convolutions run on **tensor cores in FP16/TF32**, so the AI numerator switches from integer to **tensor/FP16 FMA**. The FP32/tensor counters that were *dead weight* for SSB now become the headline — a direct payoff of the broad collection in Part A2. Bytes: weights + activations from DRAM; add the **L2** roofline (activations often live in L2 between layers) and possibly a **shared-memory** level (im2col/implicit-GEMM tiles).
- **Step 4 diagnosis:** a well-tuned conv is usually **compute-bound on tensor cores** (dot right of AI\*); a memory-bound result instead points to small batch / activation streaming.
- **Step 5 stalls:** expect the dominant stall to shift from `long_scoreboard` (SSB's memory-load wait) to **`math_pipe_throttle`** (tensor pipe saturated — good) or **`mio_throttle`/`short_scoreboard`** (shared-memory pressure). Same Bucket-3 stall counters, different winner.
- **Step 6 aggregation:** an inference pass is hundreds of kernels (conv, bias, activation, pool); sum-then-divide exactly as for the 190-kernel TQP query.
- **Step 7 fix loop:** if `short_scoreboard`/shared-mem dominates, increase tile size or switch algorithm; if occupancy is register-limited (`launch__occupancy_limit_registers`), cut registers/thread — then re-measure the same counters to confirm, mirroring Crystal-Opt.
- **Step 8 allocation:** to serve *N* inference clients on one A100, model each on a MIG slice with `Slowdown_compute = 1/AllocRatio` (inference is compute-bound, so the compute branch of the unified model applies), and validate against a couple of measured DoC points — the same RQ1/RQ2 study, retargeted from queries-per-second to inferences-per-second.

*(Second quick example — an FFT or convolution-kernel in DSP: FP32 numerator, but add a **shared-memory roofline** because radix-FFT butterflies stage data in shared memory; the bottleneck and the relevant stall reason (`mio_throttle`) differ from SSB, but every other step is identical.)*

### C5. Pitfalls when porting (don't get burned)

1. **Short-kernel inter-kernel reuse breaks the flush assumption (Part A1).** If your workload is many *small* kernels sharing a hot table/tensor in L2, per-kernel cache flushing will understate hit rate and overstate DRAM bytes. Either accept the conservative estimate, or switch to `--cache-control none` / application replay and average several runs.
2. **Mixed precision needs a mixed numerator.** If a kernel does both INT and FP work (or FP16 *and* FP32), a single op-type underestimates AI. Profile each op-type's achieved rate and sum the relevant ones for the numerator.
3. **Tensor-core peak ≠ ALU peak.** Use the measured `peak_sustained` for the *actual* op-type (recall β was ~2× a naive estimate even for plain integers). Never hand-calculate the ceiling.
4. **Add memory levels the workload actually uses.** SSB ignored shared memory; DSP/DL often live there. The cache-aware "different AI per level" idea extends to L1, shared memory, and even NVLink/PCIe for multi-GPU — add a roofline per level that carries real traffic.
5. **The AI-invariance assumption can fail under *upsizing memory*.** The paper itself flags that the model is reliable for downsizing and for compute upsizing, but can mis-estimate when you *add* memory bandwidth that's no longer the bottleneck. Validate before trusting.

---

## PART D — One-page cheat sheet

- **Replay + flush:** `ncu` re-runs each kernel several times and flushes L1+L2 before each pass (default `--cache-control all`, which the repo leaves on). ⇒ metrics are **per-kernel, cold-start**: intra-kernel reuse captured, inter-kernel reuse not. Accurate here because one long fused kernel dominates and the roofline uses request *counts*; proven stable by 51.38% vs 51.49% across runs.
- **Metrics used:** ~12 feed the calculations, ~10 are read by hand for Tables 5–7 + ceilings, ~88 are collected for the rule engine / one-shot diagnosis / confirming FP-and-tensor-are-idle. The "unused" FP/tensor/shared-mem ones are exactly what you promote for ML/HPC.
- **The five derived quantities:** total ops (`IntRate·Clock·Duration`), total bytes (`dram__bytes_read.sum`), throughput (`ops/time`), AI per level (`ops/bytes`, with L2 bytes = `requests×128`), and the ceilings (β = measured `peak×clock` = **18 247 Gops/s**; α = datasheet BW; crossover AI\* ≈ **11.7** for DRAM).
- **The loop:** measure → roofline says *which* resource → stalls/scheduler say *why* → fix in source → re-measure same counters → model the partitioning with `max(Slowdown_DRAM, Slowdown_L2)` or `1/AllocRatio`.
- **To retarget:** keep every universal step; change only (a) the AI op-type, (b) the memory levels, (c) the expected stall winner, (d) the MIG/MPS pattern. The repo's `counter_config.py` is already general enough — you mostly change which counters you divide.
