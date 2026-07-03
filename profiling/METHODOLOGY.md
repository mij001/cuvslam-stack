# Measurement & Classification Methodology

The normative description of how every number in `reports/` is produced —
written so a paper's Methodology section can be drawn from it directly, and so
a reviewer (or artifact evaluator) can audit each step. Companion documents:
`PUBLISHABILITY.md` (the open-issues register) and
`suggestions_and_summuries/Adapting_DAMOV_to_GPU.md` (the taxonomy derivation).

## 1. Measurement stack

| Layer | Tool | What it provides |
|---|---|---|
| Timeline / launch counts | Nsight Systems (CLI) | stage decomposition, kernels/frame, steady-state windowing |
| Per-kernel counters | Nsight Compute (CLI), targeted 29-metric `characterize` set | SoL, stalls, hit rates, bytes, FLOPs, coalescing |
| Ceilings | `env/measure_ceilings.py` (ctypes, no compiler) | measured DRAM BW (D2D memcpy) + FP32 (cublasSgemm) |
| Provenance | `metadata.json` per run | GPU, driver, clocks at launch, tool versions, config hash-by-copy, launch window |

Tool versions are recorded per run in `metadata.json`; the workload is always
`run.py <config>` with the exact resolved config stored beside the results.

## 2. Aggregation rules

- **Per-kernel rows** aggregate all profiled launches of the same (demangled,
  template-stripped) kernel name: percentages and ratios are **time-weighted
  means**; byte/FLOP counters are **sums**; min/max Memory-SoL across launches
  is retained as the spread.
- Kernel names are normalized across profilers (`cuvslam::cuda::X` in nsys vs
  `cuvslam::X` in ncu 2026.2 → `X`), validated by tests.
- Stage rollups weight by GPU time from the nsys kernel summary.

## 3. Metric definitions (GPU-DAMOV adaptation)

| Metric | Definition | Source counters | Known bias |
|---|---|---|---|
| Memory/Compute SoL | % of peak sustained, whole-kernel | `gpu__compute_memory_throughput`, `sm__throughput` | robust under replay |
| DRAM-SoL | % of *theoretical* peak DRAM throughput | `dram__throughput...pct_of_peak` | vs the *measured* ceiling (~82% of theoretical here) this UNDERSTATES saturation → G1 detection is conservative |
| LFMR_gpu | L2-misses / L1-misses ≈ **1 − L2 sector hit rate** (L2 accesses ≈ L1 misses) | `lts__t_sector_hit_rate` | ncu `--cache-control all` flushes caches per replay pass → cold-start hit rates → LFMR biased HIGH. Bracketed by a `--cache-control none` capture (warm, biased LOW). Truth lies between; Slice-3 simulation resolves it. |
| MPKI_gpu | DRAM sectors (32 B) per kilo **warp**-instruction | `dram__bytes_*`, `smsp__inst_executed` | instruction unit stated explicitly (warp ≠ thread, 32× difference) |
| AI | FP32 FLOPs / DRAM byte; FLOPs = fadd + fmul + 2·ffma (SASS `pred_on` counters, Yang20) | `smsp__sass_thread_inst_executed_op_*` | excludes int/tensor ops → lower bound for integer-heavy kernels |
| Coalescing | sectors per LSU global request (4 = fully coalesced, 32 = fully scattered) | `l1tex__average_t_sectors_per_request...` | LSU global only; tex-pipe kernels report 0 (n/a) |
| Working set proxy | DRAM bytes per launch vs L2 size | `dram__bytes_*` | traffic, not footprint: a streaming kernel can exceed L2 without a resident set |

## 4. Capture protocol

1. **Steady state**: track ≥200 frames before profiling (map warmed); the ncu
   window is derived as `kernels/frame × warm-frames` from a prior nsys run of
   the same config (`--auto-window`). Per-kernel launch counts inside the
   window are recorded; kernels with n < 5 launches are confidence-capped.
2. **Clocks**: laptops cannot lock clocks — `--clock-control none` is passed to
   ncu, clocks at run start recorded in metadata, and ceiling measurements
   record clocks before/mid/after. Workstation runs lock clocks
   (`env/lock_clocks.sh`) per the hw descriptor.
   **Warm-up protocol (unlocked hosts):** every capture is preceded by a fixed
   8 s clock warm-up (`--gpu-warmup 8` → `env/gpu_warmup.py`) so runs start at
   the same sustained operating point. Measured motivation: identical
   back-to-back runs starting from deep idle (300/405 MHz) vs active idle
   (1035/3500 MHz) differed **3.4× in GPU time**, and because memory and core
   clocks scale differently under DVFS, memory-bound kernels shifted even in
   relative share (share CoV up to 35%). Statistic hierarchy on unlocked
   clocks: per-kernel counter *ratios* (hit rates, sectors/req, SoL) >
   time *shares* > absolute times (unusable).
3. **Cache bracket**: hit-rate-dependent metrics are captured twice —
   `--cache-control all` (cold) and `none` (warm) — and reported as a bracket.
4. **Repeats**: nsys captures ×5, ncu captures ×3; `analysis/variance.py`
   reports per-kernel CoV for BOTH absolute time and time share (see the
   statistic hierarchy in §4.2). Instance-count CoV ≈ 0 is the
   workload-determinism check — measured max 0.13% on TUM (`async_sba=false`;
   only adaptive matcher iteration counts jitter).
5. **Loop closure**: `[slam]` workloads run in sync mode (deterministic
   attribution) AND async mode (deployment share); both captures feed the
   report.

## 5. Ceilings

Rooflines and PiM verdicts use **measured** ceilings where available
(`dram_gbps_measured`, `fp32_gflops_measured` in `hw/*.toml`), produced by
`env/measure_ceilings.py`: median of ≥7 trials, queue-depth-1 (sync per op,
≤2% underread), VRAM-budgeted, clock-sampled. GEMM sizes are powers of two
(odd tiles hit degenerate cuBLAS kernel selections — n=1536 measured 2.5×
below n=2048 on the MX450). The theoretical values remain in the descriptor
for reference; figures label which kind they used.

## 6. Classification protocol

Decision tree over the §3 metrics (`analysis/classify.py`), taxonomy from the
GPU-DAMOV adaptation study (§6) plus the emergent G7 (dependency-bound) class.
Thresholds are declared in one place (`THRESHOLDS`) and stress-tested: every
kernel is re-classified with all thresholds at ×0.75 and ×1.25; kernels whose
class flips are marked `borderline` and cannot carry `high` confidence.
PiM/ISP affinity combines the class with the stage's persistence hypothesis;
each verdict carries a plain-language rationale with the numbers that fired.

## 7. Threats to validity (paper checklist)

| Threat | Mitigation | Residual risk |
|---|---|---|
| Single-point hit rates under replay | cold/warm cache bracket (§4.3) | true steady-state needs Slice-3 sim |
| DVFS on unlocked laptop clocks | clock recording + workstation re-run policy | laptop numbers are methodology-grade only |
| Run-to-run variance | ×5/×3 repeats + CoV tables | tail kernels with CoV>10% flagged, excluded from headline claims |
| Threshold arbitrariness | ±25% sensitivity, borderline flags | thresholds still heuristic until Slice-3 clustering |
| Proxy LFMR/MPKI | derivation + bias direction stated (§3) | L2 sees non-LSU traffic; refine via NVBit traces |
| Kernel-level ≠ data-structure-level | stated in every report; TaggedAllocator milestone | claims stay kernel-scoped until then |
| Single workload mode (RGBD) | stereo configs ready (EuRoC/KITTI); dataset pending | generalization unproven until multi-dataset pass |
| sync-mode SLAM share inflation | async capture paired with sync (§4.5) | — |
| GPU-launch tax on tiny kernels | G0/G7 classes absorb them; no PiM claims made | — |
