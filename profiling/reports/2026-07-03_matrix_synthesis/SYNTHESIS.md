# Matrix Synthesis — cuVSLAM across 2 GPUs × 3 workloads (2026-07-03)

The cross-cutting analysis over the four committed reports:
`2026-07-02_tum_office_mx450` (prototype), `2026-07-03_tum_office_rtx2000ada`,
`2026-07-03_kitti06_rtx2000ada`, `2026-07-03_tumvi_corridor1_rtx2000ada`
(production, **locked clocks 1620/7001**). Data: `class_agreement.csv`,
`clusters.csv`, per-report `data/`. All workstation captures ran between the
two power failures at locked clocks; provenance in each run's `metadata.json`.

## 1. Measurement quality ladder (measured, one workload)

| Protocol | absolute-time CoV (median, 5 repeats) |
|---|---|
| laptop, unlocked clocks | 49.6% |
| laptop, unlocked + 8 s warm-up | 9.3% |
| **workstation, locked clocks** | **0.14%** (max 0.79%) |

Ceilings at lock: DRAM 205.0 GB/s (min–max spread 0.1), FP32 5445 GFLOP/s (±3).
These are the numbers that make the rest publishable.

## 2. The headline generalizes: loop-closure matching dominates GPU time

`st_track_with_cache_kernel` (keyframe-DB scan), share of total GPU time, Ada,
sync SLAM:

| Workload | scale | share | launches | avg launch |
|---|---|---|---|---|
| TUM office (RGBD) | room, 2488 fr | 51.7% | 383 | ~2.6 ms |
| TUM-VI corridor1 (fisheye) | building, 5991 fr | 40.9% | 485 | 2.7 ms |
| KITTI 06 (stereo) | street, 1101 fr | **62.8%** | 196 | **7.4 ms** |

One kernel family is the single largest GPU consumer on every workload and
both GPUs (laptop async mode: 50.6%). Always ~2% occupancy, always scattered.

## 3. The working-set / L2 crossover — the PiM/ISP design argument, quantified

`st_track_with_cache` per launch:

| Point | L2 | DRAM traffic/launch | L2 hit | sectors/req |
|---|---|---|---|---|
| MX450, TUM office | 0.5 MB | 23.5 MB | 85.9% | 18 |
| Ada, TUM office | 24 MB | 2.9 MB | 91.3% | 19 |
| Ada, TUM-VI corridor | 24 MB | 2.7 MB | 79.7% | 7 |
| Ada, KITTI 06 street | 24 MB | **10.4 MB** | 97.8% | **30** |

Reading: the scan **footprint** (~23 MB at room scale, laptop traffic ≈
footprint when the cache is negligible) sits at the Ada's L2 capacity *today*;
the outdoor map already pushes 3.6× more DRAM traffic per scan at maximal
scatter (30/32 sectors — fully uncoalesced), and share-of-GPU-time *grows*
with scene scale (62.8% on KITTI). Benchmark maps are minutes long;
Physical-AI deployments (city-scale, multi-session) grow this DB by orders of
magnitude, monotonically leaving any cache. The class flip the classifier
reports across these points (G2 scatter ↔ G3 L2-reuse) is not noise — it
**tracks the working-set/cache ratio**, which is precisely the design
parameter. ISP/near-memory verdict: scale-dependent, with the crossover now
measured at ~24 MB per scan.

## 4. Classification agreement (generalization evidence)

Decomposed (`class_agreement.csv`):

| Comparison | kernels agreeing | time-weighted |
|---|---|---|
| Ada: TUM ↔ KITTI (RGBD vs rectified stereo) | 37/42 | **97%** |
| Cross-GPU: TUM on MX450 ↔ Ada | 36/47 | **100%** |
| Ada 3-way incl. TUM-VI | 23/43 | dominated by the §3 crossover flip |

Flips concentrate in (a) sub-0.5 ms helper kernels already flagged
`borderline` by the ±25% sensitivity, and (b) physically meaningful
hardware-tracking flips: on the 205 GB/s / 24 MB-L2 part, `cast/conv`
preprocess kernels shift G1-bandwidth → G5/G6 (the bandwidth wall receded) —
on big GPUs the near-sensor argument correctly rests on the **transfer/energy
side** (H2D = the sensor upload; 41% of kernel time in copies on the laptop
pass), not on DRAM bandwidth. Edge silicon (Jetson-class, thin buses) looks
like the MX450 column.

## 5. Does the taxonomy fall out of the data? (preliminary)

Stdlib k-means over the Ada TUM feature vectors (`cluster_sweep.csv`): best
silhouette at **k=7** — the taxonomy's class count — purity 0.67 vs the tree.
Consistency-level support; the full validation (pooled matrix + refined
features) is roadmap item 5.

## 6. What this closes / what remains

- ✅ PUBLISHABILITY issue 1 (unlocked clocks): closed — locked-clock pass done.
- ✅→🟡 issue 2 (single workload): 3 workloads × 2 GPUs captured; EuRoC stereo
  still absent (server down) — matrix is defensible as-is.
- Remaining top items: Slice-3 traces (driver-gated), TaggedAllocator
  (data-structure attribution — would turn §3's footprint inference into a
  direct measurement), pooled clustering, energy.
