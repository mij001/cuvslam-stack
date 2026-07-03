# Project Status — cuVSLAM Memory Characterization for PiM/ISP

**Living document.** Snapshot of what exists, what it found, and what remains.
Updated as work lands. Companion docs: `PROFILING_PLAN.md` (strategy),
`METHODOLOGY.md` (how numbers are made), `PUBLISHABILITY.md` (reviewer-grade
issue register), `WALKTHROUGH.md` (guided tour), `reports/` (committed results).

- **Last updated:** 2026-07-03 13:00
- **Head commit:** see changelog
- **Branch:** `main` (pushed to origin/mij001)

---

## 1. The goal (one paragraph)

Motivate memory-centric hardware — Processing-in-Memory (PiM) and In-Storage
Processing (ISP) — for Physical AI, by producing a rigorous **per-kernel memory
characterization of cuVSLAM** (NVIDIA's production visual-SLAM stack). For each
CUDA kernel: is it compute- or memory-bound, what is its access pattern, which
data does it touch, how often — and therefore which memory-centric substrate (if
any) it wants. The thesis under test is a three-way **persistence taxonomy**:
streaming (per-frame images → near-sensor SRAM), hot-persistent (local map,
every frame → DRAM-PiM), cold-persistent (keyframe DB, only on loop closure →
ISP). Target: an ISPASS/IISWC-grade characterization, then a MICRO/ASPLOS/
ISCA/HPCA architecture paper built on it.

---

## 2. Current stage

**Slice 2 complete + publishability-hardened; multi-machine + multi-dataset
generalization in progress.**

| Slice | What | State |
|---|---|---|
| Slice 0 | Researched plan, hw descriptors, DAMOV→GPU adaptation study | ✅ done |
| Slice 1 | Working nsys/ncu harness on the TOML runner; ncu "no report" bug fixed | ✅ done |
| Slice 2 | Headless analysis layer → DAG, screen, roofline, bandwidth, **GPU-DAMOV classification**, report | ✅ done |
| Rigor | Measured ceilings, ±25% sensitivity, ×5/×3 variance, clock-warmup protocol, cache bracket, transfers | ✅ done |
| Multi-machine | RTX 2000 Ada workstation production pass — **clocks LOCKED** (1620/7001, root-configured), ceilings measured at lock (205.0 GB/s ±0.1, 5445 GFLOP/s ±3) | 🟡 captures running |
| Multi-dataset | KITTI color stereo + TUM-VI fisheye matrix | 🟡 configs validated, captures staged |
| Slice 3 | NVBit → locality → Accel-Sim data-movement track | 🔴 gated (driver > 575 on both hosts) |
| Source-level | TaggedAllocator + NVTX (data-structure attribution) | ⬜ not started (needs from-source build) |
| Phase 4 | PiM/ISP substrate design + simulated eval | ⬜ future |

---

## 3. What has been built (the tooling)

All **stdlib-only, headless, GPU-optional** — analysis reruns from committed
CSVs with no dataset and no GPU.

### Harness (`harness/profile.py`)
- One entrypoint wrapping `run.py <config>` under nsys or ncu into a versioned
  results dir with mandatory `metadata.json` provenance.
- Targeted ncu metric sets (`quick`/`roofline`/`characterize` — 29 counters);
  **never `--set full`** (the original bug that produced empty reports on 2 GB GPUs).
- `--auto-window` derives a steady-state ncu launch window from a prior nsys run.
- `--cache-control {all,none}` cold/warm bracket; `--kernel-filter` regex;
  `--gpu-warmup` fixed clock warm-up; `${CUVSLAM_DATASETS}` portability.

### Analysis (`analysis/`, stdlib + hand-emitted SVG)
| Module | Produces |
|---|---|
| `stages.py` | kernel → pipeline-stage → persistence-class taxonomy (regex over 47 kernels) |
| `build_dag.py` | per-stage GPU-time share, kernels/frame |
| `screen.py` | DAMOV Step-1 memory-bound screen + 9-bucket stall taxonomy + LFMR/MPKI |
| `roofline.py` | FLOP-counter arithmetic intensity vs **measured** ceilings |
| `bandwidth.py` | per-stage DRAM traffic, achieved GB/s vs ceiling |
| `classify.py` | **GPU-DAMOV G1–G7 classes → per-kernel PiM/ISP affinity** + confidence + stability |
| `transfers.py` | host↔device data movement (the copies the kernel view misses) |
| `variance.py` | run-to-run CoV (absolute time + clock-invariant share) |
| `make_report.py` | the committed markdown report + SVG figures + CSVs |

### Environment (`env/`)
- `gen_hw_descriptor.py` — auto hw TOML from `cudaDeviceGetAttribute` (exact
  structural values; ceilings flagged for measurement).
- `measure_ceilings.py` — **measured** DRAM BW (D2D memcpy) + FP32 (cublasSgemm)
  ceilings; desktop-safe (queue depth 1, VRAM-budgeted, display-active refusal).
- `gpu_warmup.py` — pre-capture clock stabilization for unlocked-clock hosts.
- `check_env.sh` preflight · `fetch_datasets.sh` (integrity-verified) ·
  `lock_clocks.sh` · `setup_perms.sh`.

### Orchestration & tests
- `run_characterization.sh` — one command: preflight → ×5 nsys → sync+async SLAM
  → cold/warm ncu bracket → report + variance.
- `tests/test_analysis.py` — 10 GPU-free tests (unit norm, name-join, decision
  tree, PiM affinity, from-CSV reproduction).
- `blocked/` — Slice-3 NVBit/Accel-Sim runners gated behind `check_capability.sh`.

---

## 4. What has been found (the science)

Prototype pass on the MX450 (TUM fr3 long_office, RGBD). **All headline claims
survived the rigor pass** (brackets, variance, measured ceilings). Full report:
`reports/2026-07-02_tum_office_mx450/`.

### The GPU-DAMOV taxonomy (derived, not just ported)
The CPU DAMOV 6-class tree does not port unchanged (GPUs hide latency with
occupancy, cache in two levels, live/die on coalescing). Implemented classes:
**G1** DRAM-bandwidth · **G2** coalescing/scatter · **G3** L2-reuse · **G4**
latency-at-low-occupancy · **G5** compute · **G6** on-chip · **G7**
dependency/ILP. G7 **emerged from the data** (the hypothesis table missed it),
as the adaptation doc prescribed.

### Per-stage verdicts
| Stage | Persistence | Class | PiM/ISP verdict |
|---|---|---|---|
| preprocess, feature_detect | streaming | G1 | **strong — near-sensor SRAM** (DRAM saturated, L2 not helping) |
| loop closure (`st_track_with_cache`) | cold-persistent | G2 | **strong — ISP / near-storage scan** (23.5 MB/launch scattered, 3% occ) |
| bundle adjustment | hot-persistent | G2 | **conditional — scatter-capable PiM** (or layout fix first) |
| tracking matchers, ba_solver | — | G7 | none — host GPU (fix occupancy first) |

### The three sharpest results
1. **Loop closure is the ISP target, measured.** `st_track_with_cache` = 69% of
   SLAM GPU time (sync) / **50.6% (async/deployment)** — scattered 18–25
   sectors/req over 23.5 MB/launch at 3% occupancy: a working set that defeats
   every GPU cache on every rare touch.
2. **The PiM boundary runs *through* bundle adjustment.**
   `reduced_system_stage_2` is G3 (LFMR 0.05 cold → 0.03 warm — its L2 reuse is
   real, PiM would forfeit it) while sibling kernels are G2 scatter. "BA is
   memory-bound" is too coarse — a data-structure-level distinction.
3. **Host↔device transfers are first-order.** Explicit copies = 41% of kernel
   GPU time; H2D = 1.68 MB/frame = the raw sensor upload a near-sensor substrate
   eliminates outright.

### Measured rigor (this is what makes it defensible)
- Ceilings: **measured** 45.7 GB/s DRAM / 1228 GFLOP/s FP32 (not 80/3760 spec).
- Determinism: kernel instance counts stable to **0.13%** across 5 runs.
- Variance: `--gpu-warmup` collapses run CoV 49.6→9.3% (time), 16.3→5.8%
  (share); ncu counter ratios stable at 1–5% CoV on headline kernels.
- Sensitivity: 36/47 kernels keep their class under ±25% threshold perturbation;
  **all headline kernels stable**; borderline ones flagged.

---

## 5. Hardware / infrastructure reality

| Host | GPU | Role | Notes |
|---|---|---|---|
| laptop `iNOMAL` | MX450 (sm_75, 512 KiB L2, 64-bit) | prototype | can't lock clocks → methodology-grade numbers only; CUDA install self-repaired via `~/.local/cuda-repair` |
| workstation `dell-workstation` | RTX 2000 Ada (sm_89, 12 MB L2, 128-bit, ECC) | **production** | reachable via `ssh ndpvslam@dell-workstation`; driver 610 → Slice-3 still gated; clock-lock needs user sudo |

Datasets on the workstation: TUM RGBD (fetching), **KITTI seq06 color stereo**
(`kitti_dataset_for_checking`), **TUM-VI corridor1 fisheye** (`av_dataset`) —
symlinked into `${CUVSLAM_DATASETS}` as `kitti_color`, `tumvi`.

---

## 6. In flight right now

1. **Workstation production pass** (bg `b7hkyxaji`): venv ✅ → TUM fetch (in
   progress) → measured ceilings → full characterization → SLAM. Gives the first
   **locked-hierarchy** numbers (12 MB L2 vs 512 KiB): does the loop-closure
   working set still defeat the cache at 24× the size? That's the core PiM
   argument's stress test.
2. **Multi-dataset matrix** (staged `~/matrix_chain.sh`): KITTI color stereo +
   TUM-VI fisheye × {odometry, SLAM}. Configs validated against real data.
   Answers "is this cuVSLAM or just this configuration of it?"

---

## 7. Roadmap (ordered by publication value)

| # | Milestone | Unblocks | Status |
|---|---|---|---|
| 1 | Workstation locked-clock pass | publishable absolute numbers | 🟡 running |
| 2 | 3-dataset matrix + `analysis/compare.py` cross-dataset agreement | generalization claim | 🟡 next |
| 3 | Slice-3 (NVBit locality + Accel-Sim) | reuse-distance, divergence, sim deltas | 🔴 driver-gated |
| 4 | TaggedAllocator + NVTX from-source build | **data-structure-level** claims | ⬜ |
| 5 | k-means over metric vectors | taxonomy *validated* not asserted | ⬜ (needs ≥3 datasets) |
| 6 | PiM/ISP substrate model + AccelWattch energy | the architecture paper | ⬜ |

**Venue framing:** (1)+(2) → ISPASS/IISWC characterization paper. +(3)+(4) →
data-structure-level motivation. +(6) → MICRO/ASPLOS/ISCA/HPCA. See
`PUBLISHABILITY.md` for the full reviewer-issue register.

---

## 8. Changelog

- **2026-07-03 (pm)** — Power cut killed the first workstation chain (lesson:
  remote chains now run under `setsid nohup`). Root access used to configure
  passwordless sudo for `ndpvslam`; **GPU clocks locked** (persistence on,
  1620/7001) — ceilings measured at lock: 205.0 GB/s DRAM (0.05% trial spread
  vs 25% on the unlocked laptop), 5445 GFLOP/s FP32. Full program relaunched
  disconnect-safe: TUM characterization (×5 + sync/async SLAM + cold/warm ncu
  bracket) → SLAM-kernel ncu → KITTI + TUM-VI matrix.
- **2026-07-03** — Workstation access; datasets located; matrix configs (KITTI
  color, TUM-VI) validated. Rigor pass committed: measured ceilings, ±25%
  sensitivity, variance protocol, cache bracket, transfers. Report regenerated.
  Workstation production pass launched. This status doc created.
- **2026-07-02** — GPU-DAMOV classification (`classify.py`, G1–G7) + PiM/ISP
  candidate report. First full characterization (TUM long_office, MX450). Slice-2
  analysis layer, portable configs, env tooling, GPU-free tests.
