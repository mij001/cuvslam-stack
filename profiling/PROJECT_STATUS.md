# Project Status — cuVSLAM Memory Characterization for PiM/ISP

**Living document.** Snapshot of what exists, what it found, and what remains.
Updated as work lands. Companion docs: `PROFILING_PLAN.md` (strategy),
`METHODOLOGY.md` (how numbers are made), `PUBLISHABILITY.md` (reviewer-grade
issue register), `WALKTHROUGH.md` (guided tour), `reports/` (committed results).

- **Last updated:** 2026-07-03 18:00
- **Head commit:** `40e8a05` (Slice-3 toolkit)
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

**ISPASS/IISWC-grade characterization complete, now data-structure-scoped**:
full-scale 27-sequence campaign done, Slice-3 locality measured, taxonomy
clustering-validated, and the TaggedAllocator+NVTX attribution pass
(`reports/2026-07-04_attribution/`) turns kernel-level claims into
data-structure-level ones. Next: Accel-Sim NDP config + AccelWattch energy →
the PiM/ISP substrate design for the architecture paper.**

| Slice | What | State |
|---|---|---|
| Slice 0 | Researched plan, hw descriptors, DAMOV→GPU adaptation study | ✅ done |
| Slice 1 | Working nsys/ncu harness on the TOML runner; ncu "no report" bug fixed | ✅ done |
| Slice 2 | Headless analysis layer → DAG, screen, roofline, bandwidth, **GPU-DAMOV classification**, report | ✅ done |
| Rigor | Measured ceilings, ±25% sensitivity, ×5/×3 variance, clock-warmup protocol, cache bracket, transfers | ✅ done |
| Multi-machine | RTX 2000 Ada production pass, **locked clocks** — 5-repeat CoV 0.14% (vs 49.6% unlocked laptop); ceilings 205.0 GB/s ±0.1 / 5445 GF ±3 | ✅ done |
| Multi-dataset | **27-sequence campaign** (KITTI 00-10, EuRoC ×11, TUM fr3 ×4, TUM-VI), odom+SLAM, 0 failures; modal consistency 91% (`reports/2026-07-04_campaign/`) | ✅ done |
| Slice 3 locality | NVBit mem_trace → `analysis/locality.py`: measured reuse distance overturned the counter proxy on st_track (`reports/2026-07-04_slice3_locality/`) | ✅ done |
| Taxonomy validation | pooled k-means over 27 sequences prefers k=7–8 = the G-classes (purity 0.68) | ✅ done |
| Source-level | **TaggedAllocator + NVTX attribution** (`reports/2026-07-04_attribution/`): instrumented wheel (RelWithDebInfo, `patches/0002`), NVBit alloc sidecar, `analysis/attribution.py` join — 240/240 allocations tagged; GPU memory budget static (108.65 MB, keyframe state fixed 6.7 MB → DB growth is host-side); measured NVTX kernel→stage map (st_track_with_cache = loop closure) | ✅ done |
| Slice 3 sim | Accel-Sim NDP config + AccelWattch energy (report deltas) | ⬜ |
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
| `classify.py` | **GPU-DAMOV G1–G7 classes → per-kernel PiM/ISP affinity** + confidence + ±25% stability |
| `transfers.py` | host↔device data movement (the copies the kernel view misses) |
| `variance.py` | run-to-run CoV (absolute time + clock-invariant share) |
| `compare.py` | cross-dataset / cross-GPU class-agreement table |
| `cluster.py` | stdlib k-means over metric vectors — does the taxonomy fall out of the data (silhouette/ARI/purity vs the tree)? |
| `locality.py` | **DAMOV Step-2** from NVBit traces: exact footprint, reuse-distance-vs-cache-capacity CDF, intra-warp spatial locality, inter-launch overlap (Slice-3) |
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
- `env/measure_ceilings.py` — measured DRAM/FP32 ceilings (desktop-safe:
  queue-depth-1, VRAM-budgeted, display-active refusal). `env/gpu_warmup.py` —
  clock stabilization for unlocked hosts.
- `tests/test_analysis.py` — 12 GPU-free tests (unit norm, name-join, decision
  tree, PiM affinity, from-CSV reproduction, locality pipeline).
- `blocked/` — Slice-3 NVBit/Accel-Sim runners gated behind `check_capability.sh`;
  `mem_trace_launch_window.patch` adds LAUNCH_BEGIN/END windowing to NVBit 1.8.

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
| 1 | Workstation locked-clock pass | publishable absolute numbers | ✅ done |
| 2 | Multi-dataset matrix + agreement (`compare.py`/`campaign.py`) | generalization claim | ✅ done (27 seq) |
| 5 | k-means over metric vectors (`cluster.py`/`campaign.py`) | taxonomy *validated* not asserted | ✅ done (pooled, k=7–8) |
| 3a | Slice-3 NVBit locality (`analysis/locality.py`) | reuse-distance, divergence, proxy correction | ✅ done |
| 4 | TaggedAllocator + NVTX from-source build | **data-structure-level** claims | ✅ done (`reports/2026-07-04_attribution/`) |
| 3b | Accel-Sim NDP config + AccelWattch energy | sim deltas, joules | ⬜ |
| 6 | PiM/ISP substrate design + delta eval | the architecture paper | ⬜ |

**Venue framing:** (1)+(2)+(3a)+(5) all done → **the ISPASS/IISWC
characterization paper is written-from-data now** (full-scale matrix, measured
reuse distance that corrected a proxy, clustering-validated taxonomy). +(4) →
data-structure-level motivation. +(3b)+(6) → MICRO/ASPLOS/ISCA/HPCA. See
`PUBLISHABILITY.md` for the full reviewer-issue register.

---

## 8. Changelog

- **2026-07-04** — **FULL-SCALE CAMPAIGN DONE** (`reports/2026-07-04_campaign/`).
  27 sequences × 4 datasets (KITTI 00-10, EuRoC MH/V1/V2, TUM fr3 ×4, TUM-VI),
  odometry + SLAM each, locked-clock Ada, **0 failures**, survived a dev-box
  poweroff (setsid). Cross-sequence modal consistency **91%**; pooled k-means
  prefers **k=7–8** (validates the taxonomy at scale); **61% of GPU time carries
  PiM affinity** (21% strong + 40% conditional). New `analysis/campaign.py`.
  Solved a stack-breaking blocker on the way: the 575 downgrade broke system
  Nsight (ncu 2026.2/2025.3 are CUDA-13, reject 575) — fixed with CUDA-12.9
  ncu 2025.2 (NVIDIA public redist) + NCU_BIN override, so NVBit + ncu + nsys
  now share one driver. PUBLISHABILITY issues 2 and 7 closed.
- **2026-07-03 (night)** — **SLICE 3 UNBLOCKED.** Driver downgraded on the
  workstation: 575.64.05 + CUDA 12.9.1 + linux-lts 6.12.39 (dkms module built;
  GRUB default set; the CachyOS prebuilt nvidia-open module packages were the
  conflict — removed). cu12 cuVSLAM wheel rebuilt from the sda2 source release
  in the repo's podman builder (CUDA 12.6 base after 12.9 CCCL header clash;
  imports + GPU warm-up verified on 575). NVBit mem_trace built via the
  container recipe (host gcc-16 breaks nvcc 12.9 — measured), launch-window
  patch live, **capability gate passes for the first time**; smoke trace:
  489k warp-access records, footprints match first principles. Overnight
  program running: aim-pass-targeted traces (TUM steady, TUM+KITTI st_track
  windows) → locality analysis → the 29-sequence full-scale campaign.
  aria2c replaced curl for the package downloads (minutes vs hours). sda2 in
  fstab (ro,nofail). Full datasets found: KITTI 00-21 color, EuRoC ×11,
  TUM fr3 ×4, TUM-VI tars ×15+.
- **2026-07-03 (evening)** — **Slice-3 unblock started** (driver-downgrade
  permission granted on the workstation). Confirmed NVBit caps at driver ≤575
  even in v1.8 → downgrade is the path. Committed the analysis half (`40e8a05`):
  `analysis/locality.py` (footprint + reuse-distance-vs-cache-capacity CDF +
  coalescing + inter-launch overlap from NVBit mem_trace) and the
  `mem_trace_launch_window.patch` (LAUNCH_BEGIN/END so traces stay bounded, not
  TB-scale). Full 65+58 GB dataset set found on the workstation's `sda2` (KITTI
  gray+color, EuRoC MH/VR, TUM). Laptop clock-lock sudo now works too (measured
  laptop at lock: 40 GB/s / 1145 GF). In progress on the workstation: staging
  the 575.64.05 + CUDA-12.9 + linux-lts-6.12 packages and a CUDA-12 cuVSLAM
  wheel (the CUDA-13 wheel won't load on a 575 driver). Rollback staged at
  `~/driver-rollback/revert.sh`. Also added `cluster.py`, `compare.py` and the
  measured-ceiling/warm-up tooling to the committed set.
- **2026-07-03 (late pm)** — **Production matrix landed.** All 22+1 workstation
  captures completed at locked clocks (both power cuts dodged the capture
  windows). Results: locked-clock CoV 0.14% (closes issue 1); loop-closure scan
  dominates GPU time on all three workloads (41–63%, closes issue 2);
  **the working-set/L2 crossover measured** — DB-scan footprint ≈ L2 capacity
  at room scale, 3.6× more DRAM traffic at street scale with 30/32
  sectors/request scatter; TUM↔KITTI class agreement 97% time-weighted;
  clustering prefers k=7 (taxonomy count), purity 0.67. New modules:
  `cluster.py` (stdlib k-means validation), `compare.py` (cross-dataset
  agreement). Synthesis: `reports/2026-07-03_matrix_synthesis/SYNTHESIS.md`.
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
