# cuVSLAM Memory Characterization — Thesis Findings & Roadmap

*Consolidated 2026-07-05. Every number in this document is measured and
committed; each finding cites the report that reproduces it. This is the
master document for thesis writing: what we know, what it licenses us to
claim, what is still missing, and the ranked paths from here to the strongest
possible thesis.*

---

## 0. The thesis in one paragraph

Physical AI workloads are bounded by **data movement, not compute**. We prove
this for cuVSLAM — NVIDIA's production visual-SLAM stack, the localization
core of real robots — via the first per-kernel, per-**data-structure** memory
characterization of a production SLAM system on GPU (DAMOV methodology adapted
to GPUs + NCU roofline). The characterization yields a **three-way persistence
taxonomy** that maps each data structure to the memory-centric substrate that
serves it best:

| class | data | lifetime / touch pattern | substrate ask |
|---|---|---|---|
| **streaming** | camera frames, pyramids, gradients | per-frame, consumed once, compulsory misses | **near-sensor SRAM** (consume before DRAM) |
| **hot-persistent** | BA linear system, local-map state | every frame, pre-sized, streamed at 96–100 % purity | **DRAM-PiM** (bank-level streaming compute) |
| **cold-persistent** | keyframe / landmark database | touched on keyframe + loop closure; grows with session | **ISP** (in-storage scan at the host store) |

The thesis is no longer inferred from counters: each row is grounded in
measured address-to-allocation attribution across 27 sequences × 4 datasets.

---

## 1. Evidence inventory (findings F1–F12)

### F1 — Publishable measurement rigor
Locked-clock RTX 2000 Ada (persistence + `-lgc 1620,1620 -lmc 7001,7001`):
5-repeat CoV median **0.14 %** (vs 49.6 % unlocked laptop, 9.3 % warmed).
Ceilings measured at lock: **205.0 GB/s DRAM ± 0.1, 5445 GFLOP/s FP32 ± 3**.
Classification carries a ±25 % sensitivity analysis; cold/warm cache bracket;
×5/×3 repeat variance. → `reports/2026-07-03_*_rtx2000ada`,
`profiling/METHODOLOGY.md`.

### F2 — Generalization at full scale
**27 sequences × 4 datasets** (KITTI 00–10, EuRoC MH/V1/V2 ×11, TUM fr3 ×4
incl. texture/structure ablations, TUM-VI), odometry + SLAM, locked clocks,
**0 failures**. Cross-sequence modal class consistency **91 %** (24/49 kernels
unanimous, 42/49 ≥ 80 %); the residual flips are the physically meaningful L2
crossover, not noise. → `reports/2026-07-04_campaign/`.

### F3 — The taxonomy is discovered, not asserted
Pooled k-means over the 27-sequence feature cloud prefers **k = 7–8**,
matching the G1–G7 GPU-DAMOV classes (silhouette-best, purity 0.68, ARI 0.30
vs the decision-tree labels, monotone in k). The decision tree is the
*labeling*; clustering is the *independent validation*. **61 % of GPU time
carries PiM affinity** (21 % strong + 40 % conditional).
→ `reports/2026-07-04_campaign/` (`cluster_sweep.csv`, PiM rollup).

### F4 — Data movement is already the bottleneck at the system edge
Explicit host↔device copies cost **41 % of kernel time** on the TUM workload;
the H2D sensor upload is **1.68 MB/frame** — the direct, measured near-sensor
opportunity. → `analysis/transfers.py`, report §5.

### F5 — Front-end is cache-immune streaming (strengthened by correction)
Front-end reuse-distance CDFs are **flat from 64 KiB to 48 MiB**: no cache
size helps; the misses are compulsory. After the memory-space correction the
claim got *cleaner*: conv_grad's apparent scatter (15.4 sectors/warp) and
92 % "reuse" were **shared-memory tiles**; global-only, every front-end kernel
streams at ~4 sectors/warp, 100 % coalesced, ~zero DRAM reuse. A cache cannot
fix this; near-sensor consumption eliminates it. (The shared/global split
that this correction rests on is compiler codegen, same caveat as F11 —
the *reuse-CDF technique* is architecture-independent, the specific split
measured here is a property of the Ada-compiled binary.)
→ `reports/2026-07-04_slice3_locality/` §2 + §5, `data_v2/tum_odom_global/`.

### F6 — The loop-closure scan is a scattered gather (two methods agree)
The full self-correction arc, on record: ncu counters said scattered gather
(18–30 sectors/request); the unfiltered trace said coalesced (2.1
sectors/warp) and the 2026-07-04 report claimed "proxy overturned"; the
attribution join then revealed 94 % of the kernel's accesses are register
spill; the space-filtered re-derivation shows the **global (data) accesses are
23.4–30.0 sectors/warp, only 2.3–6.0 % coalesced — a scattered gather,
matching the counters**. The coalesced signal was the spill stream (local
space: 2.0 sectors/warp, 100 % coalesced, by construction). **G2-scatter
stands; counters and traces now agree.**
→ `reports/2026-07-04_slice3_locality/FINDINGS.md` §5, `data_v2/*_sttrack_*`.

### F7 — The scan's working set grows and migrates with the session
st_track per-scan footprint **0.464 MB (room) → 1.093 MB (street)**;
inter-launch Jaccard **0.669 → 0.899** (10–33 % of the set turns over per
scan). The union over a deployment — the whole keyframe database,
incrementally scanned — is what no cache holds. Unaffected by the space
correction (the spill window is ~3 KB). → same report, both data/ and data_v2/.

### F8 — The GPU memory budget is static; the database grows host-side
TaggedAllocator journal over the full ~2500-frame loop-closing TUM run:
**240 device allocations, 108.65 MB, identical to a 30-frame run; peak = total
for every tag; zero mid-run frees.** cuVSLAM pre-sizes everything at init.

| tag | allocs | peak live MB |
|---|---|---|
| images_raw | 92 | 39.97 |
| pyramid_levels | 99 | 31.15 |
| ba_linear_system | 31 | 24.26 |
| keyframe_descriptors | 2 | **6.73 (fixed)** |
| feature_tracks | 8 | 4.27 |
| depth_pyramid | 3 | 2.21 |
| icp_state | 4 | 0.07 |
| + pinned-host (34 allocs) | | 17.14 (15.41 = BA mirror) |

The GPU-resident keyframe state is a **fixed 6.7 MB buffer**; the session-
scale landmark/keyframe database (LMDB-backed) grows **only on the host**.
The ISP leg of the taxonomy therefore targets host storage — shown from the
allocator, not inferred. → `reports/2026-07-04_attribution/` (F8–F10).

### F9 — Per-kernel data-structure attribution (the M7 milestone)
Three-layer pipeline (source journal with backtraces → driver sidecar with
launch-id lifetimes → streaming join): **274/274 allocations resolved, 0
unknown**. Steady-state TUM window (28.1 GiB of addresses):
`st_track_with_cache` = **94.2 % register spill + 4.9 % keyframe descriptors**
(unmapped 0.8 %); `st_build_cache` = 93 % keyframe descriptors;
`sba::reduced_system_stage_2` = **96.9 % ba_linear_system**; front-end compute
kernels 89–98 % shared-tile with their global residue exactly on
pyramid/track structures. Measured NVTX kernel→stage table (cuVSLAM's own
profiler domains, doubly-ungated): `st_track_with_cache` sits under
**SLAM:LC & optimization** — stage attribution is measured, not name-inferred.
→ `reports/2026-07-04_attribution/`.

### F10 — Attribution generalizes: composition is a kernel property
Full-matrix attribution campaign, 27 sequences, coverage audited and
gap-filled to **0 missing kernels**: **48/49 kernels have a unanimous top
data-structure tag across every sequence they appear in.** The keyframe pair
is uniform from room to street scale (st_build → descriptors 93–100 %;
st_track → 91–95 % spill + 5–8 % descriptor scatter). kitti01/04 confirmed
loop-free (the no-loop bracket). Mode differences surface in the allocator
(stereo KITTI: 215 allocs, no depth/ICP tags) — the taxonomy adapts per mode
exactly as designed. → `reports/2026-07-05_attribution_campaign/`.

### F11 — Register spill is a first-class DRAM consumer (new axis)
The loop-closure scan's DRAM *volume* is ~92 % compiler spill traffic
(9-dim patch working set exceeds the register budget), perfectly coalesced
and L2-hostile in aggregate. This is a hardware ask no prior SLAM
characterization surfaces: **spill-local SRAM / larger register file attacks
the volume; near-memory gather attacks the data-side latency.** It is also a
methodological warning we document: unfiltered GPU address traces are
dominated by spill/tile accesses (88–98 % of raw records), which silently
poison locality *and* attribution analyses.

**Scope of this claim, stated precisely (do not overclaim it):** register
allocation and spill decisions are made by the compiler for a *specific
target compute capability* (register file size, occupancy heuristics) —
they are not a property of the algorithm alone. The 94 % figure is measured
on the Ada (SM 8.9) compiled binary; a different target (e.g. Orin, SM 8.7)
could compile the same source with a different spill volume, possibly
zero. What we do expect to transfer across targets is the *taxonomy*
(which buffers exist, their sizes, their persistence class — set by
cuVSLAM's own source-level allocation, not by codegen) and the *technique*
(space-filtered reuse-distance analysis, applicable to any trace). The
spill *quantity* itself is flagged for re-validation on Orin, not assumed.

### F12 — Bounded residuals (honesty inventory)
`sba::build_full_system_2` carries ~40 % unmapped global traffic *identically
in all 27 sequences*; lk_track / cub-partition / getrf have systematically
unmapped or driver-internal globals. Consistent with **static module memory**
(`__device__` globals, invisible to both journal layers by construction) plus
**texture-path reads** (TEX fetches never appear in the trace, so image-tag
traffic is a lower bound). Stable, bounded, explained — and namable via the
optional Layer-3 refinement (§4, path C3).

### F13 — Accuracy validation: we profiled a correctly-functioning system
The characterization is only meaningful if the cuVSLAM runs it measured were
producing *correct* trajectories. Config matrix — **now 141 runs** (expanded
2026-07-07 from 104 with the full paper set: all 8 ICL-NUIM Mono-Depth
trajectories + the 10 TUM fr3 sequences) across on-disk datasets × camera
variants (stereo / stereo-inertial / mono / RGB-D) × pipeline modes (odometry /
SLAM / sync / async / GPU / CPU) — scored against ground truth and compared to
the paper (arXiv:2506.04359v3, Tables 2/3/6). Every config is validated under
BOTH the runner and the profiling harness (`validate_accuracy_configs.sh`:
141/141 `--check` + profiling-flow). Results (`reports/2026-07-07_accuracy_full/`,
plus the earlier `reports/2026-07-06_accuracy/`):
- **The profiled modes reproduce the paper.** EuRoC stereo APE
  **0.114 / 0.051 m** (odom/slam) vs paper 0.13 / 0.054 (millimetre match);
  TUM fr3 **0.060 / 0.047 m** beats the paper's 0.11 / 0.065; ICL-NUIM
  Mono-Depth **0.099 / 0.136 m** vs paper 0.026 (same order; per-trajectory
  spread, generic config); KITTI 500 m drift **0.82 %** ≈ leaderboard 0.85 %.
  Stereo and RGB-D SLAM are exactly the modes the memory characterization used.
- **Instrumentation is accuracy-neutral (QoR), outlier verified.** The
  TaggedAllocator instrumented wheel equals the baseline wheel (EuRoC
  bit-identical Δ=0.0000; TUM/kitti06 within nondeterminism ≤0.035 m). The one
  large Δ — kitti00 odom (0.335 vs 6.605 m) — was **proven to be run-to-run
  nondeterminism, not instrumentation**: re-running kitti00 odom on the
  *baseline* wheel also yields 6.605 m (bit-identical to the tagged run), i.e.
  the baseline itself is bimodal on this 3.7 km odometry-only path (GPU float
  nondeterminism, no loop closure to correct drift). SLAM mode does not show it.
  So the build we profiled behaves as the shipping binary — the linchpin
  closing the characterization's validity threat — and cuVSLAM's long-odometry
  nondeterminism is itself a documented observation.
- **Feature toggles behave as designed:** SLAM −0.32 m vs odometry (loop
  closure reduces drift); async +0.14 m vs sync (latency trade); CPU ≈ GPU
  SLAM. Independent evidence the pipeline is wired correctly.
- **One isolated defect, explained:** inertial (IMU) mode is under-tuned
  (generic config, not the paper's per-dataset IMU calibration) — only 3/11
  EuRoC inertial sequences converge; it does **not** touch the stereo/RGB-D
  characterization and has a scoped config-only fix. TUM-VI (fisheye not
  undistorted) and mono (needs Sim3 alignment) exclusions are data-prep /
  metric-definition issues, not cuVSLAM failures. → `reports/2026-07-06_accuracy/`.

---

## 2. Tooling contributions (defensible as artifacts)

1. **TaggedAllocator instrumentation** (`patches/0002-tagged-allocator-nvtx.patch`)
   — env-gated allocation journal with host backtraces in cuVSLAM's RAII
   wrappers + NVTX activation; bit-identical no-op when disabled.
2. **NVBit alloc-event sidecar** (`blocked/mem_trace_alloc_events.patch`) —
   driver-level allocation lifetimes in the trace's own launch-id clock; plus
   the launch-window/kernel-filter patch that makes GB-scale traces feasible.
3. **Space-aware attribution join** (`analysis/attribution.py`) — streaming,
   O(1)-memory, access-capped; the memory-space bucketing rule (F11).
4. **Coverage audit + gap-fill planner** (`campaign/plan_gapfill.py`) —
   guarantees windowed campaigns miss nothing; dense-cluster anchoring defeats
   run-to-run launch-id drift.
5. **Cross-sequence synthesis** (`analysis/attribution_campaign.py`) and the
   full campaign automation, resumable and power-cut-safe.
6. 18 GPU-free tests; every committed table reproduces from committed CSVs
   with stdlib Python only. This is an artifact-evaluation-ready posture.

---

## 3. What the evidence licenses the thesis to claim

1. **"cuVSLAM's memory system, not its ALUs, is the design problem."**
   (F1–F4: 61 % PiM-affine GPU time, 41 % copy overhead, flat reuse CDFs.)
2. **"Each persistence class needs a different substrate"** — with each class
   now *named by data structure and measured by bytes* (F5, F8, F9, F10),
   not asserted from kernel names.
3. **"The claims are workload-stable"**: 91 % kernel-class consistency and
   48/49 unanimous attribution across 27 sequences (F2, F10).
4. **"The method self-corrects"**: two documented correction arcs (counter →
   trace → space-filtered trace), ending with independent methods in
   agreement (F6). For a characterization venue this is a *strength*: it shows
   the pipeline catches its own errors.
5. **"Spill is a hidden memory citizen"** (F11) — a genuinely new observation
   for SLAM workloads with its own hardware implication.
6. **"The measurements are of a *correct* system"** (F13): the profiled
   stereo/RGB-D SLAM modes reproduce NVIDIA's published accuracy, and the
   instrumented build is trajectory-identical to the baseline. This is the
   claim that makes all the others admissible — you profiled cuVSLAM, not a
   broken configuration of it.

---

## 4. Gaps — what a reviewer will still poke (ranked by risk)

| # | gap | severity | closes via |
|---|---|---|---|
| G1 | **No substrate-side evaluation** — "G-classes *might* benefit; show a delta." | blocks the architecture paper only | Path B (Accel-Sim NDP + AccelWattch) |
| G2 | **No energy numbers** — PiM's headline win is joules | medium (both papers) | NVML whole-run now; AccelWattch per-kernel in Path B |
| G3 | **Host-side ISP leg is under-measured**: we prove the DB grows host-side (F8) but haven't characterized the LMDB/storage traffic itself | medium — it is *the* cold-persistent claim | Path C1 (cheap, high value) |
| G4 | **Single GPU (Ada desktop); codegen-dependent findings unvalidated elsewhere** — Physical AI deploys on edge parts, and register-spill/shared-memory splits (F11) are compiler codegen artifacts of the sm_89 target specifically, not guaranteed to hold for sm_87 (Orin) or any other compute capability — the taxonomy (which structures exist, their sizes/classes) is expected to transfer since it is source-level, not codegen-level, but this is untested | medium | Path C2 (Jetson Orin, descriptor exists) — re-derive the spill/shared/global split specifically, not just re-run for edge numbers |
| G5 | Static-memory residuals unnamed (F12) | low — bounded + explained | Path C3 (Layer-3 kernel args / module-global map) |
| G6 | TEX-path invisibility → image traffic lower bound | low — disclosed | note in limitations; ncu texture counters corroborate |
| ~~G7~~ | ~~Repo LICENSE absent~~ **CLOSED 2026-07-07**: MIT (`LICENSE`), scoped to the project's own code; third-party (cuVSLAM, vendored tools, datasets) retain their licenses | administrative | done |
| G8 | One workload family (cuVSLAM) — "is it representative?" | low-medium | argue production-grade + Isaac deployment; optionally sketch ORB-SLAM3 contrast in related work (do NOT expand scope) |
| G9 | **Inertial (IMU) mode under-tuned** (F13): only 3/11 EuRoC inertial sequences converge; generic IMU config, not the paper's per-dataset noise/extrinsics calibration | low — does NOT affect the profiled stereo/RGB-D modes; disclosed | scoped config fix: populate EuRoC IMU intrinsics/extrinsics from each `sensor.yaml`, re-run the 22 inertial configs |
| G10 | **TUM-VI / mono excluded from accuracy validation** (F13): TUM-VI fisheye not undistorted (<180° support); mono needs Sim3 alignment | low — data-prep / metric-definition, not tracking failures; disclosed | undistort TUM-VI to pinhole + re-run; add Sim3 alignment path to the evaluator for mono |

Note: the previous "single-workload accuracy unvalidated" risk is now **closed**
by F13 for the modes we actually characterize (stereo, RGB-D). G9/G10 are
residual accuracy items on modes the memory characterization does not use.

---

## 5. Paths from here — step by step

### Path A — the characterization paper (ISPASS/IISWC class). *Write it now.*
The evidence is complete; nothing below adds a required measurement.

1. **Freeze the claims table**: one row per claim → finding → figure → CSV.
   (Start from §3; the discipline prevents scope creep during writing.)
2. **Figures** (all generatable from committed CSVs, headless):
   F2 consistency heatmap; F3 silhouette-vs-k + PiM rollup; F5 reuse-CDF
   fan (front-end flat vs BA vs st_track); F6 the correction arc as a
   two-panel before/after; F8 memory-budget treemap; F10 attribution
   consistency table (the 48/49 headline); F4 transfer stack per frame.
3. **Write the methodology as the story**: DAMOV→GPU adaptation, the
   space-filter lesson (F11) as a warning to practitioners, the self-
   correction arc as evidence of soundness.
4. **Quick adds while writing** (1–2 days each, no new instrumentation):
   NVML whole-run energy per sequence (shrinks G2); host `iostat`/LMDB file
   growth during the TUM run (starts G3).
5. Target: next ISPASS or IISWC cycle (verify the current CFP dates); both
   value artifact evaluation — the repo is already in that posture (fix G7).

### Path B — the architecture paper (MICRO/ASPLOS/ISCA/HPCA class). *Phase 4.*
1. **Accel-Sim baseline**: replay the committed window traces (steady-state +
   st_ scans; they are exactly the right granularity) through an Ada-like
   SM89 config; validate simulated cache behavior against the measured
   reuse CDFs (F5) — that agreement is the calibration argument.
2. **NDP configs**: (a) near-bank streaming for `ba_linear_system` (reduced
   L2, high bank-level BW — the hot-persistent substrate); (b) near-sensor
   SRAM front-end model (bytes from F4/F8 size it: 1.68 MB/frame upload,
   ~40 MB image+pyramid budget); (c) spill-SRAM / enlarged RF sensitivity for
   st_track (F11).
3. **AccelWattch energy deltas** per taxonomy class; report *deltas*, not
   absolutes.
4. **ISP leg**: model the host LMDB scan offload analytically from Path C1
   measurements (request sizes, scan lengths, growth rate) — an in-storage
   scan against a log-structured store; simulate only if C1 shows the traffic
   is big enough to matter (if it isn't, say so — that is also a result).
5. The paper: taxonomy (from Path A) + three substrate deltas + energy.

### Path C — strengtheners, ranked by value/effort
1. **Host-side cold-persistent characterization** (~2–4 days, high value):
   run TUM/KITTI loop-closure sequences with `strace -e trace=file`/`iostat`/
   LMDB stats on the workstation; measure DB size vs frames, bytes read per
   loop-closure event, host-CPU time in the descriptor match. Directly arms
   the ISP claim (G3) and gives Path B.4 its parameters.
2. **Jetson Orin re-run** (~3–5 days if hardware available): the campaign +
   attribution stack is portable (`hw/jetson_orin_sm87.toml` exists). Edge
   numbers make the Physical-AI framing concrete (G4) — and unified memory
   on Orin changes the transfer story (F4) in a way reviewers will find
   interesting either direction.
3. **Layer-3 kernel-arg correlation** (~2–3 days): hook `cuLaunchKernel`
   params in the sidecar, map the static-memory residuals (F12) to named
   symbols via `cuModuleGetGlobal`. Turns the last unmapped percent into
   names; polish, not substance.
4. **Sensitivity sweeps** (resolution, feature density, loop rate — the
   onboarding §12.2 list): only if a specific reviewer thread demands it;
   the 27-sequence matrix already covers the natural variation axes.

### Recommended sequence (best outcome, least risk)
1. **Now → +2 weeks**: Path A steps 1–3 (paper draft) with C1 running in the
   background on the workstation; A4 quick adds fold in as they land.
2. **+2 → +4 weeks**: internal review pass; fix G7 (LICENSE — needed for
   artifact evaluation); submit A.
3. **While A is in review**: B1–B2 (Accel-Sim calibration is the long pole;
   start it early), C2 if Orin hardware is in hand, C3 opportunistically.
4. **+2 → +4 months**: B3–B5 → the architecture paper.
5. **Thesis assembly** (the two papers *are* the spine):
   - Ch. 2 Background: SLAM pipeline, PiM/ISP/near-sensor substrates, DAMOV.
   - Ch. 3 Methodology: the measurement system (§2 here), incl. the space
     filter and self-correction arcs as first-class content.
   - Ch. 4 Characterization: F1–F7 (= paper A core).
   - Ch. 5 Data-structure attribution: F8–F12 (paper A's differentiator).
   - Ch. 6 Substrate evaluation: Path B results.
   - Ch. 7 The taxonomy as a design method for Physical AI memory systems —
     the generalization argument beyond cuVSLAM.

### What NOT to do
- Don't add a second SLAM system (G8): scope creep with weak marginal
  evidence; handle in related work.
- Don't chase the last 0.3–7 % unmapped before paper A: it is explained and
  bounded (F12); C3 is post-submission polish.
- Don't run more full campaigns: 27 sequences with unanimous attribution is
  past the point of diminishing returns. New measurements should only enter
  via C1/C2 (new *axes*, not more of the same axis).

---

## 6. One-line status for the thesis committee

*Characterization complete and workload-stable at data-structure granularity
(27 sequences, 48/49 kernels unanimous); the three-way persistence taxonomy is
measured, clustering-validated, and self-corrected into agreement across
independent methods; remaining work is substrate-side evaluation (simulation +
energy), for which every input artifact already exists.*
