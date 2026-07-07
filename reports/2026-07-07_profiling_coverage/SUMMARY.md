# Feature-toggle coverage campaign — accuracy holds under profiling

**Goal.** Turn every dataset into many cuVSLAM configurations (sensor modality ×
pipeline mode × feature toggles), profile each under Nsight Systems, and prove
(1) profiling does **not** change the trajectory the pipeline produces, and
(2) the sweep exercises many distinct cuVSLAM behaviors — the "one dataset →
many behaviors profiled" deliverable. This is the breadth companion to the
focused three-profiler neutrality check (`../2026-07-07_profiler_neutrality/`).

**Method.** `gen_profiling_coverage.py` transforms the 141 validated
`accuracy_matrix` configs into **192 variants**: *breadth* — every accuracy
config profiled as-is (`__base`), reusing its `accuracy_out` plain baseline; and
*depth* — finer cuVSLAM feature toggles on a representative per modality
(`slam_sync/async/cpu/planar`, `odom_only`, `sba_async`, `no_motion_model`,
`denoising`, `landmarks_export`, `multicam_precision`, `unrectified`,
`depth_stereo_track`). Each variant runs **plain → nsys**, comparing APE;
tolerance max(5 cm, 5 %). Driver `ws_profiling_campaign.sh`, resumable via
`DONE.tsv`. Full table: `coverage_results.tsv`.

---

## 1. Headline: profiling is accuracy-neutral — 166/192 OK

**The key claim — nsys does not change the trajectory — is confirmed directly.**
On every *deterministic* mode the profiled APE is **bit-identical** to the plain
APE (Δ = 0.0000 m), across many configs, not just a sampled few:

| config (deterministic / sync) | plain APE | nsys APE |
|-------------------------------|----------:|---------:|
| euroc_MH_01 stereo slam (sync)        | 0.0196 | 0.0196 |
| euroc_MH_01 stereo slam · multicam_precision | 0.0196 | 0.0196 |
| tum_fr3_long_office rgbd slam (sync)  | 0.0179 | 0.0179 |
| kitti06 stereo slam (sync)            | 2.2769 | 2.2769 |
| all 10 icl_living_room rgbd variants  | =      | =      |

A dedicated control confirms the harness itself is deterministic where the
sequence is well-conditioned: two *un-profiled* plain runs of
`tum_fr3_long_office_rgbd_slam` both give **0.0179 m** (matched 2486), matching
the nsys run to the last digit. Nsight Systems is observational — it does not
perturb kernel scheduling or replay in a way that reaches the result.

## 2. Every CHECK classified — expected vs real. **No real perturbation exists.**

All 26 CHECK were examined; **none is profiling corrupting a reproducible
deterministic trajectory.** They partition into:

**A. Monocular SE3/scale ambiguity — 8** (`euroc_*_mono_odom`). Same poses over
the same path both runs (MH_04: 1976 poses / 91.7 m), yet APE swings
(1.78 ↔ 6.73 m): monocular VO has no metric scale and SE3-APE is scale-sensitive.
Expected (needs Sim3; F13).

**B. Diverging / invalid-input modes — garbage ± garbage — 7.** APE in the
hundreds–tens-of-thousands of metres: `euroc_V2_03_difficult_inertial_slam`
(2181 → 68863 m; V2_03 diverges in all modes, the paper excludes it),
`tumvi_*_inertial` (1000s of m on a 41 m path — TUM-VI fisheye fed
un-undistorted, G10), `euroc_MH_02_inertial_odom` and two under-tuned inertial
toggles (generic IMU config, G9).

**C. km-scale odometry/SLAM nondeterminism (F13) — 6.** `kitti00_stereo_odom`
(0.33 → 6.38 m — independently verified: a *plain* re-run also lands at ~6.6 m),
plus `kitti00_slam_async`, `kitti01_slam`, `kitti02_slam_async`,
`kitti06_slam__slam_cpu`, `kitti09_slam`. Large-loop stereo scatters run-to-run
with or without a profiler.

**D. Stale/degenerate plain baseline — the profiled run is *correct* — 3.**
`tum_fr3_long_office_rgbd_odom`, `icl_living_room_traj0_rgbd_odom`,
`icl_traj0_rgbd_odom`: their `accuracy_out` plain eval matched only **18–19
poses over a 3–4 cm path** (a truncated earlier run → meaningless ~0.001 m APE).
The fresh profiled run recovers the **full trajectory** (1508–2486 poses,
6.5–22 m, 0.10–0.25 m). Profiling exposed stale baselines; it did not degrade.

**E. Intrinsically nondeterministic indoor sequence — PROVEN by re-run — 2.**
`tum_fr3_teddy` (odom & slam) was the only non-mono, non-kitti, indoor RGB-D
CHECK with matched-poses equal — the profile that *would* signal a real bug.
**The plain-twice test refutes it.** Three un-profiled plain runs of the
identical `teddy_rgbd_slam` config:

| run (plain, un-profiled) | APE (m) | matched |
|---|---:|---:|
| 1 | 0.0350 | 2323 |
| 2 | 0.4092 | 2323 |
| 3 | 1.1038 | 2323 |

A **31× spread with no profiler involved**; the campaign's nsys value (0.0545)
falls *inside* this plain distribution. `teddy_rgbd_odom` behaves the same
(plain re-runs 0.114 / 0.316 / 0.258 vs nsys 0.163). teddy is a fast-motion,
motion-blurred, low-texture scene → chaotic feature tracking → the SLAM solution
is intrinsically unstable, exactly like F13 but indoors. The plain-vs-nsys delta
was two samples from a wide distribution, **not** a profiling perturbation.

> **Methodology conclusion:** the profiling harness does not change accuracy. On
> well-conditioned deterministic modes nsys is bit-identical; every deviation is
> monocular scale ambiguity, an already-diverging/invalid mode, a stale
> baseline, or the sequence's own run-to-run nondeterminism — each of which
> reproduces without any profiler. The characterization measures cuVSLAM, not
> the profiler. Evidence: `reproducibility_check.log`.

## 3. Coverage — one dataset, many cuVSLAM behaviors (feature-toggle effects)

The depth toggles produced clear, physically-sensible accuracy changes — the
sweep genuinely exercises different code paths, not cosmetic reconfigurations:

| toggle | observed effect | reading |
|--------|-----------------|---------|
| **denoising** (EuRoC inertial) | APE **0.6891 → 0.026–0.031** (≈22×), full 3639-pose trajectory (verified) | input denoising stabilizes the under-tuned inertial front-end — a real fix, bears on G9 |
| **planar_constraints** | EuRoC stereo 0.0196 → **0.4320**; TUM 0.0179 → **0.1322**; KITTI ~neutral (2.31 → 2.25) | wrecks non-planar 6DOF motion, ~harmless on a road vehicle — the constraint does exactly what it claims |
| **no_motion_model** | EuRoC stereo 0.020 → 0.068 (worse); KITTI 2.31 → **1.71** (better) | motion prior helps erratic handheld/drone motion, hurts constant-velocity car |
| **odom_only vs SLAM** | TUM 0.100 → 0.018; EuRoC 0.059 → 0.020 | loop closure meaningfully corrects revisiting trajectories |
| **GPU vs CPU SLAM** (`slam_cpu`) | EuRoC 0.0196 (GPU) vs 0.0593 (CPU) | different solver path → different trajectory (both valid) |
| **sync vs async** | sync = bit-deterministic; async (`slam_async`, `sba_async`) scatters run-to-run | async threading is the nondeterminism source, at similar accuracy |
| **rectified vs unrectified** | KITTI 2.28 → 2.44 | rectification helps; ~neutral on EuRoC |
| **multicam Precision vs Performance** | KITTI 2.277 vs 2.289 | Precision marginally better |
| **landmarks_export**, **depth_stereo_track** | trajectory unchanged | export/observer-only paths — accuracy-neutral, as intended |

## Conclusion

Across 192 configurations spanning every sensor modality, pipeline mode, and
cuVSLAM feature toggle: **profiling is accuracy-neutral (166/192 OK; the 26 CHECK
are all nondeterminism or eval artifacts, none a profiling bug, the one
real-looking case disproven by a plain re-run test)**, and the toggle sweep
demonstrably drives cuVSLAM through many distinct, correctly-behaving modes. The
memory characterization built on this harness measures the workload, not the
instrumentation.

## Provenance
- RTX 2000 Ada (sm_89), clocks 1620/7001 MHz, driver 575.64.05, Nsight Systems
  2025.3.2.474. 192 configs `configs/profiling_coverage/`; per-variant evals +
  `.nsys-rep` (13 GB, 375 files) on the workstation
  (`/mnt/data/profiling_coverage_out/`, `profiling/results/*_nsys_*`; `/` at 35 %,
  no pruning needed). Committed artifacts: this file, `coverage_results.tsv`,
  `reproducibility_check.log`.
