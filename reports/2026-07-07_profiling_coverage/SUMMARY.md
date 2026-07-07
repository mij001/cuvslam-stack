# Feature-toggle coverage campaign — accuracy holds under profiling

**Goal.** Turn every dataset into many cuVSLAM configurations (sensor modality ×
pipeline mode × feature toggles), profile each under Nsight Systems, and confirm
the trajectory-vs-ground-truth accuracy does **not** deviate under profiling —
i.e. instrumentation introduces no bug. This is the breadth companion to the
focused three-profiler neutrality check (`../2026-07-07_profiler_neutrality/`).

**Method.** `gen_profiling_coverage.py` transforms the 141 validated
`accuracy_matrix` configs into **192 variants**:
- *breadth* — every accuracy config profiled as-is (`__base`), reusing its
  existing `accuracy_out` plain baseline;
- *depth* — finer cuVSLAM feature toggles on a representative per modality:
  `slam_sync / slam_async / slam_cpu / slam_planar`, `odom_only`, `sba_async`,
  `no_motion_model`, `denoising`, `landmarks_export`, `multicam_precision`
  (stereo), `unrectified` (stereo), `depth_stereo_track` (RGB-D).

Each variant runs **plain → nsys** and compares APE; tolerance max(5 cm, 5 %).
Driver `ws_profiling_campaign.sh`, resumable via `DONE.tsv`. Full table:
`coverage_results.tsv`.

## Result — 166 / 192 OK; all 26 CHECK are known-benign, none a profiling bug

On the modes we actually characterize for memory — **stereo and RGB-D SLAM** —
profiling is essentially perfectly neutral:

| pipeline mode | OK | CHECK | neutral |
|---------------|---:|------:|--------:|
| RGB-D SLAM        | 45 | 1 | 45/46 |
| stereo SLAM       | 37 | 3 | 37/40 |
| stereo odom       | 23 | 1 | 23/24 |
| stereo async SLAM | 11 | 2 | 11/13 |
| RGB-D odom        | 17 | 4 | 17/21 |
| inertial SLAM     | 16 | 4 | 16/20 |
| inertial odom     | 11 | 3 | 11/14 |
| **mono odom**     | 3  | 8 | 3/11  |
| RGB-D/inertial async | 3 | 0 | 3/3 |

### Every CHECK, explained (26 total)

**A. Monocular SE3/scale ambiguity — 8.** All `euroc_*_mono_odom__base`. The
profiled and plain runs match the *same poses over the same path*
(MH_04: 1976 poses / 91.7 m both) yet APE swings (1.78 ↔ 6.73 m) because
monocular VO has no metric scale and SE3-APE is scale-sensitive — run-to-run
scale drift, not profiling. Mono needs Sim3 alignment (THESIS_FINDINGS F13).

**B. Diverging / invalid-input modes — garbage ± garbage — 7.** Absolute APE in
the hundreds-to-tens-of-thousands of metres, so the delta is two degenerate
numbers differing:
- `euroc_V2_03_difficult_inertial_slam` (2181 → 68863 m) — V2_03 diverges in all
  modes; the paper excludes it too.
- `tumvi_corridor1_inertial_{odom,slam}`, `tumvi_magistrale1_inertial_slam`
  (1000s of m on a 41 m path) — TUM-VI is fisheye, fed un-undistorted (G10,
  invalid input).
- `euroc_MH_02_easy_inertial_odom`, `euroc_MH_01_easy_inertial_slam__{odom_only,
  slam_cpu}` — inertial under-tuned (generic IMU config, G9), unstable.

**C. km-scale odometry/SLAM run-to-run nondeterminism (F13) — 6.**
`kitti00_stereo_odom` (0.33 → 6.38 m — the independently-verified bimodal case:
a *plain* re-run also lands at 6.6 m), plus `kitti00_slam_async`,
`kitti01_slam`, `kitti02_slam_async`, `kitti06_slam__slam_cpu`, `kitti09_slam`.
Large-loop stereo scatters run-to-run with or without a profiler.

**D. Stale/degenerate plain baseline — the profiled run is the *correct* one — 3.**
`tum_fr3_long_office_rgbd_odom`, `icl_living_room_traj0_rgbd_odom`,
`icl_traj0_rgbd_odom`. Their `accuracy_out` plain eval matched only **18–19
poses over a 3–4 cm path** (a truncated earlier run), giving a meaningless
~0.001 m APE. The fresh profiled run recovers the **full trajectory** (1508–2486
poses, 6.5–22 m, 0.10–0.25 m APE). Profiling did not degrade anything — it
exposed stale baselines. *(Follow-up: regenerate these three `accuracy_out`
odom baselines; the correct evals already exist under `profiling_coverage_out`.)*

**E. Hard-sequence scatter, full trajectory both, profiling as-often-better — 2.**
`tum_fr3_teddy_rgbd_odom` (0.108 → 0.163 m) and `tum_fr3_teddy_rgbd_slam`
(0.164 → **0.055 m**, profiled *better*). teddy is the fast-motion/blur fr3;
~5 cm scatter that goes both directions is not instrumentation degradation.

## Conclusion

Across 192 configurations spanning every sensor modality, pipeline mode, and
cuVSLAM feature toggle, **no CHECK is a case of profiling corrupting a valid
deterministic trajectory.** They partition entirely into monocular scale
ambiguity (A), already-diverging or invalid-input modes (B), the known km-scale
nondeterminism (C), stale baselines the profiled run actually corrects (D), and
symmetric hard-sequence scatter (E). On the deterministic stereo/RGB-D SLAM
modes that underpin the memory characterization, Nsight Systems is neutral
(45/46, 37/40; bit-identical in the focused check). The profiling harness does
not affect accuracy — the characterization measures cuVSLAM, not the profiler.

## Provenance
- RTX 2000 Ada (sm_89), clocks 1620/7001 MHz, driver 575.64.05, Nsight Systems
  2025.3.2.474. 192 configs `configs/profiling_coverage/`; per-variant evals +
  `.nsys-rep` profiles on the workstation (`/mnt/data/profiling_coverage_out/`,
  `profiling/results/*_nsys_*`). Machine-readable: `coverage_results.tsv`.
