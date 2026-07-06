# Accuracy matrix vs the cuVSLAM paper (arXiv:2506.04359)

104 runs evaluated against ground truth (full sequences, baseline wheel, RTX 2000 Ada); 62 converged (avgRTE < 5.0% segment-relative), 42 excluded with stated reasons (see 'Convergence & exclusions').

## Mode averages vs paper Table 2 (converged runs only)

APE is the definition-stable comparison; avgRTE/avgRE use our segment definitions (KITTI 100–800 m; EuRoC 8/16/32 m; TUM 1/2/4 m) vs the paper's Table-2 values — directional only. Averages exclude diverged/invalid runs; `n` is the converged count and the excluded ones are itemized below.

| dataset/variant/mode | n | APE m (ours) | APE m (paper) | Δ | avgRTE % (ours/paper) | avgRE ° (ours/paper) |
|---|---|---|---|---|---|---|
| kitti stereo odom | 11 | 4.270 | 3.0 | +1.270 | 1.334 / 0.33 | 0.72 / 1.14 |
| kitti stereo slam | 11 | 3.627 | 1.98 | +1.647 | 1.292 / 0.27 | 0.59 / 0.93 |
| euroc stereo odom | 10 | 0.114 | 0.13 | -0.016 | 1.082 / 0.29 | 1.54 / 1.96 |
| euroc stereo slam | 10 | 0.051 | 0.054 | -0.003 | 0.922 / 0.17 | 1.09 / 1.12 |
| euroc inertial odom | 3 | 0.401 | 0.19 | +0.211 | 2.753 / 0.39 | 6.79 / 2.69 |
| euroc inertial slam | 3 | 0.328 | 0.13 | +0.198 | 2.506 / 0.29 | 6.21 / 2.27 |
| tum rgbd odom | 1 | 0.109 | 0.11 | -0.001 | 1.798 / 1.35 | 1.19 / 5.52 |
| tum rgbd slam | 1 | 0.018 | 0.065 | -0.047 | 1.447 / 0.99 | 1.01 / 4.13 |

## Convergence & exclusions

42 of 104 runs excluded from the paper comparison:

| run | avgRTE % | APE m | reason |
|---|---|---|---|
| euroc_MH_01_easy_mono_odom | 110.1 | 0.4 | scale-ambiguous: monocular needs Sim3 (scale) alignment; SE3-based avgRTE/APE are not meaningful for mono |
| euroc_MH_02_easy_inertial_odom | 5.8 | 0.5 | DIVERGED (avgRTE 5.8% ≥ 5.0%) |
| euroc_MH_02_easy_inertial_slam | 12.4 | 2.5 | DIVERGED (avgRTE 12.4% ≥ 5.0%) |
| euroc_MH_02_easy_mono_odom | 344.9 | 1.3 | scale-ambiguous: monocular needs Sim3 (scale) alignment; SE3-based avgRTE/APE are not meaningful for mono |
| euroc_MH_03_medium_inertial_odom | 12.3 | 3.4 | DIVERGED (avgRTE 12.3% ≥ 5.0%) |
| euroc_MH_03_medium_inertial_slam | 8.4 | 1.9 | DIVERGED (avgRTE 8.4% ≥ 5.0%) |
| euroc_MH_03_medium_mono_odom | 21.3 | 2.6 | scale-ambiguous: monocular needs Sim3 (scale) alignment; SE3-based avgRTE/APE are not meaningful for mono |
| euroc_MH_04_difficult_inertial_odom | 5.6 | 1.4 | DIVERGED (avgRTE 5.6% ≥ 5.0%) |
| euroc_MH_04_difficult_inertial_slam | 5.3 | 0.9 | DIVERGED (avgRTE 5.3% ≥ 5.0%) |
| euroc_MH_04_difficult_mono_odom | 11.6 | 1.8 | scale-ambiguous: monocular needs Sim3 (scale) alignment; SE3-based avgRTE/APE are not meaningful for mono |
| euroc_MH_05_difficult_inertial_odom | 9.7 | 2.8 | DIVERGED (avgRTE 9.7% ≥ 5.0%) |
| euroc_MH_05_difficult_inertial_slam | 9.7 | 2.8 | DIVERGED (avgRTE 9.7% ≥ 5.0%) |
| euroc_MH_05_difficult_mono_odom | 285.0 | 3.4 | scale-ambiguous: monocular needs Sim3 (scale) alignment; SE3-based avgRTE/APE are not meaningful for mono |
| euroc_V1_01_easy_mono_odom | 466.3 | 1.5 | scale-ambiguous: monocular needs Sim3 (scale) alignment; SE3-based avgRTE/APE are not meaningful for mono |
| euroc_V1_02_medium_inertial_odom | 12.4 | 1.6 | DIVERGED (avgRTE 12.4% ≥ 5.0%) |
| euroc_V1_02_medium_inertial_slam | 6.2 | 0.6 | DIVERGED (avgRTE 6.2% ≥ 5.0%) |
| euroc_V1_02_medium_mono_odom | 396.8 | 1.6 | scale-ambiguous: monocular needs Sim3 (scale) alignment; SE3-based avgRTE/APE are not meaningful for mono |
| euroc_V1_03_difficult_inertial_odom | 16.2 | 2.3 | DIVERGED (avgRTE 16.2% ≥ 5.0%) |
| euroc_V1_03_difficult_inertial_slam | 13.8 | 2.0 | DIVERGED (avgRTE 13.8% ≥ 5.0%) |
| euroc_V1_03_difficult_mono_odom | 47.5 | 1.5 | scale-ambiguous: monocular needs Sim3 (scale) alignment; SE3-based avgRTE/APE are not meaningful for mono |
| euroc_V2_01_easy_mono_odom | 832.9 | 1.8 | scale-ambiguous: monocular needs Sim3 (scale) alignment; SE3-based avgRTE/APE are not meaningful for mono |
| euroc_V2_02_medium_inertial_odom | 14.1 | 1.5 | DIVERGED (avgRTE 14.1% ≥ 5.0%) |
| euroc_V2_02_medium_inertial_slam | 10.8 | 1.7 | DIVERGED (avgRTE 10.8% ≥ 5.0%) |
| euroc_V2_02_medium_mono_odom | 48.3 | 1.7 | scale-ambiguous: monocular needs Sim3 (scale) alignment; SE3-based avgRTE/APE are not meaningful for mono |
| euroc_V2_03_difficult_inertial_odom | 236542.1 | 68863.3 | DIVERGED (avgRTE 236542.1%): V2_03_difficult is the hardest EuRoC sequence — aggressive motion + blur; the paper reports it only in stereo-inertial and excludes it from stereo averages |
| euroc_V2_03_difficult_inertial_slam | 10801.9 | 2181.5 | DIVERGED (avgRTE 10801.9%): V2_03_difficult is the hardest EuRoC sequence — aggressive motion + blur; the paper reports it only in stereo-inertial and excludes it from stereo averages |
| euroc_V2_03_difficult_mono_odom | 53.9 | 1.8 | scale-ambiguous: monocular needs Sim3 (scale) alignment; SE3-based avgRTE/APE are not meaningful for mono |
| euroc_V2_03_difficult_stereo_odom | 634.4 | 98.9 | DIVERGED (avgRTE 634.4%): V2_03_difficult is the hardest EuRoC sequence — aggressive motion + blur; the paper reports it only in stereo-inertial and excludes it from stereo averages |
| euroc_V2_03_difficult_stereo_slam | 634.4 | 98.9 | DIVERGED (avgRTE 634.4%): V2_03_difficult is the hardest EuRoC sequence — aggressive motion + blur; the paper reports it only in stereo-inertial and excludes it from stereo averages |
| tum_fr3_nostructure_notexture_far_rgbd_odom | 48.6 | 0.4 | DIVERGED (avgRTE 48.6% ≥ 5.0%) |
| tum_fr3_nostructure_notexture_far_rgbd_slam | 48.6 | 0.4 | DIVERGED (avgRTE 48.6% ≥ 5.0%) |
| tum_fr3_nostructure_notexture_far_rgbd_slam_cpu | 48.6 | 0.4 | DIVERGED (avgRTE 48.6% ≥ 5.0%) |
| tum_fr3_nostructure_notexture_near_withloop_rgbd_odom | 19.8 | 0.4 | DIVERGED (avgRTE 19.8% ≥ 5.0%) |
| tum_fr3_nostructure_notexture_near_withloop_rgbd_slam | 19.8 | 0.4 | DIVERGED (avgRTE 19.8% ≥ 5.0%) |
| tum_fr3_nostructure_notexture_near_withloop_rgbd_slam_cpu | 19.8 | 0.4 | DIVERGED (avgRTE 19.8% ≥ 5.0%) |
| tum_fr3_nostructure_texture_far_rgbd_odom | 18.6 | 0.3 | DIVERGED (avgRTE 18.6% ≥ 5.0%) |
| tum_fr3_nostructure_texture_far_rgbd_slam | 18.6 | 0.3 | DIVERGED (avgRTE 18.6% ≥ 5.0%) |
| tum_fr3_nostructure_texture_far_rgbd_slam_cpu | 18.6 | 0.3 | DIVERGED (avgRTE 18.6% ≥ 5.0%) |
| tumvi_corridor1_inertial_odom | 31984.9 | 4970.6 | INVALID: TUM-VI ~195° fisheye not undistorted to pinhole (<180° cuVSLAM support) — data-prep gap, not a tracking result |
| tumvi_corridor1_inertial_slam | 36249.6 | 5922.1 | INVALID: TUM-VI ~195° fisheye not undistorted to pinhole (<180° cuVSLAM support) — data-prep gap, not a tracking result |
| tumvi_magistrale1_inertial_odom | 16390.9 | 4209.3 | INVALID: TUM-VI ~195° fisheye not undistorted to pinhole (<180° cuVSLAM support) — data-prep gap, not a tracking result |
| tumvi_magistrale1_inertial_slam | 4770.8 | 1198.9 | INVALID: TUM-VI ~195° fisheye not undistorted to pinhole (<180° cuVSLAM support) — data-prep gap, not a tracking result |

## TUM fr3 per-sequence vs paper Table 6 (shared sequences)

| sequence / mode | APE m (ours) | APE m (paper) | avgRTE (ours/paper) | avgRE (ours/paper) |
|---|---|---|---|---|
| fr3_long_office_household odom | 0.109 | 0.2 | 1.798 / 1.27 | 1.19 / 7.89 |
| fr3_long_office_household slam | 0.018 | 0.06 | 1.447 / 0.96 | 1.01 / 5.94 |
| fr3_nostructure_texture_far odom | 0.272 | 0.07 | 18.580 / 1.29 | 8.68 / 1.66 |
| fr3_nostructure_texture_far slam | 0.272 | 0.06 | 18.580 / 1.44 | 8.68 / 1.63 |

## Feature-toggle deltas (converged pairs only, ours vs ours)

Same sequence, one feature toggled; negative ΔAPE = the toggle improved accuracy. Pairs with a diverged member are dropped (count shown).

| toggle | pairs | mean ΔAPE m (toggle − base) |
|---|---|---|
| SLAM vs odometry | 25 | -0.3204 |
| async SLAM vs sync SLAM (KITTI) | 11 | 0.1405 |
| CPU SLAM vs GPU SLAM (TUM) | 1 | 0.0579 |
| IMU on vs off (EuRoC odom) | 3 | 0.2996 |
| IMU on vs off (EuRoC slam) | 3 | 0.2779 |

## QoR: instrumented wheel vs baseline (same configs)

| run | APE m baseline | APE m instrumented | Δ |
|---|---|---|---|
| euroc_MH_01_easy_inertial_slam | 0.689 | 0.689 | +0.0000 |
| euroc_V1_01_easy_stereo_slam | 0.037 | 0.037 | +0.0000 |
| euroc_V2_02_medium_stereo_odom | 0.177 | 0.177 | +0.0000 |
| kitti00_stereo_odom | 6.841 | 6.605 | -0.2360 |
| kitti06_stereo_slam | 2.312 | 2.277 | -0.0351 |
| tum_fr3_long_office_household_rgbd_slam | 0.018 | 0.021 | +0.0031 |

## All runs (per-sequence)

| run | APE m | avgRTE % | avgRE ° | RPE@1s m/° |
|---|---|---|---|---|
| euroc_MH_01_easy_inertial_odom | 0.714 | 2.245 | 6.81 | 0.034/1.28 |
| euroc_MH_01_easy_inertial_slam | 0.689 | 2.025 | 6.53 | 0.037/1.27 |
| euroc_MH_01_easy_mono_odom | 0.401 | 110.084 | 7.94 | 1.521/1.67 |
| euroc_MH_01_easy_stereo_odom | 0.059 | 0.614 | 0.86 | 0.011/0.08 |
| euroc_MH_01_easy_stereo_slam | 0.020 | 0.637 | 0.73 | 0.014/0.13 |
| euroc_MH_02_easy_inertial_odom | 0.506 | 5.830 | 8.16 | 0.132/9.63 |
| euroc_MH_02_easy_inertial_slam | 2.540 | 12.439 | 23.38 | 0.851/14.29 |
| euroc_MH_02_easy_mono_odom | 1.334 | 344.905 | 2.48 | 5.818/1.80 |
| euroc_MH_02_easy_stereo_odom | 0.024 | 0.443 | 0.45 | 0.008/0.10 |
| euroc_MH_02_easy_stereo_slam | 0.022 | 0.514 | 0.47 | 0.014/0.15 |
| euroc_MH_03_medium_inertial_odom | 3.410 | 12.275 | 23.28 | 0.542/9.91 |
| euroc_MH_03_medium_inertial_slam | 1.938 | 8.402 | 10.36 | 0.531/6.86 |
| euroc_MH_03_medium_mono_odom | 2.581 | 21.320 | 2.56 | 6.517/1.08 |
| euroc_MH_03_medium_stereo_odom | 0.071 | 0.943 | 0.59 | 0.029/0.17 |
| euroc_MH_03_medium_stereo_slam | 0.032 | 0.945 | 0.73 | 0.033/0.24 |
| euroc_MH_04_difficult_inertial_odom | 1.416 | 5.644 | 5.81 | 0.152/2.83 |
| euroc_MH_04_difficult_inertial_slam | 0.863 | 5.301 | 5.20 | 0.145/2.11 |
| euroc_MH_04_difficult_mono_odom | 1.779 | 11.577 | 4.94 | 1.595/2.56 |
| euroc_MH_04_difficult_stereo_odom | 0.081 | 1.222 | 0.56 | 0.032/0.14 |
| euroc_MH_04_difficult_stereo_slam | 0.085 | 1.257 | 0.57 | 0.032/0.16 |
| euroc_MH_05_difficult_inertial_odom | 2.845 | 9.667 | 15.00 | 0.222/4.74 |
| euroc_MH_05_difficult_inertial_slam | 2.845 | 9.667 | 15.00 | 0.222/4.74 |
| euroc_MH_05_difficult_mono_odom | 3.365 | 285.030 | 4.48 | 4.778/1.94 |
| euroc_MH_05_difficult_stereo_odom | 0.141 | 1.019 | 0.85 | 0.029/0.17 |
| euroc_MH_05_difficult_stereo_slam | 0.065 | 0.911 | 0.80 | 0.030/0.19 |
| euroc_V1_01_easy_inertial_odom | 0.057 | 1.944 | 3.07 | 0.045/0.54 |
| euroc_V1_01_easy_inertial_slam | 0.039 | 1.933 | 2.94 | 0.045/0.53 |
| euroc_V1_01_easy_mono_odom | 1.476 | 466.330 | 18.55 | 9.986/2.38 |
| euroc_V1_01_easy_stereo_odom | 0.078 | 1.949 | 3.12 | 0.044/0.47 |
| euroc_V1_01_easy_stereo_slam | 0.037 | 1.930 | 2.90 | 0.044/0.48 |
| euroc_V1_02_medium_inertial_odom | 1.622 | 12.446 | 30.08 | 0.444/9.73 |
| euroc_V1_02_medium_inertial_slam | 0.634 | 6.244 | 19.35 | 0.293/7.63 |
| euroc_V1_02_medium_mono_odom | 1.634 | 396.845 | 36.70 | 15.545/5.42 |
| euroc_V1_02_medium_stereo_odom | 0.152 | 1.038 | 1.85 | 0.039/0.45 |
| euroc_V1_02_medium_stereo_slam | 0.031 | 0.743 | 0.74 | 0.044/0.56 |
| euroc_V1_03_difficult_inertial_odom | 2.301 | 16.191 | 57.77 | 0.496/17.75 |
| euroc_V1_03_difficult_inertial_slam | 1.977 | 13.807 | 58.18 | 0.484/16.43 |
| euroc_V1_03_difficult_mono_odom | 1.526 | 47.493 | 90.42 | 4.266/30.59 |
| euroc_V1_03_difficult_stereo_odom | 0.190 | 1.214 | 1.97 | 0.046/0.58 |
| euroc_V1_03_difficult_stereo_slam | 0.064 | 0.968 | 1.35 | 0.061/0.87 |
| euroc_V2_01_easy_inertial_odom | 0.433 | 4.071 | 10.49 | 0.063/3.06 |
| euroc_V2_01_easy_inertial_slam | 0.255 | 3.561 | 9.18 | 0.081/3.18 |
| euroc_V2_01_easy_mono_odom | 1.816 | 832.852 | 28.80 | 12.473/3.24 |
| euroc_V2_01_easy_stereo_odom | 0.168 | 1.419 | 2.73 | 0.036/0.89 |
| euroc_V2_01_easy_stereo_slam | 0.093 | 0.736 | 1.28 | 0.042/0.97 |
| euroc_V2_02_medium_inertial_odom | 1.500 | 14.092 | 37.42 | 0.397/15.69 |
| euroc_V2_02_medium_inertial_slam | 1.650 | 10.812 | 32.57 | 0.429/14.56 |
| euroc_V2_02_medium_mono_odom | 1.666 | 48.281 | 4.65 | 2.613/1.92 |
| euroc_V2_02_medium_stereo_odom | 0.177 | 0.963 | 2.40 | 0.028/0.54 |
| euroc_V2_02_medium_stereo_slam | 0.063 | 0.576 | 1.34 | 0.058/1.04 |
| euroc_V2_03_difficult_inertial_odom | 68863.254 | 236542.094 | 116.29 | 11750.607/47.21 |
| euroc_V2_03_difficult_inertial_slam | 2181.508 | 10801.909 | 111.94 | 450.080/42.70 |
| euroc_V2_03_difficult_mono_odom | 1.803 | 53.920 | 44.52 | 3.433/15.14 |
| euroc_V2_03_difficult_stereo_odom | 98.916 | 634.396 | 96.98 | 20.665/17.20 |
| euroc_V2_03_difficult_stereo_slam | 98.914 | 634.399 | 96.97 | 20.665/17.20 |
| kitti00_stereo_odom | 6.841 | 0.892 | 0.94 | —/— |
| kitti00_stereo_slam | 2.131 | 0.846 | 0.70 | —/— |
| kitti00_stereo_slam_async | 2.146 | 0.845 | 0.68 | —/— |
| kitti01_stereo_odom | 11.587 | 1.899 | 0.39 | —/— |
| kitti01_stereo_slam | 10.453 | 1.820 | 0.51 | —/— |
| kitti01_stereo_slam_async | 11.586 | 1.899 | 0.39 | —/— |
| kitti02_stereo_odom | 4.548 | 0.975 | 0.71 | —/— |
| kitti02_stereo_slam | 5.347 | 1.112 | 0.82 | —/— |
| kitti02_stereo_slam_async | 6.086 | 1.144 | 1.07 | —/— |
| kitti03_stereo_odom | 3.941 | 2.265 | 0.99 | —/— |
| kitti03_stereo_slam | 4.054 | 2.276 | 0.73 | —/— |
| kitti03_stereo_slam_async | 4.054 | 2.276 | 0.73 | —/— |
| kitti04_stereo_odom | 2.155 | 1.967 | 0.09 | —/— |
| kitti04_stereo_slam | 2.156 | 1.967 | 0.09 | —/— |
| kitti04_stereo_slam_async | 2.156 | 1.967 | 0.09 | —/— |
| kitti05_stereo_odom | 3.310 | 0.974 | 0.72 | —/— |
| kitti05_stereo_slam | 2.155 | 0.840 | 0.48 | —/— |
| kitti05_stereo_slam_async | 2.142 | 0.837 | 0.47 | —/— |
| kitti06_stereo_odom | 2.883 | 1.248 | 0.76 | —/— |
| kitti06_stereo_slam | 2.312 | 1.083 | 0.46 | —/— |
| kitti06_stereo_slam_async | 2.324 | 1.094 | 0.45 | —/— |
| kitti07_stereo_odom | 1.438 | 0.982 | 1.13 | —/— |
| kitti07_stereo_slam | 1.032 | 0.785 | 0.51 | —/— |
| kitti07_stereo_slam_async | 1.033 | 0.784 | 0.51 | —/— |
| kitti08_stereo_odom | 4.566 | 1.221 | 0.96 | —/— |
| kitti08_stereo_slam | 4.386 | 1.214 | 1.00 | —/— |
| kitti08_stereo_slam_async | 4.289 | 1.217 | 0.95 | —/— |
| kitti09_stereo_odom | 3.528 | 1.238 | 0.54 | —/— |
| kitti09_stereo_slam | 3.691 | 1.263 | 0.52 | —/— |
| kitti09_stereo_slam_async | 3.459 | 1.254 | 0.60 | —/— |
| kitti10_stereo_odom | 2.169 | 1.010 | 0.66 | —/— |
| kitti10_stereo_slam | 2.180 | 1.007 | 0.69 | —/— |
| kitti10_stereo_slam_async | 2.169 | 1.010 | 0.66 | —/— |
| tum_fr3_long_office_household_rgbd_odom | 0.109 | 1.798 | 1.19 | 0.011/0.50 |
| tum_fr3_long_office_household_rgbd_slam | 0.018 | 1.447 | 1.01 | 0.010/0.49 |
| tum_fr3_long_office_household_rgbd_slam_cpu | 0.076 | 1.638 | 1.03 | 0.010/0.49 |
| tum_fr3_nostructure_notexture_far_rgbd_odom | 0.414 | 48.562 | 11.55 | 0.160/3.94 |
| tum_fr3_nostructure_notexture_far_rgbd_slam | 0.414 | 48.562 | 11.55 | 0.160/3.94 |
| tum_fr3_nostructure_notexture_far_rgbd_slam_cpu | 0.414 | 48.562 | 11.55 | 0.160/3.94 |
| tum_fr3_nostructure_notexture_near_withloop_rgbd_odom | 0.390 | 19.839 | 13.25 | 0.125/4.30 |
| tum_fr3_nostructure_notexture_near_withloop_rgbd_slam | 0.390 | 19.839 | 13.25 | 0.125/4.30 |
| tum_fr3_nostructure_notexture_near_withloop_rgbd_slam_cpu | 0.390 | 19.839 | 13.25 | 0.125/4.30 |
| tum_fr3_nostructure_texture_far_rgbd_odom | 0.272 | 18.580 | 8.68 | 0.094/2.29 |
| tum_fr3_nostructure_texture_far_rgbd_slam | 0.272 | 18.580 | 8.68 | 0.094/2.29 |
| tum_fr3_nostructure_texture_far_rgbd_slam_cpu | 0.272 | 18.580 | 8.68 | 0.094/2.29 |
| tumvi_corridor1_inertial_odom | 4970.569 | 31984.922 | 118.99 | 1471.092/49.25 |
| tumvi_corridor1_inertial_slam | 5922.069 | 36249.639 | 104.45 | 1651.333/46.05 |
| tumvi_magistrale1_inertial_odom | 4209.333 | 16390.887 | 91.84 | 827.782/39.35 |
| tumvi_magistrale1_inertial_slam | 1198.924 | 4770.758 | 99.60 | 423.992/38.74 |
