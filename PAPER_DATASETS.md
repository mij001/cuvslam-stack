# Paper-benchmark dataset coverage

Every dataset benchmarked in the cuVSLAM technical report
([arXiv:2506.04359](https://arxiv.org/abs/2506.04359)) has a ready config in
`configs/`. This covers **all result tables** — the main Tables 1–3 and the
appendix per-sequence tables (4–11) and figures (10–13). The runner reports the
paper's own metrics: `avgRTE %`, `avgRE deg`, `RMSE APE` (EVO-style: timestamp
association + alignment, Sim3 scale for monocular).

Reading the whole paper confirmed there are **no additional datasets** beyond
the ten below (the appendix tables are per-sequence breakdowns of these). Note:
NCLT, HILTI, M2DGR and UT-CODA appear in the references only as datasets cuVSLAM
did **not** use (cited as unsuitable for hardware-synced multi-camera), so they
are intentionally not included.

## Coverage map

| Paper mode | Dataset | Tables/Figs | Config | Source | Status |
|---|---|---|---|---|---|
| Mono-Depth | AR-table (Chen 2023) | 2, 5 | `artable_rgbd.toml` | image_folder (rgb+depth) | config — D455 intrinsics |
| Mono-Depth | ICL-NUIM (Handa 2014) | 2 | `iclnuim_rgbd.toml` | tum | config — verify intrinsics |
| Mono-Depth | TUM RGB-D (Sturm 2012) | 2, 6 | `tum_rgbd.toml` | tum | ✓ runnable + eval |
| Stereo | EuRoC (Burri 2016) | 2 | `euroc_v1_eval.toml` | euroc | **validated** (ATE 7.78 cm) |
| Stereo | KITTI (Geiger 2013) | 2, Fig 10 | `kitti_eval.toml`, `kitti_stereo.toml` | image_folder | **validated** tracking; eval needs GT poses |
| Stereo-Inertial | EuRoC (Burri 2016) | 2 | `euroc_inertial.toml` | euroc | **validated** (runs V1_01) |
| Stereo-Inertial | TUM-VI Room (Schubert 2018) | 2, 4 | `tumvi_room_inertial.toml` | image_folder + imu | config — undistort first |
| Multi-Stereo | TartanAir V2 (Wang 2023) | 2/3, 8, 10, 11 | `tartanair_v2_multicam.toml` | edex | config — convert via tool |
| Multi-Stereo | TartanGround (Patel 2025) | 3, 7, 9 | `tartan_multicam.toml` | edex | ✓ rig parses (12 cams) |
| Multi-Stereo | R2B (NVIDIA, proprietary) | 3, Fig 12 | `r2b_multicam.toml` | edex (jsonl) | config — data is private |

## Per-dataset validation setup (from the appendix)

- **KITTI** (A.1.1, Fig 10): sequences **00–10** (GT poses available). The paper
  reports relative translation/rotation over **100–800 m segments** (KITTI
  leaderboard method): **0.85 % / 0.0025 deg/m**. `kitti_eval.toml` uses
  `gt_format="kitti"` + `rpe_distances="kitti"` and indexes frames at 10 Hz to
  align with the GT poses. KITTI GT poses are a separate download.
- **TUM-VI Room** (A.1.2, Table 4): rooms **room1–room6**, stereo-inertial.
  **Undistorted 512×512** images + **50 px edge masks**; the raw ~195°-FOV
  fisheye exceeds cuVSLAM's pinhole-undistort fisheye support (<180°), so frames
  must be undistorted to pinhole first. Paper metric is RMSE RPE over 1-s segments.
- **AR-table** (A.1.3, Table 5): RealSense D455, **848×480** RGB+depth, masks
  **24 px sides / 10 px bottom**, depth in mm. Sequences **table_01–table_08**
  (3,4,5,8 have depth dropouts → segmented). Paper metric: ATE (cm) + rotation (deg).
- **TUM RGB-D** (A.1.4, Table 6): 10 freiburg3 sequences — large_cabinet,
  long_office_household, nostructure_texture_far, nostructure_texture_near_withloop,
  sitting_halfsphere, sitting_xyz(+validation), structure_texture_far/near, teddy.
  Rectified RGB-D, depth scale 5000.
- **TartanGround / TartanAir V2** (A.1.5, Tables 7–11): **four stereo cameras**
  (front/back/left/right), **640×640 undistorted** — the top/bottom pairs add
  <5 % and are omitted in the paper. TartanGround: 216 seqs (odom), 32 (SLAM);
  TartanAir V2 Hard: 539 (odom), 506 (SLAM). `tartan_multicam.toml` lists all 12
  cameras to match the repo's example `.edex`; trim to 8 for the paper config.
- **R2B** (A.1.6, Fig 12): four hardware-synced **Hawk stereo** cameras,
  **1920×1200 @ 30 fps**. Office (1–3) + Warehouse (1–4) environments.

## How each mode is driven
- **Mono-Depth (RGBD)** — one camera + aligned `uint16` depth. `tum` source when
  the dataset ships `rgb.txt`/`depth.txt`; `image_folder` (with a `depth` glob)
  for parallel rgb/depth folders. Set `[odometry.rgbd].depth_scale_factor`.
- **Stereo** — `euroc` (rig from dataset yaml) or `image_folder` + explicit `[rig]`.
- **Stereo-Inertial** — add IMU: `euroc`, or `image_folder` + `[input.imu]` + `[rig.imu]`.
- **Multi-Stereo** — `edex`: N-camera rig from a `.edex` JSON; frames from
  per-camera folders (`layout="folders"`) or a `frame_metadata.jsonl` (`layout="jsonl"`, R2B).

## Status legend
- **validated** — actually run here against the cu13 wheel with real data.
- **✓** — runs/parses with data present; format confirmed.
- **config** — schema-valid config with published calibration to confirm against
  your specific download (intrinsics/extrinsics/depth-scale vary per release; the
  runner cannot infer them for `image_folder`/`video`).

Multi-stereo GT (TartanAir/TartanGround) uses per-camera NED pose files; a
frame-converting GT loader isn't wired, so those configs track without `[eval]`.
EuRoC/TUM/KITTI GT formats are fully supported for ATE/RPE.
