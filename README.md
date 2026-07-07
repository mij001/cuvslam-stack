# cuVSLAM stack

Umbrella project that builds **NVIDIA cuVSLAM** from source and runs it through a
single-TOML **runner**. The main project is *not* the runner — it ties together:

```
cuvslam-stack/                 (this repo — the main project)
├── run.py evaluate.py run_list.py runner.sh setup_env.sh   the TOML runner entrypoints
├── cuvslam_runner/            the runner package itself
├── configs/                   THE config tree (see configs/README.md):
│     *.toml (examples) · base/ (65 canonical, committed) · profiling/ (workloads)
│     generated/ + custom/     every variant = a mutation of a base (gitignored, regen)
├── scripts/                   gen_base_configs + mutate_configs (THE mutation engine),
│                              validation_regime.sh, dataset tooling, campaign drivers
│                              (shared shell lib.sh · dataset_catalog.py — one source each)
├── profiling/                 memory characterization: regime.py (cohesive pipeline:
│                              nsys→window→ncu→nvbit→analyses), harness/profile.py (one
│                              capture entrypoint), analysis/ (incl. substrate dynamics)
├── viz/ · dashboard/ · site/  figures for every output · web UI · static results site
├── reports/                   committed campaign results (accuracy, coverage, neutrality)
├── cuvslam_src/               git submodule -> nvidia-isaac/cuVSLAM @ efdfbe56 (release 15.0)
├── patches/                   our build tooling, applied onto the pristine submodule
└── Makefile                   build the wheel + verify, reproducibly
```

The `cuvslam_src/` submodule is pinned to upstream commit `efdfbe56` and kept
**pristine**; the Podman/CUDA-13 wheel tooling lives in `patches/` and is applied
at build time (`git apply`). (It is named `cuvslam_src`, not `cuvslam`, so the
source directory never shadows the installed `cuvslam` Python package.) So a clean
clone reproduces the exact wheel anywhere:

```bash
git clone https://github.com/mij001/cuvslam-stack && cd cuvslam-stack
make wheel     # fetch the pinned cuVSLAM submodule -> apply patches -> build the wheel (Podman, Python 3.10, CUDA 13)
make verify    # install that wheel via setup_env.sh and run configs/kitti_eval.toml
```

`make wheel` runs `git submodule update --init` for you (with `GIT_LFS_SKIP_SMUDGE=1`,
since the build needs only the source, not the LFS-stored example media). Don't
`git clone --recurse-submodules` — that pulls the upstream LFS assets and is slow;
let `make` fetch the submodule instead (or `GIT_LFS_SKIP_SMUDGE=1 git submodule update --init cuvslam_src`).

`make` targets: `wheel`, `verify`, `check`, `clean`, `unpatch`, `all`. The wheel
lands in `cuvslam_src/dist/`; `verify` passes it to the runner's `setup_env.sh`.

---

## Full workflow — from a clean clone to results

Everything needed to run lives in this repo (code, configs, harness, hardware
descriptors, reports). The only things **not** in git are the large externals:
the datasets, the built wheel, the Python venv, and raw profiler captures
(`*.nsys-rep`/`*.ncu-rep`). So **nuke-and-reclone works**: delete the checkout,
re-clone, rebuild the wheel, re-point the datasets, and every run/campaign below
reproduces.

The Makefile separates the phases — **build** (wheel/venv, no GPU profiling),
**configs** (bases → mutations), **profiling** (validation regime + the cohesive
capture pipeline), **analysis** (substrate + figures + site). `make help` prints
the map.

```bash
# 0 · prerequisites: NVIDIA GPU + driver, Podman (for the wheel build), python3.10
git clone https://github.com/mij001/cuvslam-stack && cd cuvslam-stack

# BUILD phase — wheel + venv (kept apart from profiling)
make build                 # = make wheel (submodule+patches, Podman) + ./setup_env.sh
#   CUDA-mismatched host? build/install a matching wheel and pass WHEEL=... to setup_env.sh

# datasets (paper benchmark set) onto a data volume
DATA=/mnt/data scripts/fetch_paper_datasets.sh  # aria2c: TUM fr3 + ICL-NUIM (see PAPER_DATASETS.md)
#   KITTI / EuRoC / TUM-VI are large — fetch per PAPER_DATASETS.md and place under $DATA

# CONFIG phase — ONE tree: configs/base + script mutations (see configs/README.md)
make configs               # gen_base_configs (scans $DATA) + mutate_configs --select all
#   -> configs/generated/{accuracy,coverage,window}; bases are committed

# run one config, a list, or everything
./cuvslam_venv/bin/python run.py configs/base/euroc_MH_01_easy_stereo_slam.toml
python run_list.py --configs configs/generated/accuracy --check    # validate the whole set

# PROFILING phase (workstation, clocks locked)
make validate SCOPE=accuracy   # the validation regime: {base+mutated} × {plain,nsys,ncu,nvbit}
                               # scopes: reps | accuracy | coverage | full (see the script header)
make profile CFG=configs/base/kitti06_stereo_slam.toml   # cohesive pipeline on one workload:
                               # nsys → steady-window → ncu characterize → nvbit trace →
                               # DAG/screen/roofline/classify/locality analyses + manifest

# ANALYSIS phase (anywhere — works from the committed tables)
make analyze               # substrate candidacy (GPU/CPU/PiM/ISP) + dynamics/flips
make site                  # every figure + the browsable results site (site/index.html)
./cuvslam_venv/bin/python dashboard/serve.py             # web UI on http://127.0.0.1:8642/
```

**Pointing configs at your datasets.** The committed `configs/base/` files use
absolute dataset paths (the workstation's `/mnt/data/...`). On a different host,
regenerate the bases against your own data root (`make configs DATA=/your/data`),
edit the `[input].path` / `[eval].ground_truth` fields, or use the **dashboard**
(§ *Dashboard & results site*): register a dataset from a preset template and it
writes correctly pointed TOML variants into `configs/custom/` for you.

**Reproduce the campaigns** (workstation; each streams a tailable log and is
resumable). These produced the committed results under `reports/`:

```bash
scripts/ws_accuracy_matrix.sh        # 141-run accuracy matrix vs the paper  -> reports/2026-07-07_accuracy_full/
scripts/validation_regime.sh accuracy# configs × {plain,nsys,ncu,nvbit} accuracy-neutrality regime
                                     #   (supersedes the retired ws_profiling_campaign /
                                     #    ws_profiler_neutrality scripts; those results:
                                     #    reports/2026-07-07_profiling_coverage + _profiler_neutrality)
scripts/validate_accuracy_configs.sh # every config validates under runner + profiling flow
```

The rest of this README documents the runner itself; deeper profiling docs are in
[`profiling/README.md`](profiling/README.md) and
[`profiling/PROJECT_STATUS.md`](profiling/PROJECT_STATUS.md).

---

# cuvslam_runner

**Run NVIDIA cuVSLAM end-to-end from a single TOML file — any input, any mode,
with built-in trajectory evaluation against ground truth.**

`cuvslam_runner` is a thin, declarative harness around the `cuvslam` Python
module. One TOML file fully describes a run: where the images/IMU come from
(recorded dataset *or* live camera/stream), the camera+IMU rig, every Odometry
and SLAM knob, the output trajectory/map, and an optional accuracy evaluation
(ATE / RPE / avgRTE / avgRE) against ground truth. Nothing else is needed — no
per-dataset Python script.

```bash
python run.py configs/kitti_stereo.toml            # run a config
python run.py configs/euroc_v1_eval.toml            # run + evaluate vs ground truth
python run.py configs/kitti_stereo.toml --check     # validate config only (no cuvslam import)
python -m cuvslam_runner configs/tum_rgbd.toml      # equivalent module form
python evaluate.py est.txt gt.csv --gt-format euroc # evaluate an existing trajectory
python run_list.py                                   # run every config + summary table
python run_list.py --check                           # validate every config
```

Or use the **`runner.sh`** launcher, which finds a Python that has `cuvslam`,
sets `PYTHONPATH`, and dispatches — no venv/PYTHONPATH juggling:

```bash
./runner.sh configs/euroc_v1_eval.toml      # run one config        (-> run.py)
./runner.sh check configs/kitti_stereo.toml  # validate one config   (-> run.py --check)
./runner.sh all                              # run every config      (-> run_list.py)
./runner.sh eval est.txt gt.csv --gt-format euroc   # evaluate         (-> evaluate.py)
# Override the interpreter explicitly when needed:
CUVSLAM_PYTHON=/path/to/venv/bin/python ./runner.sh configs/tum_rgbd.toml
```

Interpreter search order: `$CUVSLAM_PYTHON` → `./.venv/bin/python` →
`../../wheel/cuvslam_env/bin/python` (this repo's layout) → `python3`/`python`.
The first one that can `import cuvslam` wins; otherwise the first that exists
(so `check` still works without the wheel).

---

## Table of contents

- [Why this exists](#why-this-exists)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Input sources](#input-sources)
- [The TOML reference](#the-toml-reference)
  - [`[run]`](#run)
  - [`[input]`](#input)
  - [`[rig]` — cameras and IMU](#rig--cameras-and-imu)
  - [`[odometry]`](#odometry)
  - [`[slam]`](#slam)
  - [`[output]`](#output)
  - [`[eval]`](#eval)
- [Choosing a mode](#choosing-a-mode)
- [Evaluation metrics explained](#evaluation-metrics-explained)
- [Validated benchmark results](#validated-benchmark-results)
- [Dashboard & results site](#dashboard--results-site)
- [Bundled configs](#bundled-configs)
- [Running all configs](#running-all-configs)
- [Recipes](#recipes)
- [Coordinate conventions](#coordinate-conventions)
- [Extending the system](#extending-the-system)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Why this exists

The cuVSLAM repo ships one bespoke Python script per dataset (KITTI, EuRoC, TUM,
TartanGround, RealSense, ZED, …). They duplicate the same scaffolding — build a
rig, configure the tracker, loop over frames, call `track()`, collect poses.

`cuvslam_runner` factors that into a single, reusable engine driven by
configuration. The same binary runs the paper's **benchmark datasets**, the
**repo's example datasets**, and **arbitrary external data** (recorded folders,
video files, USB/IP cameras), and then **scores the result** against ground
truth. Adapting to a new dataset means writing a TOML file, not code.

---

## Installation

`cuvslam_runner` needs the `cuvslam` wheel plus a few common Python packages.

```bash
# 1) core dependencies for the harness
pip install -r requirements.txt

# 2) the cuVSLAM module itself (from the releases page or your local build)
pip install cuvslam-15.0.0+cu13-cp310-cp310-manylinux_2_35_x86_64.whl
```

> Make sure the wheel matches your CUDA runtime (e.g. a `cu11`-built wheel needs
> `libcusolver.so.11` on the library path; a `cu13` build needs CUDA 13).

**Scripted setup → run → teardown.** Three composable scripts manage a
throwaway venv and run a list of configs:

```bash
# 1) build the venv and install requirements.txt + the newest ../dist/cuvslam-*.whl
./setup_env.sh
#    overrides: PYBIN=python3.10  VENV=/path  WHEEL=/path/to/cuvslam-*.whl

# 2) run every config listed in a text file (one TOML path per line)
python run_list.py runlist.txt            # track each   (-> run.py per config)
python run_list.py runlist.txt --check    # validate each (no cuvslam needed)
#    run_list.py auto-uses ./cuvslam_venv/bin/python when present; override with --python

# 3) remove the venv (and optionally generated outputs)
./cleanup_env.sh                          # remove the venv + __pycache__
./cleanup_env.sh --outputs                # also remove out/
```

`runlist.txt` lists the configs to run (blank lines and `#`-comments ignored;
relative paths resolve against the list file). Each config runs in its own
subprocess, so one failure never stops the rest, and `run_list.py` prints the
same summary table in both modes (status, frames, ATE, RTE%). If the wheel
can't be imported (e.g. a CUDA mismatch), `setup_env.sh` says so and `--check`
still works for validation.

**Dependencies**

| Package | Needed for |
|---|---|
| numpy, pillow | core (image loading, math) |
| pyyaml | EuRoC / TUM calibration parsing |
| scipy | rotations, Umeyama alignment (rig + eval) |
| tomli | TOML parsing on Python < 3.11 (3.11+ has `tomllib`) |
| `cuvslam` | the tracker (only imported at run time, not for `--check`) |
| rerun-sdk *(optional)* | `[output].visualize = true` |
| opencv-python *(optional)* | `type = "video"`, and Bayer demosaicing |
| pyrealsense2 *(optional)* | `type = "realsense"` live camera |

> `--check` and the whole config/parsing/evaluation layer work **without
> `cuvslam` installed**, which makes configs easy to validate anywhere.

---

## Quick start

```bash
# Stereo visual odometry on a KITTI sequence
python run.py configs/kitti_stereo.toml

# EuRoC V1_01 stereo, then ATE/RPE against the Vicon ground truth
python run.py configs/euroc_v1_eval.toml

# A live monocular webcam (needs opencv-python)
python run.py configs/webcam_mono.toml
```

Every run prints a one-line summary, e.g.:

```
[runner] done: {'frames_tracked': 2912, 'track_failures': 0,
                'pose_source': 'odometry', 'slam_enabled': False,
                'ate_rmse_m': 0.07777, 'avg_rte_pct': 5.229, 'avg_re_deg': 2.3388}
```

---

## Architecture

The design keeps a hard wall between *parsing* (no cuVSLAM) and *construction*
(cuVSLAM), so configs and datasets can be validated/tested without a built wheel.

```
                         one TOML file
                              │
                  ┌───────────▼────────────┐
                  │       config.py         │  parse + validate -> dataclasses
                  └───────────┬────────────┘
                              │  specs.py  (plain dataclasses, NO cuvslam)
        ┌─────────────────────┼─────────────────────────────┐
        │                     │                             │
 ┌──────▼───────┐    ┌────────▼─────────┐          ┌─────────▼────────┐
 │  sources/*   │    │   builders.py    │          │     eval.py      │
 │ FrameEvent / │    │  specs -> live   │          │ ATE / RPE / ...  │
 │ ImuEvent +   │    │  cuvslam objects │          │ (no cuvslam)     │
 │ RigSpec      │    │ (ONLY cuvslam    │          └─────────▲────────┘
 └──────┬───────┘    │     importer)    │                    │
        │            └────────┬─────────┘                    │
        └─────────────┐       │                              │
                  ┌───▼───────▼──────┐    poses (ns)          │
                  │     runner.py    │────────────────────────┘
                  │  Tracker loop    │
                  └───┬─────────┬────┘
                      │         │
              trajectory.py   viz.py
              (TUM file)      (rerun, optional)
```

**Module map**

| File | Responsibility |
|---|---|
| `config.py` | Load TOML, validate keys, build `specs` dataclasses. Rejects unknown keys with helpful messages. |
| `specs.py` | Dependency-free dataclasses: `RigSpec`, `CameraSpec`, `OdometrySpec`, `SlamSpec`, `EvalSpec`, … |
| `sources/` | Input adapters; each yields `FrameEvent`/`ImuEvent` and may provide a `RigSpec`. |
| `builders.py` | The only module importing `cuvslam`; turns specs into `Rig`, `OdometryConfig`, `SlamConfig`, etc. |
| `runner.py` | Drives the `Tracker` loop, collects poses, writes outputs, runs eval. |
| `trajectory.py` | TUM-format trajectory writer (`ts tx ty tz qx qy qz qw`). |
| `eval.py` | Association, Umeyama alignment, ATE/RPE metrics; standalone-capable. |
| `images.py` | Robust image loading (mono/RGB→BGR, uint16 depth, Bayer demosaic). |
| `viz.py` | Optional, generic Rerun visualization (degrades to a no-op). |
| `cli.py` / `run.py` / `__main__.py` | Entry points and `--check` dry run. |

---

## Input sources

Selected by `[input].type`. The first three columns answer your core question —
**recorded vs realtime, and where calibration comes from**.

| `type` | Recorded / Realtime | Calibration from | Typical datasets / hardware |
|---|---|---|---|
| `image_folder` | recorded | explicit `[rig]` | KITTI, TartanGround, Oxford RobotCar, **any** image-per-file dataset; optional `[input.imu]` enables Inertial |
| `euroc`        | recorded | dataset `sensor.yaml` (+IMU) | EuRoC MAV (Machine Hall, Vicon Room) |
| `tum`          | recorded | rig yaml | TUM RGB-D |
| `edex`         | recorded | `.edex` JSON | TartanGround, R2B Galileo (rosbag→edex) |
| `video`        | **both** | explicit `[rig]` | recorded `.mp4`/`.avi`; live webcam (`"0"`); IP/RTSP stream |
| `realsense`    | realtime | device | Intel RealSense stereo (live) |

Two of these are deliberately *generic* and cover "any dataset you throw at it":

- **`image_folder`** — N camera folders matched by sorted filename order, with
  optional per-camera `depth` and `mask` streams, optional Bayer demosaic, and
  flexible timestamping (from a file, from filenames, from FPS, or frame index).
  Add an `[input.imu]` CSV and it drives **Inertial** mode too.
- **`video`** — anything `cv2.VideoCapture` opens: a file (recorded), a device
  index like `"0"` (live), or an `rtsp://`/`http://` URL (live stream). A packed
  stereo frame can be split side-by-side (`split = "sbs"`) or top/bottom
  (`split = "tb"`) into two camera images.

---

## The TOML reference

A configuration has up to seven tables. Only `[input]` is mandatory; `[rig]` is
required unless the source supplies calibration. Unknown keys are rejected.

### `[run]`

Runtime behaviour (maps to library globals + loop control).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `verbosity` | int | `0` | `cuvslam.set_verbosity` (0 silent, 1 error, 2 warn, 3 info) |
| `warm_up_gpu` | bool | `false` | call `cuvslam.warm_up_gpu()` before tracking (reduces first-frame latency) |
| `start_index` | int | `0` | skip the first N **frame** events |
| `max_frames` | int | `0` | stop after N tracked frames (0 = all) |
| `sleep_ms` | float | `0` | sleep between frames; helps the async SLAM thread keep up |

### `[input]`

`type` selects the source; the remaining keys are source-specific.

**`image_folder`**
```toml
[input]
type = "image_folder"
root = "dataset/sequences/06"      # optional path prefix for every glob below
depth_scale_to_uint16 = 1.0        # multiply depth pixels before casting to uint16

  [[input.cameras]]                # one table per camera, in rig order
  images = "image_0/*.png"         # REQUIRED glob; sorted order == frame order
  depth  = "depth_0/*.png"         # optional uint16 depth (enables `depths`)
  mask   = "mask_0/*.png"          # optional dynamic masks (uint8)
  bayer  = "GBRG"                  # optional: demosaic raw Bayer -> BGR (needs opencv)
  bgr    = true                    # RGB->BGR on load (default true; matches examples)

  [input.timestamps]
  mode = "file"                    # index | fps | file | filename
  path = "times.txt"               # for mode=file (one value per line)
  unit = "s"                       # s | ms | us | ns  (file/filename)
  # fps = 30                       # for mode=fps

  [input.imu]                      # OPTIONAL -> enables odometry_mode="Inertial"
  path = "imu0/data.csv"
  format = "euroc"                 # euroc | generic
  # generic only:
  #   columns = ["timestamp","gx","gy","gz","ax","ay","az"]
  #   timestamp_unit = "ns"        # s | ms | us | ns
  #   delimiter = ","
  #   skip_header = true
  #   angular_unit = "rad"         # rad | deg
```

**`euroc`** — builds the stereo rig + IMU from `mav0/cam0|cam1|imu0/sensor.yaml`.
```toml
[input]
type = "euroc"
path = "datasets/euroc/.../V1_01_easy/mav0"
use_imu = true                     # interleave IMU samples (for Inertial); false = vision only
```

**`tum`** — associates `rgb.txt`/`depth.txt` by timestamp; reads intrinsics/scale from the rig yaml.
```toml
[input]
type = "tum"
path = "datasets/rgbd_dataset_freiburg3_long_office_household"
rig_yaml = "freiburg3_rig.yaml"    # optional; supplies a 1-camera rig + depth scale
max_time_diff = 0.02               # s, rgb/depth association window
max_gap = 0.5                      # s, skip across large temporal gaps
```

**`edex`** — multi-camera rig from a `.edex` JSON; images in folders or a jsonl manifest.
```toml
[input]
type = "edex"
edex = "tartan_ground.edex"
layout = "folders"                 # folders | jsonl
data_root = "dataset/.../P2000"
camera_names = ["lcam_front", "rcam_front", ...]   # folders layout
image_pattern = "{frame:06d}_{name}.png"           # folders layout
```

**`video`** — recorded files or live devices/streams via OpenCV.
```toml
[input]
type = "video"
  [[input.cameras]]
  source = "0"                     # device index | file path | rtsp://… | http://…
  split = "sbs"                    # none | sbs | tb  (split a packed stereo frame)
  grayscale = true                 # convert to mono8 (else BGR passthrough)
  [input.timing]
  mode = "auto"                    # auto | wallclock | fps | index
  fps = 30                         # for mode=fps (and auto on files lacking metadata)
  max_frames = 0                   # stop after N (0 = until end/stopped)
```

**`realsense`** — live Intel RealSense stereo (rig read from the device).
```toml
[input]
type = "realsense"
width = 640
height = 360
fps = 30
warmup_frames = 60
disable_emitter = true             # the IR dot pattern hurts feature tracking
```

### `[rig]` — cameras and IMU

Explicit calibration via `[[rig.cameras]]` and an optional `[rig.imu]`.
**Required** when the source can't provide it
(`image_folder`, `video`); **optional override** otherwise.

A camera (the `[rig.cameras.…]` sub-tables attach to the most recent `[[rig.cameras]]`):
```toml
[[rig.cameras]]
size = [1241, 376]                 # [width, height] in pixels
focal = [718.856, 718.856]         # [fx, fy]
principal = [607.19, 185.22]       # [cx, cy]
  [rig.cameras.rig_from_camera]    # extrinsics: camera frame -> rig frame
  rotation = [0, 0, 0, 1]          # quaternion [x, y, z, w]
  translation = [0.537, 0, 0]      # metres
  [rig.cameras.distortion]
  model = "Pinhole"                # Pinhole(0) | Fisheye(4) | Brown(5) | Polynomial(8)
  parameters = []
# static feature masks (pixels), optional:
# border_top = 0 ; border_bottom = 0 ; border_left = 0 ; border_right = 0
```

An IMU (required for `odometry_mode = "Inertial"`):
```toml
[rig.imu]
gyroscope_noise_density = 1.6968e-04      # rad/(s·√Hz)
accelerometer_noise_density = 2.0e-03     # m/(s²·√Hz)
gyroscope_random_walk = 1.9393e-05        # rad/(s²·√Hz)
accelerometer_random_walk = 3.0e-03       # m/(s³·√Hz)
frequency = 200.0                         # Hz
  [rig.imu.rig_from_imu]
  rotation = [0, 0, 0, 1]
  translation = [0, 0, 0]
```

### `[odometry]`

Maps to `cuvslam.Tracker.OdometryConfig`. Optional knobs left unset keep the
library default.

| Key | Type | Values / note |
|---|---|---|
| `odometry_mode` | str | `Multicamera` \| `Inertial` \| `RGBD` \| `Mono` |
| `multicam_mode` | str | `Performance` \| `Moderate` \| `Precision` |
| `use_gpu` | bool | GPU-accelerated feature work |
| `async_sba` | bool | run bundle adjustment off the track() thread |
| `use_motion_model` | bool | constant-velocity pose prediction |
| `use_denoising` | bool | denoise input images |
| `rectified_stereo_camera` | bool | set true only for rectified, row-aligned stereo |
| `enable_observations_export` | bool | needed for viz / SLAM (default true) |
| `enable_landmarks_export` | bool | needed for viz / SLAM (default true) |
| `enable_final_landmarks_export` | bool | export final landmarks at end |
| `max_frame_delta_s` | float | max time gap before a track reset |
| `[odometry.rgbd]` | table | `depth_scale_factor`, `depth_camera_id`, `enable_depth_stereo_tracking` |

For `tum`, `depth_scale_factor` is auto-filled from the rig yaml unless you set
`[odometry.rgbd]` explicitly.

### `[slam]`

Presence of this table enables the SLAM layer (loop closure + pose graph) on top
of odometry. Maps to `cuvslam.Tracker.SlamConfig`.

| Key | Type | Meaning |
|---|---|---|
| `enabled` | bool | default true when the table exists; set false to disable |
| `map_cache_path` | str | persist the map to LMDB at this path (else in-memory) |
| `use_gpu` | bool | GPU acceleration inside SLAM |
| `sync_mode` | bool | run SLAM on the track() thread (deterministic, slow) vs async |
| `enable_reading_internals` | bool | allow reading pose graph / landmarks |
| `planar_constraints` | bool | constrain motion to a horizontal plane |
| `gt_align_mode` | bool | special ground-truth-aligned mapping mode |
| `map_cell_size` | float | map cell size (0 = auto from baseline) |
| `max_landmarks_distance` | float | drop landmarks beyond this distance |
| `max_map_size` | int | max pose-graph nodes (0 = unlimited) |
| `throttling_time_ms` | int | min time between loop-closure events |
| `[slam.localize]` | table | localize in an existing map before tracking (below) |

```toml
[slam.localize]
map_path = "out/kitti_06_map"
horizontal_search_radius = 8.0
vertical_search_radius = 2.0
horizontal_step = 0.5
vertical_step = 0.2
angular_step_rads = 0.03
  [slam.localize.guess]            # initial pose guess
  rotation = [0, 0, 0, 1]
  translation = [0, 0, 0]
```

### `[output]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `trajectory` | str | `""` | TUM-format output path (`""` = don't write) |
| `pose_source` | str | `auto` | `auto` \| `odometry` \| `slam` (auto = slam if SLAM on) |
| `timestamp_unit` | str | `s` | `s` \| `ms` \| `us` \| `ns` for the timestamp column |
| `save_map` | str | `""` | folder to save the SLAM map (SLAM only) |
| `visualize` | bool | `false` | live Rerun visualization if `rerun-sdk` is present |
| `print_every` | int | `50` | progress print cadence (0 = silent) |

### `[eval]`

Optional. When present and enabled, the runner evaluates the produced trajectory
against ground truth and folds `ate_rmse_m` / `avg_rte_pct` / `avg_re_deg` into
the run summary.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | bool | true if table present | turn evaluation on/off |
| `ground_truth` | str | — | GT path (resolved against the input root if relative) |
| `gt_format` | str | `euroc` | `euroc` \| `tum` \| `kitti` |
| `gt_time_unit` | str | `s` | timestamp unit for `tum` GT |
| `gt_fps` | float | `10` | index→time rate for `kitti` GT |
| `align` | str | `auto` | `auto` \| `se3` \| `sim3` \| `none` (auto = sim3 for Mono, else se3) |
| `apply_gt_extrinsic` | str | `auto` | `auto` \| `euroc_cam0` \| `none` (move EuRoC body-frame GT into the cam0 frame) |
| `max_time_diff` | float | `0.02` | GT/estimate association window (s) |
| `rpe_distances` | list/str | auto | metres list, or `"kitti"` for 100…800 m |
| `report` | str | `""` | optional path to write the text report |

---

## Choosing a mode

| `odometry_mode` | Sensors | Scale | Use when |
|---|---|---|---|
| `Multicamera` | ≥1 synchronized cameras (stereo, or N stereo pairs) | metric | default; robust outdoor/indoor, occlusion tolerance with more cameras |
| `Inertial` | stereo + IMU | metric | aggressive motion / motion blur; needs good IMU noise params |
| `RGBD` | 1 camera + aligned depth | metric | indoor with a depth sensor; `enable_depth_stereo_tracking` to add the stereo partner |
| `Mono` | 1 camera | up-to-scale | quick "does it track?" checks; rotation accurate, translation needs Sim3 scale alignment to evaluate |

`MulticameraMode`: `Performance` (features from the primary camera only — fast),
`Moderate`, `Precision` (all cameras — most accurate).

---

## Evaluation metrics explained

`eval.py` associates each estimated pose to the nearest ground-truth pose by
timestamp, aligns the trajectories, and reports:

| Metric | Definition | Unit |
|---|---|---|
| **ATE / RMSE APE** | RMS of per-frame position error after global alignment | m (printed also in cm) |
| **avgRTE** | mean relative **translation** error over fixed travelled-distance segments | % of distance |
| **RPE rotation** | mean relative **rotation** error per travelled metre | deg/m |
| **avgRE** | mean relative rotation error per segment (not distance-normalized) | deg |

Plus APE mean/median/std/min/max and a per-segment-length breakdown.

**How it's computed**

1. **Association** — nearest GT timestamp within `max_time_diff` (default 20 ms).
2. **Alignment** — Umeyama (1991) least-squares fit of estimate→GT positions.
   `se3` is rigid (rotation+translation); `sim3` additionally solves a global
   scale (use for monocular, which is scale-free). `none` skips alignment.
3. **ATE** — Euclidean position error of every aligned pose; report the RMS.
4. **RPE (KITTI-style)** — for each segment length `L`, slide a window over the
   trajectory, take the relative motion `Pᵢ⁻¹Pⱼ` over a sub-path of length `L`
   in both GT and estimate, and measure the residual
   `E = (relGT)⁻¹(relEST)`. Translation error is `‖trans(E)‖ / L` (→ %) and
   rotation error is `angle(rot(E)) / L` (→ deg/m). Averaging over all windows
   and lengths gives **avgRTE %** and **RPE deg/m**; the un-normalized mean of
   `angle(rot(E))` gives **avgRE deg**.

**Frame note (EuRoC):** ground truth is in the IMU/body frame while cuVSLAM
outputs the cam0 frame. With `apply_gt_extrinsic = "euroc_cam0"` (the default for
EuRoC) the GT is right-multiplied by cam0's `T_BS` so both share the cam0 frame —
important for a correct rotation RPE. ATE after alignment is robust either way.

**Ground-truth formats**
- `euroc`: `state_groundtruth_estimate0/data.csv` (`ts[ns], px,py,pz, qw,qx,qy,qz, …`).
- `tum`: `ts tx ty tz qx qy qz qw` (set `gt_time_unit`).
- `kitti`: 12 numbers/line (3×4 row-major); index-timed at `gt_fps`. Note KITTI
  GT poses are a **separate download** from the odometry images.

---

## Validated benchmark results

Run against the `cuvslam-15.0.0+cu13` wheel on **EuRoC V1_01_easy** (Vicon Room),
stereo (`Multicamera`), 2912 frames, 0 tracking failures, GT moved into the cam0
frame, SE3 alignment (`configs/euroc_v1_eval.toml`):

```
ATE / RMSE APE : 7.78 cm        (0.0778 m)
APE mean/median: 0.0704 / 0.0674 m
avgRTE         : 5.23 %
RPE rotation   : 0.67 deg/m
avgRE          : 2.34 deg
  len(m)  trans%   rot(deg/m)
    1.0   8.147    1.0357
    2.0   6.890    0.8983
    4.0   5.546    0.7032
    8.0   3.193    0.4277
   16.0   1.412    0.1784
```

RPE shrinking with segment length is the expected signature (short segments are
dominated by per-frame noise; long segments by genuine drift). Stereo-inertial on
the same sequence also runs cleanly; tune the IMU noise params for best results
(the default EuRoC `sensor.yaml` values are known to be loose — the repo's euroc
example ships recalibrated yamls).

---

## Dashboard & results site

Everything the repo produces has a visual counterpart:

```bash
python3 viz/make_figures.py     # PNG figures from every committed CSV/TSV artifact
python3 viz/build_site.py      # assembles them into site/index.html (static, browsable)
python3 dashboard/serve.py     # web dashboard on http://127.0.0.1:8642/
```

The dashboard registers a **new dataset** (pick a preset — the template is a
validated accuracy-matrix config — set paths, optionally override intrinsics),
emits **TOML feature variants** (SLAM / odom-only / async / CPU / denoising / …,
reusing `gen_profiling_coverage`'s transforms) into `configs/custom/`, **runs**
any config plain or under nsys/ncu with a live log tail, and embeds the results
site. `viz/make_figures.py --only trajectories --traj-root /mnt/data/accuracy_out`
additionally draws estimated-vs-ground-truth trajectory grids where the run
outputs live (the workstation). Requires `matplotlib` (dashboard itself is
stdlib-only).

## Bundled configs

| Config | Source | Mode | Notes |
|---|---|---|---|
| `kitti_stereo.toml` | image_folder | Multicamera | repo KITTI example; explicit rig |
| `kitti_slam.toml` | image_folder | Multicamera + SLAM | loop closure + `save_map` |
| `euroc_inertial.toml` | euroc | Inertial | rig+IMU from dataset yaml |
| `euroc_mono.toml` | euroc | Mono | vision-only sanity check |
| `euroc_v1_eval.toml` | euroc | Multicamera + **eval** | the validated ATE/RPE benchmark |
| `tum_rgbd.toml` | tum | RGBD | depth scale auto from rig yaml; border masks |
| `tartan_multicam.toml` | edex | Multicamera | 12-camera (6 stereo) rig |
| `robotcar_mono.toml` | image_folder | Mono | Bayer demosaic + hood mask (arbitrary dataset) |
| `generic_inertial.toml` | image_folder + `[input.imu]` | Inertial | **any** folder+IMU dataset → inertial |
| `usb_stereo_video.toml` | video | Multicamera | USB stereo, side-by-side split |
| `webcam_mono.toml` | video | Mono | live webcam |
| `realsense_stereo.toml` | realsense | Multicamera | live RealSense |
| `kitti_eval.toml` | image_folder | Multicamera + **eval** | paper: KITTI seq, 100–800 m segment RPE |
| `iclnuim_rgbd.toml` | tum | RGBD | paper: ICL-NUIM (+ TUM-format eval) |
| `artable_rgbd.toml` | image_folder | RGBD | paper: AR-table (template) |
| `tumvi_room_inertial.toml` | image_folder + `[input.imu]` | Inertial | paper: TUM-VI room (fisheye + mocap eval) |
| `tartanair_v2_multicam.toml` | edex | Multicamera | paper: TartanAir V2 (12-cam) |
| `r2b_multicam.toml` | edex (jsonl) | Multicamera | paper: R2B (4 stereo) |

## Paper benchmark coverage

Every dataset benchmarked in the cuVSLAM technical report (arXiv:2506.04359,
Tables 2 & 3) has a ready config — see **[PAPER_DATASETS.md](PAPER_DATASETS.md)**
for the full dataset → config → source → eval mapping. The runner reports the
paper's own metrics (`avgRTE %`, `avgRE deg`, `RMSE APE`). EuRoC and KITTI are
validated here against the cu13 wheel; the rest ship schema-valid configs with
published calibration to fill in for your specific download.

---

## Running all configs

`run_list.py` (no arguments) executes — or with `--check` validates — every `configs/*.toml` in turn; give it a list file or `--configs DIR` to select others. Each config
runs in its **own subprocess**, so a missing dataset, an absent camera, or a
crash in one never stops the rest. At the end it prints a summary table and
writes a per-config log; the exit code is 0 only if every selected config
succeeded.

```bash
python run_list.py                          # run every config in ./configs
python run_list.py --check                  # validate only (no cuvslam, no tracking)
python run_list.py --configs configs/accuracy_matrix   # explicit directory or glob
python run_list.py --only euroc_v1_eval,kitti_stereo
python run_list.py --skip realsense_stereo,webcam_mono
python run_list.py --timeout 600            # per-config seconds (0 = no limit)
python run_list.py --python /path/to/venv/bin/python   # interpreter for each run
```

It parses each run's `[runner] done: {…}` line, so the table includes the frame
count and — when `[eval]` is set — the ATE and avgRTE:

```
================================================================================
SUMMARY
================================================================================
config                       status    frames    ATE(m)    RTE%  time(s)  note
euroc_v1_inertial            OK          2912    0.1688    5.58     99.0
euroc_v1_stereo              OK          2912    0.0778    5.23    121.4
--------------------------------------------------------------------------------
2/2 succeeded
```

Each config's exact command is echoed (`$ python run.py …`) and its output —
including any Python traceback — is **streamed live** (prefixed with `| `), so
nothing is hidden. Pass `--quiet` to suppress the live stream (the full output
is still printed on failure and always saved under `out/run_list/<name>.log`).
`run_list.py` behaves identically and writes to `out/run_list/`.

## Recipes

**Bring your own recorded dataset (stereo, with timestamps file)**
```toml
[input]
type = "image_folder"
root = "/data/my_seq"
  [[input.cameras]]
  images = "left/*.png"
  [[input.cameras]]
  images = "right/*.png"
  [input.timestamps]
  mode = "file"
  path = "timestamps.txt"
  unit = "ns"
# + an explicit [rig] with your calibration, + [odometry] odometry_mode="Multicamera"
```

**Turn any recorded folder dataset into a stereo-inertial run** — add `[input.imu]`
(see `generic_inertial.toml`) and set `odometry_mode = "Inertial"` plus `[rig.imu]`.

**Live monocular from a webcam** — `configs/webcam_mono.toml` (`type="video"`,
`source="0"`, `mode="wallclock"`).

**Evaluate a trajectory you already have**
```bash
python evaluate.py est_tum.txt /data/seq/mav0/state_groundtruth_estimate0/data.csv \
  --gt-format euroc --euroc-cam0-yaml /data/seq/mav0/cam0/sensor.yaml \
  --align se3 --rpe-distances 1,2,4,8,16
# KITTI:
python evaluate.py est_tum.txt poses/06.txt --gt-format kitti --align se3 --rpe-distances kitti
```

---

## Coordinate conventions

- cuVSLAM uses the OpenCV camera frame: **x right, y down, z forward**.
- Poses are `world_from_rig`; the rig frame is defined by your extrinsics
  (typically the first/primary camera is the rig origin).
- Quaternions are `[x, y, z, w]` everywhere in the TOML and outputs.
- Trajectory files are TUM format: `timestamp tx ty tz qx qy qz qw`.
- 3-channel images are converted RGB→BGR on load by default (`bgr = true`),
  matching the cuVSLAM example pipeline; depth is `uint16`.

---

## Extending the system

Add a new input source in three steps:

1. Create `cuvslam_runner/sources/my_source.py` with a class subclassing
   `FrameSource`. Implement `__iter__` (yield `FrameEvent`/`ImuEvent` in ascending
   timestamp order), set `self.num_cameras`, and optionally `build_rig_spec()` and
   `__len__`.
2. Register it in `sources/__init__.py`'s `_REGISTRY`
   (`"my_type": ("cuvslam_runner.sources.my_source", "MySource")`). It's imported
   lazily, so optional third-party deps only load when that type is requested.
3. Write a config with `type = "my_type"`. Nothing in `builders.py`/`runner.py`
   needs to change.

Sources emit plain dataclasses (`specs.py`), never `cuvslam` objects — keep the
parsing layer cuVSLAM-free so it stays testable with `--check`.

---

## Testing

```bash
python tests/test_smoke.py          # or: python -m pytest tests/
```

The smoke tests run **without `cuvslam`**: config parse + key validation,
`image_folder` enumeration, timestamp modes, IMU merge ordering, optional video
source guard, image helpers, and the evaluation math (which is checked to be
exact under rigid/Sim3 transforms and sane under injected drift).

Validate any config offline with:
```bash
python run.py configs/<name>.toml --check
```
which parses the TOML, wires the source, resolves the rig, and reports the frame
count — all without importing `cuvslam`.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `No TOML parser available` | Python < 3.11 without tomli → `pip install tomli` |
| `type='video' needs opencv-python` | `pip install opencv-python` |
| `type='realsense' needs pyrealsense2` | install pyrealsense2 + connect a camera |
| `Bayer demosaicing requires opencv-python` | install opencv, or drop `bayer=` |
| `… does not provide calibration; define an explicit [rig]` | add `[[rig.cameras]]` (image_folder/video can't infer intrinsics) |
| `Only N GT/estimate matches …` | wrong `gt_format`/units, or widen `max_time_diff` |
| Many "Failed to track frame" | check sync/timestamps, calibration, `rectified_stereo_camera`, frame rate; see the repo's main Performance notes |
| RPE rotation looks inflated on EuRoC | ensure `apply_gt_extrinsic` moved GT to the cam0 frame |
| SLAM poses "jump" | expected at loop closures; use async SLAM and read smoothed poses, or `sync_mode` for determinism |

---

*This harness is a thin wrapper; the underlying tracking, accuracy, and platform
guidance live in the main cuVSLAM repository README and technical report
(arXiv:2506.04359).*
