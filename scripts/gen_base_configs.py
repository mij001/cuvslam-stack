#!/usr/bin/env python3
"""gen_accuracy_configs.py — generate the accuracy-matrix configs.

The profiling campaign answered "how does cuVSLAM use memory"; this matrix
answers "does our cuVSLAM produce the accuracy the paper (arXiv:2506.04359)
claims, across every feature combination the on-disk datasets support":

  KITTI 00-10   color stereo   : odom | slam | slam_async
  EuRoC  x11    stereo         : odom | slam        (paper: V2_03 excluded
                inertial (IMU) : odom | slam         from pure-stereo avg)
                mono           : odom  (Sim3 alignment; up-to-scale)
  TUM fr3 x4    RGB-D          : odom | slam | slam_cpu ([slam].use_gpu=false)
  TUM-VI  x2    stereo-inertial: odom | slam        (corridor1/magistrale1 —
                                                     GT-eval only; the paper
                                                     used room1-6)

Every config carries an [eval] block (ATE/avgRTE/avgRE vs ground truth, the
paper's own metrics via cuvslam_runner.eval) and writes trajectory + report
into OUT_ROOT/<run_name>/. Run on the host that mounts the data (workstation):

    python3 gen_accuracy_configs.py --root /mnt/data --out configs/accuracy_matrix
"""
from __future__ import annotations

import argparse
import glob
import os

# sequence lists + intrinsics + calibration readers live in ONE place
from dataset_catalog import (  # noqa: F401  (TARTANAIR used by the tartanair pass)
    EUROC, FR3, ICL, ICL_NUIM, KITTI_SEQS, TARTANAIR, TUM_RGBD,
    kitti_calib, png_size,
)

OUT_ROOT = "/mnt/data/accuracy_out"

# [eval] extras per dataset family — hoisted so no f-string expression needs a
# backslash (SyntaxError before Python 3.12; the venv is 3.10).
EVAL_EXTRA_VI = ('apply_gt_extrinsic = "auto"\nrpe_distances = [8, 16, 32]\n'
                 'rpe_delta = 1\nrpe_delta_unit = "s"')
EVAL_EXTRA_TUM = ('gt_time_unit = "s"\nrpe_distances = [1, 2, 4]\n'
                  'rpe_delta = 1\nrpe_delta_unit = "s"')
EVAL_EXTRA_TARTAN = ('gt_time_unit = "s"\nrpe_distances = [8, 16, 32]\n'
                     'rpe_delta = 1\nrpe_delta_unit = "s"')


def slam_block(kind):
    if kind == "odom":
        return "", ""
    sync = "false" if kind == "slam_async" else "true"
    gpu = "false" if kind == "slam_cpu" else "true"
    async_sba = "true" if kind == "slam_async" else "false"
    block = f"""
[slam]
enabled = true
use_gpu = {gpu}
sync_mode = {sync}
enable_reading_internals = true
"""
    return block, async_sba


def out_paths(name):
    d = f"{OUT_ROOT}/{name}"
    return (f"""[output]
trajectory = "{d}/traj_tum.txt"
pose_source = "{'slam' if '_slam' in name else 'odometry'}"
visualize = false
print_every = 0""", d)


def eval_block(gt, fmt, report, align="se3", extra=""):
    return f"""[eval]
ground_truth = "{gt}"
gt_format = "{fmt}"
align = "{align}"
report = "{report}"
{extra}"""


def kitti_cfg(root, seq, kind):
    name = f"kitti{seq}_stereo_{kind}"
    sdir = f"{root}/dataset/sequences/{seq}"
    f, cx, cy, base = kitti_calib(f"{sdir}/calib.txt")
    w, h = png_size(sorted(glob.glob(f"{sdir}/image_2/*.png"))[0])
    slam, async_sba = slam_block(kind)
    out, d = out_paths(name)
    gt = f"{root}/KITTI/datasets/poses/{seq}.txt"
    kitti_eval_extra = 'gt_fps = 10.0\nrpe_distances = "kitti"'   # no backslash in f-expr (py<3.12)
    return name, f"""# accuracy matrix: KITTI {seq} color stereo {kind}
[run]
verbosity = 0
max_frames = 0

[input]
type = "image_folder"
root = "{sdir}"

  [[input.cameras]]
  images = "image_2/*.png"

  [[input.cameras]]
  images = "image_3/*.png"

  [input.timestamps]
  mode = "file"
  path = "times.txt"
  unit = "s"

[odometry]
odometry_mode = "Multicamera"
multicam_mode = "Performance"
rectified_stereo_camera = true
async_sba = {async_sba or 'false'}
{slam}
[[rig.cameras]]
size = [{w}, {h}]
focal = [{f}, {f}]
principal = [{cx}, {cy}]

[[rig.cameras]]
size = [{w}, {h}]
focal = [{f}, {f}]
principal = [{cx}, {cy}]
  [rig.cameras.rig_from_camera]
  translation = [{base:.6f}, 0.0, 0.0]

{out}
timestamp_unit = "s"

{eval_block(gt, "kitti", f"{d}/eval.txt", extra=kitti_eval_extra)}
"""


def euroc_cfg(root, rel, seq, variant, kind):
    name = f"euroc_{seq}_{variant}_{kind}"
    mav0 = f"{root}/EuRoC/{rel}/{seq}/mav0"
    mode = {"stereo": "Multicamera", "inertial": "Inertial", "mono": "Mono"}[variant]
    use_imu = "true" if variant == "inertial" else "false"
    align = "sim3" if variant == "mono" else "se3"
    slam, async_sba = slam_block(kind)
    out, d = out_paths(name)
    return name, f"""# accuracy matrix: EuRoC {seq} {variant} {kind}
[run]
verbosity = 0
max_frames = 0

[input]
type = "euroc"
path = "{mav0}"
use_imu = {use_imu}

[odometry]
odometry_mode = "{mode}"
rectified_stereo_camera = false
async_sba = {async_sba or 'false'}
{slam}
{out}
timestamp_unit = "ns"

{eval_block(f"{mav0}/state_groundtruth_estimate0/data.csv", "euroc",
            f"{d}/eval.txt", align=align,
            extra=EVAL_EXTRA_VI)}
"""


def tum_cfg(root, seq, kind):
    short = seq.replace("rgbd_dataset_freiburg3_", "fr3_")
    name = f"tum_{short}_rgbd_{kind}"
    path = f"{root}/TUM_RGBD/extracted/{seq}"
    slam, async_sba = slam_block(kind)
    out, d = out_paths(name)
    return name, f"""# accuracy matrix: TUM {short} RGBD {kind}
[run]
verbosity = 0
max_frames = 0

[input]
type = "tum"
path = "{path}"
max_time_diff = 0.02
max_gap = 0.5

[odometry]
odometry_mode = "RGBD"
async_sba = {async_sba or 'false'}

  [odometry.rgbd]
  depth_scale_factor = 5000.0
  depth_camera_id = 0
{slam}
[[rig.cameras]]
size = [640, 480]
focal = {FR3['focal']}
principal = {FR3['principal']}
border_top = 20
border_bottom = 20
border_left = 10
border_right = 50

{out}
timestamp_unit = "s"

{eval_block(f"{path}/groundtruth.txt", "tum", f"{d}/eval.txt",
            extra=EVAL_EXTRA_TUM)}
"""


def icl_cfg(root, seq, kind):
    """ICL-NUIM (Mono-Depth). TUM-format release → tum source + tum gt_format."""
    name = f"icl_{seq.replace('_frei_png','')}_rgbd_{kind}"
    path = f"{root}/ICL-NUIM/{seq}"
    slam, async_sba = slam_block(kind)
    out, d = out_paths(name)
    return name, f"""# accuracy matrix: ICL-NUIM {seq} RGBD {kind}  (paper Mono-Depth)
[run]
verbosity = 0
max_frames = 0

[input]
type = "tum"
path = "{path}"
max_time_diff = 0.5
max_gap = 2.0                        # ICL timestamps are frame indices (dt=1.0)

[odometry]
odometry_mode = "RGBD"
async_sba = {async_sba or 'false'}

  [odometry.rgbd]
  depth_scale_factor = 5000.0
  depth_camera_id = 0
{slam}
[[rig.cameras]]
size = [640, 480]
focal = {ICL['focal']}
principal = {ICL['principal']}

{out}
timestamp_unit = "s"

{eval_block(f"{path}/groundtruth.txt", "tum", f"{d}/eval.txt",
            extra=EVAL_EXTRA_TUM)}
"""


def tartanair_cfg(root, env, diff, pxx, kind):
    """TartanAir stereo. image_folder source; GT pre-converted to TUM format
    (groundtruth_tum.txt) and a synthetic times.txt written by the prep script,
    so no runner change is needed. Pinhole 640x480, baseline 0.25 m."""
    name = f"tartan_{env}_{diff}_{pxx}_stereo_{kind}"
    sdir = f"{root}/tartanair/{env}/{diff}/{pxx}"
    f = TARTANAIR["focal"][0]
    cx, cy = TARTANAIR["principal"]
    base = TARTANAIR["baseline"]
    slam, async_sba = slam_block(kind)
    out, d = out_paths(name)
    return name, f"""# accuracy matrix: TartanAir {env}/{diff}/{pxx} stereo {kind}
[run]
verbosity = 0
max_frames = 0

[input]
type = "image_folder"
root = "{sdir}"

  [[input.cameras]]
  images = "image_left/*.png"

  [[input.cameras]]
  images = "image_right/*.png"

  [input.timestamps]
  mode = "file"
  path = "times.txt"
  unit = "s"

[odometry]
odometry_mode = "Multicamera"
multicam_mode = "Performance"
rectified_stereo_camera = true
async_sba = {async_sba or 'false'}
{slam}
[[rig.cameras]]
size = [640, 480]
focal = [{f}, {f}]
principal = [{cx}, {cy}]

[[rig.cameras]]
size = [640, 480]
focal = [{f}, {f}]
principal = [{cx}, {cy}]
  [rig.cameras.rig_from_camera]
  translation = [{base:.6f}, 0.0, 0.0]

{out}
timestamp_unit = "s"

{eval_block(f"{sdir}/groundtruth_tum.txt", "tum", f"{d}/eval.txt",
            align="se3", extra=EVAL_EXTRA_TARTAN)}
"""


def tumvi_cfg(mav0, name_stub, kind):
    name = f"tumvi_{name_stub}_inertial_{kind}"
    slam, async_sba = slam_block(kind)
    out, d = out_paths(name)
    return name, f"""# accuracy matrix: TUM-VI {name_stub} stereo-inertial {kind}
[run]
verbosity = 0
max_frames = 0

[input]
type = "euroc"
path = "{mav0}"
use_imu = true

[odometry]
odometry_mode = "Inertial"
rectified_stereo_camera = false
async_sba = {async_sba or 'false'}
{slam}
{out}
timestamp_unit = "ns"

{eval_block(f"{mav0}/mocap0/data.csv", "euroc", f"{d}/eval.txt",
            extra=EVAL_EXTRA_VI)}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/mnt/data")
    ap.add_argument("--tumvi-extracted", default=os.path.expanduser("~/tumvi_extracted"))
    ap.add_argument("--tartanair-v1-extra", action="store_true",
                    help="also emit TartanAir V1 single-stereo configs (NOT a "
                         "paper benchmark; the paper uses V2 multi-stereo)")
    ap.add_argument("--out", default="configs/base")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    n = 0

    def emit(name, body):
        nonlocal n
        with open(os.path.join(args.out, name + ".toml"), "w") as fh:
            fh.write(body)
        n += 1

    # BASES ONLY: one full-featured SLAM config per sequence × modality (odom
    # where SLAM does not apply, i.e. mono). Every other variant — odom,
    # slam_async, slam_cpu, feature toggles, frame windows — is derived by
    # scripts/mutate_configs.py; nothing else generates configs.
    for seq in KITTI_SEQS:
        if not os.path.isfile(f"{args.root}/dataset/sequences/{seq}/calib.txt"):
            print(f"[skip] KITTI {seq}")
            continue
        emit(*kitti_cfg(args.root, seq, "slam"))

    for rel, seq in EUROC:
        if not os.path.isdir(f"{args.root}/EuRoC/{rel}/{seq}/mav0"):
            print(f"[skip] EuRoC {seq}")
            continue
        for variant, kind in (("stereo", "slam"), ("inertial", "slam"), ("mono", "odom")):
            emit(*euroc_cfg(args.root, rel, seq, variant, kind))

    for seq in TUM_RGBD:
        if not os.path.isdir(f"{args.root}/TUM_RGBD/extracted/{seq}"):
            print(f"[skip] TUM {seq}")
            continue
        emit(*tum_cfg(args.root, seq, "slam"))

    # ICL-NUIM (paper Mono-Depth) — TUM-format RGB-D
    for seq in ICL_NUIM:
        if not os.path.isdir(f"{args.root}/ICL-NUIM/{seq}"):
            print(f"[skip] ICL-NUIM {seq}")
            continue
        emit(*icl_cfg(args.root, seq, "slam"))

    # TartanAir — NOTE: the paper (2506.04359v3 Table 3) evaluates TartanAir
    # **V2, Hard, in MULTI-STEREO (multi-camera) mode**, NOT V1 single-stereo.
    # The V1 single-stereo configs below are a valid extra cuVSLAM workload but
    # are NOT the paper's benchmark, so they are OFF by default. The paper's
    # TartanAir (V2 multi-cam) + TartanGround need multi-camera rig configs — a
    # separate builder (tartanair_v2_cfg) once that dataset is on disk.
    if args.tartanair_v1_extra:
        for pxx_dir in sorted(glob.glob(f"{args.root}/tartanair/*/Hard/P0*") +
                              glob.glob(f"{args.root}/tartanair/*/Easy/P0*")):
            if not os.path.isdir(os.path.join(pxx_dir, "image_left")):
                continue
            if not os.path.isfile(os.path.join(pxx_dir, "groundtruth_tum.txt")):
                print(f"[skip] TartanAir {pxx_dir} (run prep_tartanair first)")
                continue
            parts = pxx_dir.split("/")
            env, diff, pxx = parts[-3], parts[-2], parts[-1]
            emit(*tartanair_cfg(args.root, env, diff, pxx, "slam"))

    for d in sorted(glob.glob(os.path.join(args.tumvi_extracted, "dataset-*_512_16"))):
        mav0 = os.path.join(d, "mav0")
        if not os.path.isdir(mav0):
            mav0 = os.path.join(d, "dso", "..", "mav0")  # layout guard
        if not os.path.isdir(mav0):
            print(f"[skip] TUM-VI {d}")
            continue
        stub = os.path.basename(d).replace("dataset-", "").replace("_512_16", "")
        emit(*tumvi_cfg(mav0, stub, "slam"))

    print(f"[✓] {n} base configs -> {args.out}")


if __name__ == "__main__":
    main()
