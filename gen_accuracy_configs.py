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
import struct

KITTI_SEQS = [f"{i:02d}" for i in range(11)]
EUROC = [("MH/machine_hall", s) for s in
         ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]] + \
        [("VR1/vicon_room1", s) for s in ["V1_01_easy", "V1_02_medium", "V1_03_difficult"]] + \
        [("VR2/vicon_room2", s) for s in ["V2_01_easy", "V2_02_medium", "V2_03_difficult"]]
TUM_RGBD = ["rgbd_dataset_freiburg3_long_office_household",
            "rgbd_dataset_freiburg3_nostructure_notexture_far",
            "rgbd_dataset_freiburg3_nostructure_notexture_near_withloop",
            "rgbd_dataset_freiburg3_nostructure_texture_far"]
FR3 = dict(focal=[535.4, 539.2], principal=[320.1, 247.6])
OUT_ROOT = "/mnt/data/accuracy_out"


def png_size(path):
    with open(path, "rb") as fh:
        d = fh.read(26)
    return struct.unpack(">II", d[16:24])


def kitti_calib(path):
    P = {}
    for line in open(path):
        k, _, rest = line.partition(":")
        vals = rest.split()
        if len(vals) == 12:
            P[k] = [float(v) for v in vals]
    f = P["P2"][0]
    return f, P["P2"][2], P["P2"][6], (P["P2"][3] - P["P3"][3]) / f


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

{eval_block(gt, "kitti", f"{d}/eval.txt",
            extra='gt_fps = 10.0\nrpe_distances = "kitti"')}
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
            extra='apply_gt_extrinsic = "auto"\nrpe_distances = [8, 16, 32]\nrpe_delta = 1\nrpe_delta_unit = "s"')}
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
            extra='gt_time_unit = "s"\nrpe_distances = [1, 2, 4]\nrpe_delta = 1\nrpe_delta_unit = "s"')}
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
            extra='apply_gt_extrinsic = "auto"\nrpe_distances = [8, 16, 32]\nrpe_delta = 1\nrpe_delta_unit = "s"')}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/mnt/data")
    ap.add_argument("--tumvi-extracted", default=os.path.expanduser("~/tumvi_extracted"))
    ap.add_argument("--out", default="configs/accuracy_matrix")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    n = 0

    def emit(name, body):
        nonlocal n
        with open(os.path.join(args.out, name + ".toml"), "w") as fh:
            fh.write(body)
        n += 1

    for seq in KITTI_SEQS:
        if not os.path.isfile(f"{args.root}/dataset/sequences/{seq}/calib.txt"):
            print(f"[skip] KITTI {seq}")
            continue
        for kind in ("odom", "slam", "slam_async"):
            emit(*kitti_cfg(args.root, seq, kind))

    for rel, seq in EUROC:
        if not os.path.isdir(f"{args.root}/EuRoC/{rel}/{seq}/mav0"):
            print(f"[skip] EuRoC {seq}")
            continue
        for variant, kinds in (("stereo", ("odom", "slam")),
                               ("inertial", ("odom", "slam")),
                               ("mono", ("odom",))):
            for kind in kinds:
                emit(*euroc_cfg(args.root, rel, seq, variant, kind))

    for seq in TUM_RGBD:
        if not os.path.isdir(f"{args.root}/TUM_RGBD/extracted/{seq}"):
            print(f"[skip] TUM {seq}")
            continue
        for kind in ("odom", "slam", "slam_cpu"):
            emit(*tum_cfg(args.root, seq, kind))

    for d in sorted(glob.glob(os.path.join(args.tumvi_extracted, "dataset-*_512_16"))):
        mav0 = os.path.join(d, "mav0")
        if not os.path.isdir(mav0):
            mav0 = os.path.join(d, "dso", "..", "mav0")  # layout guard
        if not os.path.isdir(mav0):
            print(f"[skip] TUM-VI {d}")
            continue
        stub = os.path.basename(d).replace("dataset-", "").replace("_512_16", "")
        for kind in ("odom", "slam"):
            emit(*tumvi_cfg(mav0, stub, kind))

    print(f"[✓] {n} configs -> {args.out}")


if __name__ == "__main__":
    main()
