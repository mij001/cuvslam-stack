#!/usr/bin/env python3
"""Reproduce the paper's KITTI table: run all GT sequences (00-10), average, compare.

For each sequence we read its own calib.txt (intrinsics + stereo baseline), run
cuVSLAM in stereo Odometry and in SLAM (optimized pose graph), evaluate with the
paper's metrics, then average over sequences and print the deviation from the
report (Table 2 RMSE APE, Fig 10 segment avgRTE / deg-per-m).

    python kitti_paper_sweep.py [--modes odom,slam] [--seqs 00,06,10]

Metrics per sequence (all against the official GT poses, index-associated):
  * RMSE APE  - Sim(3) alignment (scale-corrected, as the paper states)
  * avgRTE %  - KITTI 100-800 m segment relative translation error  (Fig 10)
  * deg/m     - KITTI 100-800 m segment relative rotation error      (Fig 10)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cuvslam_runner import run as run_config              # noqa: E402
from cuvslam_runner.specs import (                        # noqa: E402
    CameraSpec, Config, EvalSpec, OdometrySpec, OutputSpec,
    PoseSpec, RigSpec, RunSpec, SlamSpec,
)

DATASET = os.environ.get(
    "KITTI_DIR", "/home/m_inomal/Projects/cuvslam_datasets/dataset")
KITTI_SEGMENTS = [100, 200, 300, 400, 500, 600, 700, 800]

# Paper reference values (Table 2 stereo KITTI + Fig 10 SLAM segment metric).
PAPER = {
    "odom": {"ate": 3.00, "rte": None,  "rot_dpm": None},     # Table 2: avgRTE 0.33%, avgRE 1.14 deg, APE 3.00
    "slam": {"ate": 1.98, "rte": 0.85,  "rot_dpm": 0.0025},   # Table 2 APE 1.98; Fig 10 0.85% / 0.0025 deg/m
}
PAPER_T2 = {"odom": {"rte": 0.33, "re": 1.14}, "slam": {"rte": 0.27, "re": 0.93}}  # "without segmentation"


def read_calib(seq_dir: str):
    """KITTI calib.txt -> (size, focal, principal, baseline) for the gray stereo pair."""
    vals = {}
    with open(os.path.join(seq_dir, "calib.txt")) as fh:
        for line in fh:
            if not line.strip():
                continue
            key, *nums = line.split()
            vals[key.rstrip(":")] = [float(x) for x in nums]
    P0, P1 = vals["P0"], vals["P1"]
    fx, fy, cx, cy = P0[0], P0[5], P0[2], P0[6]
    baseline = -P1[3] / fx                      # P1[3] = -fx * baseline
    w, h = Image.open(sorted(glob.glob(os.path.join(seq_dir, "image_0", "*.png")))[0]).size
    return [w, h], [fx, fy], [cx, cy], baseline


def make_config(seq: str, mode: str) -> Config:
    seq_dir = os.path.join(DATASET, "sequences", seq)
    size, focal, principal, baseline = read_calib(seq_dir)

    cam0 = CameraSpec(size=size, focal=focal, principal=principal)
    cam1 = CameraSpec(size=size, focal=focal, principal=principal,
                      rig_from_camera=PoseSpec(translation=[baseline, 0.0, 0.0]))
    rig = RigSpec(cameras=[cam0, cam1])

    slam_on = (mode == "slam")
    return Config(
        run=RunSpec(verbosity=0, sleep_ms=(10 if slam_on else 0)),
        input={
            "type": "image_folder", "root": seq_dir,
            "cameras": [{"images": "image_0/*.png"}, {"images": "image_1/*.png"}],
            # 10 Hz timestamps; KITTI eval is index-associated anyway.
            "timestamps": {"mode": "fps", "fps": 10.0},
        },
        odometry=OdometrySpec(odometry_mode="Multicamera", multicam_mode="Performance",
                              rectified_stereo_camera=True, async_sba=False),
        slam=SlamSpec(enabled=slam_on, sync_mode=False),
        output=OutputSpec(trajectory="", pose_source=("slam" if slam_on else "odometry"),
                          slam_pose_mode="optimized",   # paper evaluates the optimized graph
                          print_every=0),
        rig=rig,
        eval=EvalSpec(
            enabled=True,
            ground_truth=os.path.join(DATASET, "poses", f"{seq}.txt"),
            gt_format="kitti",
            align="sim3",                     # scale-corrected APE (paper methodology)
            rpe_distances=list(KITTI_SEGMENTS),  # Fig 10: 100-800 m segments
        ),
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", default="odom,slam")
    p.add_argument("--seqs", default="00,01,02,03,04,05,06,07,08,09,10")
    args = p.parse_args(argv)
    modes = [m for m in args.modes.split(",") if m]
    seqs = [s for s in args.seqs.split(",") if s]

    results = {m: {} for m in modes}
    for mode in modes:
        for seq in seqs:
            print(f"\n===== KITTI {seq}  mode={mode} =====", flush=True)
            summary = run_config(make_config(seq, mode))
            results[mode][seq] = summary
            print(f"  -> APE {summary.get('ate_rmse_m')} m | "
                  f"avgRTE {summary.get('avg_rte_pct')}% | "
                  f"{summary.get('rpe_rot_deg_per_m')} deg/m", flush=True)

    # ---- aggregate + compare ------------------------------------------------ #
    print("\n" + "=" * 78)
    print("KITTI sweep — per-sequence and average (vs paper)")
    print("=" * 78)
    for mode in modes:
        rows = results[mode]
        print(f"\n[{mode}]  seq    RMSE_APE(m)   avgRTE(%)   rot(deg/m)   avgRE(deg)")
        ate = rte = dpm = re = 0.0
        n = 0
        for seq in seqs:
            s = rows.get(seq, {})
            if "ate_rmse_m" not in s:
                print(f"        {seq}    (failed)")
                continue
            n += 1
            ate += s["ate_rmse_m"]; rte += s["avg_rte_pct"]
            dpm += s["rpe_rot_deg_per_m"]; re += s["avg_re_deg"]
            print(f"        {seq}    {s['ate_rmse_m']:9.3f}   {s['avg_rte_pct']:8.3f}   "
                  f"{s['rpe_rot_deg_per_m']:9.4f}   {s['avg_re_deg']:8.3f}")
        if n:
            ate/=n; rte/=n; dpm/=n; re/=n
            print(f"      ----  ----------   --------   ---------   --------")
            print(f"       AVG    {ate:9.3f}   {rte:8.3f}   {dpm:9.4f}   {re:8.3f}   (n={n})")
            pap = PAPER[mode]; t2 = PAPER_T2[mode]
            print(f"     PAPER    {pap['ate']:>9}   "
                  f"{(pap['rte'] if pap['rte'] is not None else t2['rte']):>8}   "
                  f"{(pap['rot_dpm'] if pap['rot_dpm'] is not None else '   -'):>9}   "
                  f"{t2['re']:>8}   (Table2 APE; Fig10 %/deg-m; Table2 avgRE)")
            print(f"      DEV     APE {ate-pap['ate']:+.3f} m"
                  + (f" | avgRTE {rte-pap['rte']:+.3f}%" if pap['rte'] else "")
                  + (f" | rot {dpm-pap['rot_dpm']:+.4f} deg/m" if pap['rot_dpm'] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
