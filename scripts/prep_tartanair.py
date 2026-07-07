#!/usr/bin/env python3
"""prep_tartanair.py — make downloaded TartanAir sequences runnable by the stack.

TartanAir ships stereo PNGs (image_left/, image_right/) + a NED pose file
(pose_left.txt: one row per frame, `tx ty tz qx qy qz qw` in the NED world
frame) and no timestamps. The runner's image_folder source wants a times.txt,
and its evaluator understands TUM-format GT. So per sequence this writes:

  times.txt            synthetic timestamps at --fps (default 10), one per
                       left image, matching the image sort order.
  groundtruth_tum.txt  `t tx ty tz qx qy qz qw` at the same timestamps, i.e.
                       pose_left.txt reformatted with a leading timestamp.

No coordinate conversion is applied: cuVSLAM's estimate and this GT are both
per-frame camera poses, and the evaluator's SE3 (Umeyama) alignment absorbs a
consistent world-frame convention (NED vs cuVSLAM). That is the standard way
TartanAir is scored for VO/SLAM ATE.

Usage:  python3 prep_tartanair.py /mnt/data/tartanair [--fps 10]
        (scans <root>/<env>/<Easy|Hard>/P0xx/)
"""
from __future__ import annotations

import argparse
import glob
import os


def prep_seq(sdir, fps):
    left = sorted(glob.glob(os.path.join(sdir, "image_left", "*.png")))
    pose = os.path.join(sdir, "pose_left.txt")
    if not left or not os.path.isfile(pose):
        return f"[skip] {sdir} (missing image_left/*.png or pose_left.txt)"
    poses = [ln.split() for ln in open(pose) if ln.strip()]
    if len(poses) != len(left):
        return (f"[warn] {sdir}: {len(left)} images vs {len(poses)} poses — "
                "truncating to min (TartanAir usually matches; verify)")
    n = min(len(left), len(poses))
    dt = 1.0 / fps
    with open(os.path.join(sdir, "times.txt"), "w") as ft, \
         open(os.path.join(sdir, "groundtruth_tum.txt"), "w") as fg:
        fg.write("# timestamp tx ty tz qx qy qz qw  (TartanAir NED, SE3-aligned at eval)\n")
        for i in range(n):
            t = i * dt
            ft.write(f"{t:.6f}\n")
            fg.write(f"{t:.6f} " + " ".join(poses[i][:7]) + "\n")
    return f"[✓] {os.path.relpath(sdir)}: {n} frames"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()
    dirs = sorted(glob.glob(os.path.join(args.root, "*", "Hard", "P0*")) +
                  glob.glob(os.path.join(args.root, "*", "Easy", "P0*")))
    if not dirs:
        print(f"no TartanAir sequences under {args.root}/<env>/<Hard|Easy>/P0xx/")
        return
    for d in dirs:
        print(prep_seq(d, args.fps))


if __name__ == "__main__":
    main()
