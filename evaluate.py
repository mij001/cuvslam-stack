#!/usr/bin/env python3
"""Standalone trajectory evaluation (independent of cuvslam).

Compute ATE / RPE / avgRTE / avgRE for an existing TUM-format trajectory file
against ground truth:

    python evaluate.py est_tum.txt gt.csv --gt-format euroc \
        --euroc-cam0-yaml /path/mav0/cam0/sensor.yaml --align se3

    python evaluate.py est_tum.txt poses.txt --gt-format kitti --align se3
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cuvslam_runner import eval as ev  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trajectory", help="estimated trajectory (TUM format)")
    p.add_argument("ground_truth", help="ground-truth file")
    p.add_argument("--traj-time-unit", default="s", choices=["s", "ms", "us", "ns"])
    p.add_argument("--gt-format", default="euroc", choices=["euroc", "tum", "kitti"])
    p.add_argument("--gt-time-unit", default="s", choices=["s", "ms", "us", "ns"])
    p.add_argument("--gt-fps", type=float, default=10.0, help="kitti GT index timing")
    p.add_argument("--align", default="se3", choices=["se3", "sim3", "none"])
    p.add_argument("--euroc-cam0-yaml", default="",
                   help="apply this cam0 sensor.yaml T_BS to move EuRoC GT into the cam0 frame")
    p.add_argument("--max-time-diff", type=float, default=0.02, help="association window (s)")
    p.add_argument("--rpe-distances", default="",
                   help='comma-separated metres, or "kitti" for 100..800; empty=auto')
    p.add_argument("--report", default="", help="optional path to write the report")
    args = p.parse_args(argv)

    est = ev.load_tum(args.trajectory, args.traj_time_unit)
    if args.gt_format == "euroc":
        gt = ev.load_gt_euroc(args.ground_truth)
    elif args.gt_format == "tum":
        gt = ev.load_gt_tum(args.ground_truth, args.gt_time_unit)
    else:
        gt = ev.load_gt_kitti(args.ground_truth, args.gt_fps)

    if args.euroc_cam0_yaml:
        T = ev.read_euroc_cam0_extrinsic(args.euroc_cam0_yaml)
        gt = ev.apply_right_extrinsic(gt, T)

    rpe_distances = None
    if args.rpe_distances == "kitti":
        rpe_distances = [100, 200, 300, 400, 500, 600, 700, 800]
    elif args.rpe_distances:
        rpe_distances = [float(x) for x in args.rpe_distances.split(",")]

    result = ev.evaluate(est, gt, align=args.align,
                         max_diff_ns=int(args.max_time_diff * 1e9),
                         rpe_distances=rpe_distances)
    report = ev.format_report(result, title=os.path.basename(args.ground_truth))
    print(report)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as handle:
            handle.write(report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
