#!/usr/bin/env python3
"""Convert KITTI ground-truth poses (3x4 row-major per line) to TUM format.

evo's `evo_ape tum` needs both files as `timestamp tx ty tz qx qy qz qw`, but
KITTI ground truth (poses/<seq>.txt) is 12 numbers per line with no timestamps.
This rewrites it as TUM, taking timestamps from the sequence's times.txt so they
match a cuVSLAM run that used `[input.timestamps] mode="file" path="times.txt"`.

    python kitti_poses_to_tum.py poses/06.txt sequences/06/times.txt > gt_06_tum.txt
    # times.txt optional; without it, frame indices are used as timestamps.
"""

import sys

import numpy as np
from scipy.spatial.transform import Rotation as R


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    poses = [np.array(line.split(), dtype=float).reshape(3, 4)
             for line in open(sys.argv[1]) if line.strip()]
    if len(sys.argv) > 2:
        times = [float(line.split()[0]) for line in open(sys.argv[2]) if line.strip()]
    else:
        times = list(range(len(poses)))
    if len(times) < len(poses):
        print(f"error: {len(times)} timestamps < {len(poses)} poses", file=sys.stderr)
        return 1

    out = []
    for i, m in enumerate(poses):
        t = m[:, 3]
        q = R.from_matrix(m[:, :3]).as_quat()  # x, y, z, w
        out.append(f"{times[i]:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                   f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}")
    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
