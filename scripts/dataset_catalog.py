#!/usr/bin/env python3
"""dataset_catalog.py — the single source of dataset knowledge for every config
generator (accuracy matrix, profiling coverage, characterization campaign).

Sequence lists, calibrated intrinsics, and the tiny calibration readers were
previously duplicated per generator; they live here once. Paths are NOT here —
each generator resolves its own data root (--root / ${CUVSLAM_DATA2}).
"""
from __future__ import annotations

import struct

# ── sequence lists ───────────────────────────────────────────────────────────
KITTI_SEQS = [f"{i:02d}" for i in range(11)]            # KITTI odometry 00-10

EUROC = [("MH/machine_hall", s) for s in
         ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]] + \
        [("VR1/vicon_room1", s) for s in ["V1_01_easy", "V1_02_medium", "V1_03_difficult"]] + \
        [("VR2/vicon_room2", s) for s in ["V2_01_easy", "V2_02_medium", "V2_03_difficult"]]

# TUM fr3 — the cuVSLAM paper's Table-4 set (10 sequences) UNION our two
# earlier nostructure_notexture extras. Generators are data-driven: an absent
# sequence is skipped, so downloading a missing one just activates more configs.
TUM_RGBD_PAPER = [
    "rgbd_dataset_freiburg3_cabinet",
    "rgbd_dataset_freiburg3_long_office_household",
    "rgbd_dataset_freiburg3_nostructure_texture_far",
    "rgbd_dataset_freiburg3_nostructure_texture_near_withloop",
    "rgbd_dataset_freiburg3_sitting_halfsphere",
    "rgbd_dataset_freiburg3_sitting_xyz",
    "rgbd_dataset_freiburg3_structure_texture_far",
    "rgbd_dataset_freiburg3_structure_texture_near",
    "rgbd_dataset_freiburg3_teddy",
]
TUM_RGBD_EXTRA = [
    "rgbd_dataset_freiburg3_nostructure_notexture_far",
    "rgbd_dataset_freiburg3_nostructure_notexture_near_withloop",
]
TUM_RGBD = TUM_RGBD_PAPER + TUM_RGBD_EXTRA

# ICL-NUIM (Handa et al.) — synthetic RGB-D, the paper's "Mono-Depth" benchmark.
# TUM-format ("_frei_png") release: tum source + tum gt_format, depth scale 5000.
ICL_NUIM = [f"living_room_traj{i}_frei_png" for i in range(4)] + \
           [f"traj{i}_frei_png" for i in range(4)]      # office room traj0..3

# ── calibrated intrinsics (fixed per dataset family) ─────────────────────────
FR3 = dict(focal=[535.4, 539.2], principal=[320.1, 247.6])          # TUM fr3 640x480
ICL = dict(focal=[481.20, 480.00], principal=[319.50, 239.50])      # ICL 640x480
# TartanAir: pinhole 640x480, fx=fy=320, baseline 0.25 m, poses in NED.
TARTANAIR = dict(focal=[320.0, 320.0], principal=[320.0, 240.0], baseline=0.25)

# ── tiny calibration readers ─────────────────────────────────────────────────
def png_size(path):
    """(width, height) straight out of a PNG header."""
    with open(path, "rb") as fh:
        d = fh.read(26)
    return struct.unpack(">II", d[16:24])


def kitti_calib(path):
    """(f, cx, cy, baseline_m) for the color pair from calib.txt P2/P3."""
    P = {}
    for line in open(path):
        k, _, rest = line.partition(":")
        vals = rest.split()
        if len(vals) == 12:
            P[k] = [float(v) for v in vals]
    f = P["P2"][0]
    return f, P["P2"][2], P["P2"][6], (P["P2"][3] - P["P3"][3]) / f
