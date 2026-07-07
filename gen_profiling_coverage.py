#!/usr/bin/env python3
"""gen_profiling_coverage.py — turn each dataset into many cuVSLAM-feature
variants, for a profiling campaign that also checks accuracy is unchanged.

Idea: one sequence, profiled under many toggle combinations, exercises many
different cuVSLAM code paths (stereo vs inertial vs mono vs RGB-D; odometry vs
SLAM; sync vs async; GPU vs CPU SLAM; motion-model / denoising / rectification
/ landmark-export / depth-stereo-tracking / planar-constraints on/off; SBA
sync/async; multicam Performance/Precision). Every emitted config KEEPS its
[eval] block so the profiling campaign can confirm the trajectory-vs-ground-
truth metrics do not deviate under profiling (no bug introduced).

Transforms the already-validated `configs/accuracy_matrix/*_slam.toml` bases
(so intrinsics/paths/eval are correct) into toggle variants. Data-driven:
only bases that exist are transformed.

Usage:
  python3 gen_profiling_coverage.py --matrix configs/accuracy_matrix \
      --out configs/profiling_coverage
"""
from __future__ import annotations

import argparse
import os
import re

# representative base per modality family (stereo / inertial / rgbd / rgbd-syn)
BASES = [
    "kitti06_stereo_slam",
    "euroc_MH_01_easy_stereo_slam",
    "euroc_MH_01_easy_inertial_slam",
    "tum_fr3_long_office_household_rgbd_slam",
    "icl_living_room_traj1_rgbd_slam",
]

STEREO = ("stereo",)          # toggles valid only for a stereo base
RGBD = ("rgbd",)


def modality(base):
    if "_stereo_" in base:
        return "stereo"
    if "_inertial_" in base:
        return "inertial"
    return "rgbd"


def set_key(text, section, key, value):
    """Set/insert `key = value` inside [section]; append section if missing."""
    val = value if isinstance(value, str) and value in ("true", "false") else value
    val = f'"{value}"' if isinstance(value, str) and value not in ("true", "false") else value
    # section header may be indented (e.g. the nested [odometry.rgbd] table)
    pat = re.compile(rf"(?ms)^[ \t]*\[{re.escape(section)}\][ \t]*$.*?(?=^[ \t]*\[|\Z)")
    m = pat.search(text)
    if not m:
        text = text.rstrip() + f"\n\n[{section}]\n{key} = {val}\n"
        return text
    block = m.group(0)
    kpat = re.compile(rf"(?m)^([ \t]*){re.escape(key)}[ \t]*=.*$")
    if kpat.search(block):
        newblock = kpat.sub(rf"\g<1>{key} = {val}", block, count=1)
    else:
        # insert right after the (possibly indented) section header line
        newblock = re.sub(rf"(?m)^([ \t]*\[{re.escape(section)}\][ \t]*)$",
                          rf"\1\n{key} = {val}", block, count=1)
    return text[:m.start()] + newblock + text[m.end():]


def set_rgbd_key(text, key, value):
    """Set a key inside the nested [odometry.rgbd] table."""
    return set_key(text, "odometry.rgbd", key, value)


def remove_slam(text):
    text = re.sub(r"(?ms)^\[slam\]\s*$.*?(?=^\[|\Z)", "", text)
    # a SLAM base sets pose_source="slam"; odometry-only must not
    text = text.replace('pose_source = "slam"', 'pose_source = "odometry"')
    return text.rstrip() + "\n"


def retitle(text, tag):
    """Update the leading comment + output/eval paths to a unique per-variant
    run dir so variants don't overwrite each other's trajectory/eval."""
    text = re.sub(r"(?m)^# .*$", f"# profiling-coverage variant: {tag}", text, count=1)
    text = text.replace("/accuracy_out/", "/profiling_coverage_out/")
    # rename the run-dir segment (the one right after the out root) to <tag>
    text = re.sub(r"(/profiling_coverage_out/)[^/\"]+(/)", rf"\g<1>{tag}\g<2>", text)
    return text


# (variant-suffix, applies-to-modalities or None=all, transform)
VARIANTS = [
    ("slam_sync",        None,   lambda t: set_key(t, "slam", "sync_mode", "true")),
    ("slam_async",       None,   lambda t: set_key(t, "slam", "sync_mode", "false")),
    ("slam_cpu",         None,   lambda t: set_key(t, "slam", "use_gpu", "false")),
    ("slam_planar",      None,   lambda t: set_key(t, "slam", "planar_constraints", "true")),
    ("odom_only",        None,   remove_slam),
    ("sba_async",        None,   lambda t: set_key(t, "odometry", "async_sba", "true")),
    ("no_motion_model",  None,   lambda t: set_key(t, "odometry", "use_motion_model", "false")),
    ("denoising",        None,   lambda t: set_key(t, "odometry", "use_denoising", "true")),
    ("landmarks_export", None,   lambda t: set_key(t, "odometry", "enable_final_landmarks_export", "true")),
    ("multicam_precision", STEREO, lambda t: set_key(t, "odometry", "multicam_mode", "Precision")),
    ("unrectified",      STEREO, lambda t: set_key(t, "odometry", "rectified_stereo_camera", "false")),
    ("depth_stereo_track", RGBD, lambda t: set_rgbd_key(t, "enable_depth_stereo_tracking", "true")),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default="configs/accuracy_matrix")
    ap.add_argument("--out", default="configs/profiling_coverage")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    n = 0

    def emit(tag, text):
        nonlocal n
        with open(os.path.join(args.out, tag + ".toml"), "w") as fh:
            fh.write(retitle(text, tag))
        n += 1

    for base in BASES:
        bpath = os.path.join(args.matrix, base + ".toml")
        if not os.path.isfile(bpath):
            print(f"[skip] base absent: {base}")
            continue
        btext = open(bpath).read()
        mod = modality(base)
        # the base itself (slam, default toggles) is variant #1
        emit(f"{base}__base", btext)
        for suffix, mods, fn in VARIANTS:
            if mods is not None and mod not in mods:
                continue
            emit(f"{base}__{suffix}", fn(btext))

    print(f"[✓] {n} profiling-coverage configs -> {args.out}")


if __name__ == "__main__":
    main()
