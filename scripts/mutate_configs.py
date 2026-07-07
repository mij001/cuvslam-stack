#!/usr/bin/env python3
"""mutate_configs.py — THE config mutation engine.

One base config per dataset-sequence × sensor modality lives in configs/base/
(a full-featured SLAM config with [eval], or an odometry config where SLAM
does not apply, e.g. mono). Every other config the stack consumes is a
MUTATION of a base, produced by this script — nothing else generates variant
TOMLs, so the mutation matrix is defined exactly once:

  accuracy set   pipeline-kind mutations, matching the paper's evaluation
                 axes:  <base>            (slam, as-is)
                        <stem>_odom       ([slam] removed, pose from odometry)
                        <base>_async      (sync_mode=false + async_sba=true;
                                           KITTI bases — the paper's async row)
                        <base>_cpu        ([slam].use_gpu=false; TUM RGB-D
                                           bases — the paper's CPU row)
  coverage set   accuracy set × feature toggles: every accuracy config as
                 `__base` (breadth) + finer cuVSLAM toggles on one
                 representative per modality (depth): sync/async/cpu/planar,
                 odom_only, sba_async, motion-model, denoising, landmark
                 export, multicam Precision, unrectified, depth-stereo-track
  window set     bounded-frame captures for kernel-level profiling:
                 <name>__win<START>x<COUNT> ([run].start_index/max_frames)

Usage:
  python3 scripts/mutate_configs.py --select accuracy            # -> configs/generated/accuracy/
  python3 scripts/mutate_configs.py --select coverage            # -> configs/generated/coverage/
  python3 scripts/mutate_configs.py --select all
  python3 scripts/mutate_configs.py --select window --window 200:260

configs/generated/ is disposable (gitignored); regenerate at will. The
dashboard imports set_key/remove_slam from here, so its custom-config variants
apply the identical transforms.
"""
from __future__ import annotations

import argparse
import os
import re

# representative base per modality family for the coverage depth pass
COVERAGE_REPS = [
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


# ─────────────────────────── core transforms ───────────────────────────
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


def retarget_kind(text, old_name, new_name, kind_word):
    """Accuracy-kind mutation bookkeeping: run-dir paths + title kind word."""
    text = text.replace(f"/accuracy_out/{old_name}/", f"/accuracy_out/{new_name}/")
    lines = text.split("\n", 1)
    if lines[0].startswith("#"):
        # swap the kind word itself (titles may carry trailing annotations)
        lines[0] = re.sub(r"\bslam\b", kind_word, lines[0], count=1)
        text = "\n".join(lines)
    return text


def set_window(text, start, count):
    text = set_key(text, "run", "start_index", start)
    return set_key(text, "run", "max_frames", count)


# ─────────────────────────── mutation sets ───────────────────────────
# feature toggles for the coverage depth pass
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


def accuracy_set(base_dir):
    """bases -> the full accuracy matrix as [(name, text)], sorted by name.

    Pipeline-kind targets mirror the paper's evaluation axes per family:
    every SLAM base yields slam+odom; KITTI adds slam_async; TUM RGB-D adds
    slam_cpu; odometry-only bases (mono) pass through unchanged.
    """
    out = []
    for fn in sorted(os.listdir(base_dir)):
        if not fn.endswith(".toml"):
            continue
        name = fn[:-5]
        text = open(os.path.join(base_dir, fn)).read()
        if not name.endswith("_slam"):        # odometry-only base (mono)
            out.append((name, text))
            continue
        stem = name[: -len("_slam")]
        out.append((name, text))                                     # slam
        out.append((f"{stem}_odom",
                    retarget_kind(remove_slam(text), name, f"{stem}_odom", "odom")))
        if name.startswith("kitti"):
            t = set_key(set_key(text, "odometry", "async_sba", "true"),
                        "slam", "sync_mode", "false")
            out.append((f"{name}_async",
                        retarget_kind(t, name, f"{name}_async", "slam_async")))
        if name.startswith("tum_") and "_rgbd_" in f"{name}_":
            t = set_key(text, "slam", "use_gpu", "false")
            out.append((f"{name}_cpu",
                        retarget_kind(t, name, f"{name}_cpu", "slam_cpu")))
    return sorted(out)


def coverage_set(acc):
    """accuracy set -> the feature-toggle coverage set as [(tag, text)]."""
    out = []
    # breadth: every accuracy config, retargeted into profiling_coverage_out
    for name, text in acc:
        out.append((f"{name}__base", retitle(text, f"{name}__base")))
    # depth: finer toggles on one representative per modality
    acc_by_name = dict(acc)
    for base in COVERAGE_REPS:
        btext = acc_by_name.get(base)
        if btext is None:
            print(f"[skip] representative absent: {base}")
            continue
        mod = modality(base)
        for suffix, mods, fn in VARIANTS:
            if mods is not None and mod not in mods:
                continue
            tag = f"{base}__{suffix}"
            out.append((tag, retitle(fn(btext), tag)))
    return out


# ─────────────────────────── cli ───────────────────────────
def write_set(pairs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, text in pairs:
        with open(os.path.join(out_dir, name + ".toml"), "w") as fh:
            fh.write(text)
    return len(pairs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="configs/base", help="canonical base configs")
    ap.add_argument("--out", default="configs/generated", help="output root (gitignored)")
    ap.add_argument("--select", default="all",
                    choices=["accuracy", "coverage", "window", "all"])
    ap.add_argument("--window", default="200:260", metavar="START:COUNT",
                    help="frame window for --select window")
    args = ap.parse_args()

    acc = accuracy_set(args.base)
    total = 0
    if args.select in ("accuracy", "all"):
        total += write_set(acc, os.path.join(args.out, "accuracy"))
    if args.select in ("coverage", "all"):
        total += write_set(coverage_set(acc), os.path.join(args.out, "coverage"))
    if args.select == "window":
        start, count = args.window.split(":")
        win = [(f"{n}__win{start}x{count}", set_window(t, int(start), int(count)))
               for n, t in acc]
        total += write_set(win, os.path.join(args.out, "window"))
    print(f"[✓] {total} mutated configs -> {args.out} (from {len(acc)} accuracy "
          f"configs / {sum(1 for _ in os.listdir(args.base) if _.endswith('.toml'))} bases)")


if __name__ == "__main__":
    main()
