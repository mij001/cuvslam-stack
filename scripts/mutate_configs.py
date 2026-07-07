#!/usr/bin/env python3
"""mutate_configs.py — THE config mutation engine.

The human owns configs/base/ — one full-featured SLAM config per
dataset-sequence × sensor modality (odometry-only where SLAM does not apply,
e.g. mono), each with an [eval] block. This script derives EVERYTHING else
into configs/generated/ (flat, gitignored, regenerate at will):

  pipeline kinds     <stem>_odom      [slam] removed, pose from odometry
                     <base>_async     sync_mode=false + async_sba=true (KITTI)
                     <base>_cpu       [slam].use_gpu=false (TUM RGB-D)
  feature toggles    <base>__<toggle> on one representative per modality:
                     sync/async/cpu/planar, odom_only, sba_async,
                     motion-model, denoising, landmarks export, multicam
                     Precision, unrectified, depth-stereo-tracking
  frame windows      <name>__win<START>x<COUNT> — OPTIONAL (--window); the
                     regime profiles full sequences by default

Deep-profiling selection is EMBEDDED IN THE CONFIGS: a `[profiling]
nvbit = true` block marks the runs that also get the (expensive) NVBit
memory-trace leg. The rule is deterministic: every base keeps its own marker
(the human's knob), and mutations in IMPORTANT_TOGGLES re-gain it; plain
pipeline kinds drop it. ~bases + important toggles ≈ 30% of the matrix.

Usage:
  python3 scripts/mutate_configs.py                      # -> configs/generated/ (flat)
  python3 scripts/mutate_configs.py --window 200:260     # + __win variants

The dashboard imports set_key/remove_slam from here, so custom configs go
through the identical transforms.
"""
from __future__ import annotations

import argparse
import os
import re

# representative base per modality family for the toggle pass
TOGGLE_REPS = [
    "kitti06_stereo_slam",
    "euroc_MH_01_easy_stereo_slam",
    "euroc_MH_01_easy_inertial_slam",
    "tum_fr3_long_office_household_rgbd_slam",
    "icl_living_room_traj1_rgbd_slam",
]

# toggles whose accuracy/behaviour effect is PROVEN large (coverage campaign,
# reports/2026-07-07_profiling_coverage) — these mutations keep the nvbit
# marker, so the deep memory-trace leg lands on the runs that matter.
IMPORTANT_TOGGLES = {"denoising", "slam_planar", "no_motion_model", "slam_cpu",
                     "odom_only", "unrectified", "slam_async"}

STEREO = ("stereo",)          # toggles valid only for a stereo base
RGBD = ("rgbd",)

NVBIT_BLOCK = "\n[profiling]\nnvbit = true\n"


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


def strip_nvbit(text):
    return re.sub(r"(?ms)^\[profiling\]\s*$.*?(?=^\[|\Z)", "", text).rstrip() + "\n"


def mark_nvbit(text):
    text = strip_nvbit(text)
    return text.rstrip() + "\n" + NVBIT_BLOCK


def nvbit_marked(text):
    return re.search(r"(?ms)^\[profiling\].*?^nvbit\s*=\s*true", text) is not None


def retarget(text, old_name, new_name, title=None):
    """Point the run-dir segment of output/eval paths at <new_name> and retitle."""
    text = text.replace(f"/{old_name}/", f"/{new_name}/")
    if title:
        text = re.sub(r"(?m)^# .*$", f"# {title}", text, count=1)
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
# feature toggles (variant-suffix, applies-to-modalities or None=all, transform)
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


def mutations(base_dir):
    """bases -> every derived config as [(name, text)], flat.

    Pipeline kinds mirror the paper's evaluation axes per family (KITTI adds
    async, TUM RGB-D adds cpu); feature toggles go on one representative per
    modality. The nvbit marker survives only where the deep leg matters:
    kind mutations drop it, IMPORTANT_TOGGLES re-gain it.
    """
    out = []
    for fn in sorted(os.listdir(base_dir)):
        if not fn.endswith(".toml"):
            continue
        name = fn[:-5]
        text = open(os.path.join(base_dir, fn)).read()
        if not name.endswith("_slam"):
            continue                      # odometry-only base (mono): no kinds
        stem = name[: -len("_slam")]
        out.append((f"{stem}_odom",
                    retarget_kind(strip_nvbit(remove_slam(text)),
                                  name, f"{stem}_odom", "odom")))
        if name.startswith("kitti"):
            t = set_key(set_key(text, "odometry", "async_sba", "true"),
                        "slam", "sync_mode", "false")
            out.append((f"{name}_async",
                        retarget_kind(strip_nvbit(t), name, f"{name}_async", "slam_async")))
        if name.startswith("tum_") and "_rgbd_" in f"{name}_":
            t = set_key(text, "slam", "use_gpu", "false")
            out.append((f"{name}_cpu",
                        retarget_kind(strip_nvbit(t), name, f"{name}_cpu", "slam_cpu")))

    # feature toggles on the representatives
    for base in TOGGLE_REPS:
        bpath = os.path.join(base_dir, base + ".toml")
        if not os.path.isfile(bpath):
            print(f"[skip] representative absent: {base}")
            continue
        btext = open(bpath).read()
        mod = modality(base)
        for suffix, mods, fn in VARIANTS:
            if mods is not None and mod not in mods:
                continue
            tag = f"{base}__{suffix}"
            t = fn(btext)
            t = retarget(t, base, tag, title=f"toggle mutation: {tag}")
            t = mark_nvbit(t) if suffix in IMPORTANT_TOGGLES else strip_nvbit(t)
            out.append((tag, t))
    return sorted(out)


# ─────────────────────────── cli ───────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="configs/base", help="human-owned base configs")
    ap.add_argument("--out", default="configs/generated", help="output dir (gitignored, flat)")
    ap.add_argument("--window", metavar="START:COUNT",
                    help="ALSO emit __win<START>x<COUNT> frame-window variants (optional)")
    args = ap.parse_args()

    muts = mutations(args.base)
    os.makedirs(args.out, exist_ok=True)
    for name, text in muts:
        with open(os.path.join(args.out, name + ".toml"), "w") as fh:
            fh.write(text)
    n = len(muts)
    if args.window:
        start, count = args.window.split(":")
        for fn in sorted(os.listdir(args.base)):
            if not fn.endswith(".toml"):
                continue
            text = open(os.path.join(args.base, fn)).read()
            tag = f"{fn[:-5]}__win{start}x{count}"
            with open(os.path.join(args.out, tag + ".toml"), "w") as fh:
                fh.write(set_window(strip_nvbit(text), int(start), int(count)))
            n += 1
    nb = sum(1 for _, t in muts if nvbit_marked(t))
    base_nb = sum(1 for f in os.listdir(args.base)
                  if f.endswith(".toml") and nvbit_marked(open(os.path.join(args.base, f)).read()))
    total_cfgs = n + sum(1 for f in os.listdir(args.base) if f.endswith(".toml"))
    print(f"[✓] {n} mutations -> {args.out}; matrix = {total_cfgs} configs; "
          f"nvbit-marked = {base_nb} bases + {nb} mutations")


if __name__ == "__main__":
    main()
