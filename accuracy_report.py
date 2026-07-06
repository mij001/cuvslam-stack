#!/usr/bin/env python3
"""accuracy_report.py — aggregate the accuracy matrix and compare to the paper.

Walks the eval reports produced by ws_accuracy_matrix.sh, parses the paper's
three metrics (RMSE APE, avgRTE, avgRE) plus the fixed-1s TUM RPE, aggregates
per (dataset, variant, mode) and prints/writes the comparison against the
cuVSLAM technical report (arXiv:2506.04359) Tables 2 and 6.

Comparison discipline:
  * RMSE APE is the apples-to-apples metric (standard definition, alignment
    stated). avgRTE/avgRE depend on segment definitions — the paper's Table 2
    is "without segmentation" while ours uses distance segments — so those
    columns carry a definition caveat and APE drives the verdict.
  * EuRoC pure-stereo averages exclude V2_03 (paper footnote: V2_03 only in
    Stereo-Inertial); the per-sequence table still shows it.
  * TUM: paper averages 10 fr3 sequences, we have 4 on disk (2 overlap) →
    per-sequence comparison for the overlapping two, no mode-average claim.

Usage:  python3 accuracy_report.py [--root /mnt/data/accuracy_out] [--out results/]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re

# (avgRTE %, avgRE deg, RMSE APE m) — cuVSLAM paper Table 2
PAPER_T2 = {
    ("kitti", "stereo", "odom"): (0.33, 1.14, 3.00),
    ("kitti", "stereo", "slam"): (0.27, 0.93, 1.98),
    ("euroc", "stereo", "odom"): (0.29, 1.96, 0.13),
    ("euroc", "stereo", "slam"): (0.17, 1.12, 0.054),
    ("euroc", "inertial", "odom"): (0.39, 2.69, 0.19),
    ("euroc", "inertial", "slam"): (0.29, 2.27, 0.13),
    ("tum", "rgbd", "odom"): (1.35, 5.52, 0.11),
    ("tum", "rgbd", "slam"): (0.99, 4.13, 0.065),
    # Mono-Depth — ICL-NUIM (2506.04359v3 Table 2, exact)
    ("icl", "rgbd", "odom"): (0.41, 0.99, 0.026),
    ("icl", "rgbd", "slam"): (0.44, 0.97, 0.026),
    # Stereo-Inertial — TUM-VI Room (Table 2)
    ("tumvi", "inertial", "odom"): (0.20, 3.85, 0.18),
    ("tumvi", "inertial", "slam"): (0.12, 3.00, 0.12),
    # Mono-Depth — AR-table (Chen 2023; dataset not yet on disk, baseline kept
    # for when it is): odom 0.34/3.59/0.09, slam 0.19/1.68/0.025.
    # ("artable", "rgbd", "odom"): (0.34, 3.59, 0.09),
}

# Table 3 — Multi-Stereo (multi-camera) mode. TartanAir here is **V2, Hard,
# Multi-Stereo** (not the V1 single-stereo); TartanGround is also multi-camera;
# R2B is proprietary. These need multi-camera rig configs (separate effort).
PAPER_T3_MULTISTEREO = {
    ("tartanair_v2", "hard", "odom"): (2.44, 13.98, 5.24),
    ("tartanair_v2", "hard", "slam"): (2.26, 12.76, 4.99),
    ("tartanground", "-", "odom"): (0.21, 0.48, 0.09),
    ("tartanground", "-", "slam"): (0.17, 0.37, 0.07),
    ("r2b", "-", "odom"): (0.18, 1.15, 0.28),      # proprietary
    ("r2b", "-", "slam"): (0.11, 0.70, 0.18),
}
# Table 6 per-sequence rows we have on disk (avgRTE, avgRE, RMSE APE m)
PAPER_T6 = {
    ("fr3_long_office_household", "odom"): (1.27, 7.89, 0.20),
    ("fr3_long_office_household", "slam"): (0.96, 5.94, 0.06),
    ("fr3_nostructure_texture_far", "odom"): (1.29, 1.66, 0.07),
    ("fr3_nostructure_texture_far", "slam"): (1.44, 1.63, 0.06),
}

_PAT = {
    "ape_m": re.compile(r"ATE / RMSE APE\s*:\s*[\d.]+ cm\s*\(([\d.]+) m\)"),
    "rte_pct": re.compile(r"avgRTE \(translation\):\s*([\d.]+) %"),
    "rot_dpm": re.compile(r"RPE rotation\s*:\s*([\d.]+) deg/m"),
    "re_deg": re.compile(r"avgRE \(rotation\)\s*:\s*([\d.]+) deg"),
    "rpe1s_m": re.compile(r"RPE translation\s*:\s*([\d.]+) m rmse"),
    "rpe1s_deg": re.compile(r"RPE rotation\s*:\s*([\d.]+) deg rmse"),
    "matched": re.compile(r"matched poses\s*:\s*(\d+)"),
}

_NAME = re.compile(
    r"^(kitti(?P<kseq>\d\d)_(?P<kvar>stereo)_(?P<kkind>odom|slam|slam_async)"
    r"|euroc_(?P<eseq>[A-Z0-9_a-z]+?)_(?P<evar>stereo|inertial|mono)_(?P<ekind>odom|slam)"
    r"|tum_(?P<tseq>fr3_\w+?)_(?P<tvar>rgbd)_(?P<tkind>odom|slam|slam_cpu)"
    r"|icl_(?P<iseq>[a-z0-9_]+?)_(?P<ivar>rgbd)_(?P<ikind>odom|slam)"
    r"|tartan_(?P<aseq>[a-z0-9_]+?_Hard_P\d+)_(?P<avar>stereo)_(?P<akind>odom|slam)"
    r"|tumvi_(?P<vseq>\w+?)_(?P<vvar>inertial)_(?P<vkind>odom|slam))$")


def parse_eval(path):
    txt = open(path, errors="replace").read()
    out = {}
    for k, rx in _PAT.items():
        m = rx.search(txt)
        if m:
            out[k] = float(m.group(1))
    return out


def parse_name(name):
    m = _NAME.match(name)
    if not m:
        return None
    g = m.groupdict()
    if g["kseq"]:
        return "kitti", g["kseq"], g["kvar"], g["kkind"]
    if g["eseq"]:
        return "euroc", g["eseq"], g["evar"], g["ekind"]
    if g["tseq"]:
        return "tum", g["tseq"], g["tvar"], g["tkind"]
    if g["iseq"]:
        return "icl", g["iseq"], g["ivar"], g["ikind"]
    if g["aseq"]:
        return "tartan", g["aseq"], g["avar"], g["akind"]
    return "tumvi", g["vseq"], g["vvar"], g["vkind"]


def collect(root):
    rows = []
    for rep in sorted(glob.glob(os.path.join(root, "*", "eval.txt"))):
        name = os.path.basename(os.path.dirname(rep))
        parsed = parse_name(name)
        if not parsed:
            continue
        rows.append((name, *parsed, parse_eval(rep)))
    return rows


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


# Convergence gate: avgRTE (segment-relative translational drift) is
# scale-independent — it works for KITTI's kilometre trajectories and EuRoC's
# metre ones alike — so it is the right "did tracking hold?" criterion. A run
# above this diverged; a diverged run must not enter a mode average.
CONVERGE_RTE_PCT = 5.0


def exclusion_reason(name, ds, seq, var, kind, m):
    """Why a run is excluded from paper comparison, or '' if it converged."""
    rte = m.get("rte_pct")
    if ds == "tumvi":
        return ("INVALID: TUM-VI ~195° fisheye not undistorted to pinhole "
                "(<180° cuVSLAM support) — data-prep gap, not a tracking result")
    if var == "mono":
        return ("scale-ambiguous: monocular needs Sim3 (scale) alignment; "
                "SE3-based avgRTE/APE are not meaningful for mono")
    if rte is None:
        return "no relative-error metric parsed"
    if rte >= CONVERGE_RTE_PCT:
        if "V2_03" in seq:
            return (f"DIVERGED (avgRTE {rte:.1f}%): V2_03_difficult is the "
                    "hardest EuRoC sequence — aggressive motion + blur; the "
                    "paper reports it only in stereo-inertial and excludes it "
                    "from stereo averages")
        return f"DIVERGED (avgRTE {rte:.1f}% ≥ {CONVERGE_RTE_PCT}%)"
    return ""


def converged(name, ds, seq, var, kind, m):
    return exclusion_reason(name, ds, seq, var, kind, m) == ""


def fmt(v, nd=3):
    return f"{v:.{nd}f}" if v is not None else "—"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/mnt/data/accuracy_out")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    rows = collect(args.root)
    if not rows:
        raise SystemExit(f"no eval reports under {args.root}")
    os.makedirs(args.out, exist_ok=True)

    # long CSV
    csv_path = os.path.join(args.out, "accuracy_matrix.csv")
    keys = ["ape_m", "rte_pct", "rot_dpm", "re_deg", "rpe1s_m", "rpe1s_deg", "matched"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "dataset", "sequence", "variant", "mode"] + keys)
        for name, ds, seq, var, kind, m in rows:
            w.writerow([name, ds, seq, var, kind] + [m.get(k, "") for k in keys])

    conv = [r for r in rows if converged(*r)]
    lines = ["# Accuracy matrix vs the cuVSLAM paper (arXiv:2506.04359)", ""]
    lines.append(f"{len(rows)} runs evaluated against ground truth "
                 f"(full sequences, baseline wheel, RTX 2000 Ada); "
                 f"{len(conv)} converged (avgRTE < {CONVERGE_RTE_PCT}% "
                 f"segment-relative), {len(rows) - len(conv)} excluded with "
                 f"stated reasons (see 'Convergence & exclusions').")
    lines.append("")

    # mode averages vs Table 2 — over CONVERGED runs only
    lines.append("## Mode averages vs paper Table 2 (converged runs only)")
    lines.append("")
    lines.append("APE is the definition-stable comparison; avgRTE/avgRE use "
                 "our segment definitions (KITTI 100–800 m; EuRoC 8/16/32 m; "
                 "TUM 1/2/4 m) vs the paper's Table-2 values — directional "
                 "only. Averages exclude diverged/invalid runs; `n` is the "
                 "converged count and the excluded ones are itemized below.")
    lines.append("")
    lines.append("| dataset/variant/mode | n | APE m (ours) | APE m (paper) | Δ | avgRTE % (ours/paper) | avgRE ° (ours/paper) |")
    lines.append("|---|---|---|---|---|---|---|")
    for (ds, var, kind), paper in PAPER_T2.items():
        sel = [m for (_n, d, s, v, k, m) in conv
               if d == ds and v == var and k == kind]
        if not sel:
            continue
        ape = mean([m.get("ape_m") for m in sel])
        rte = mean([m.get("rte_pct") for m in sel])
        re_ = mean([m.get("re_deg") for m in sel])
        p_rte, p_re, p_ape = paper
        d_ape = f"{ape - p_ape:+.3f}" if ape is not None else "—"
        lines.append(f"| {ds} {var} {kind} | {len(sel)} | {fmt(ape)} | {p_ape} "
                     f"| {d_ape} | {fmt(rte)} / {p_rte} | {fmt(re_, 2)} / {p_re} |")
    lines.append("")

    # convergence & exclusions — the honesty section
    lines.append("## Convergence & exclusions")
    lines.append("")
    excluded = [r for r in rows if not converged(*r)]
    lines.append(f"{len(excluded)} of {len(rows)} runs excluded from the "
                 "paper comparison:")
    lines.append("")
    lines.append("| run | avgRTE % | APE m | reason |")
    lines.append("|---|---|---|---|")
    for name, ds, seq, var, kind, m in excluded:
        lines.append(f"| {name} | {fmt(m.get('rte_pct'), 1)} | "
                     f"{fmt(m.get('ape_m'), 1)} | {exclusion_reason(name, ds, seq, var, kind, m)} |")
    lines.append("")

    # TUM per-sequence vs Table 6
    lines.append("## TUM fr3 per-sequence vs paper Table 6 (shared sequences)")
    lines.append("")
    lines.append("| sequence / mode | APE m (ours) | APE m (paper) | avgRTE (ours/paper) | avgRE (ours/paper) |")
    lines.append("|---|---|---|---|---|")
    for (seq, kind), (p_rte, p_re, p_ape) in PAPER_T6.items():
        sel = [m for (_n, d, s, v, k, m) in rows
               if d == "tum" and s == seq and k == kind]
        if not sel:
            continue
        m = sel[0]
        lines.append(f"| {seq} {kind} | {fmt(m.get('ape_m'))} | {p_ape} "
                     f"| {fmt(m.get('rte_pct'))} / {p_rte} | {fmt(m.get('re_deg'), 2)} / {p_re} |")
    lines.append("")

    # feature-toggle deltas within our own runs (definition-consistent).
    # Only pairs where BOTH runs converged — a diverged twin makes the delta
    # meaningless (that's what turned "SLAM vs odom" into a −1762 m artifact).
    lines.append("## Feature-toggle deltas (converged pairs only, ours vs ours)")
    lines.append("")
    lines.append("Same sequence, one feature toggled; negative ΔAPE = the "
                 "toggle improved accuracy. Pairs with a diverged member are "
                 "dropped (count shown).")
    lines.append("")
    lines.append("| toggle | pairs | mean ΔAPE m (toggle − base) |")
    lines.append("|---|---|---|")
    conv_names = {r[0] for r in conv}
    by_run = {n: (d, s, v, k, m) for (n, d, s, v, k, m) in rows}

    def pair_delta(kind_a, kind_b, var=None, ds=None):
        deltas = []
        for n, (d, s, v, k, m) in by_run.items():
            if k != kind_a or (var and v != var) or (ds and d != ds):
                continue
            twin = n.replace(f"_{kind_a}", f"_{kind_b}")
            if twin in by_run and n in conv_names and twin in conv_names:
                mb = by_run[twin][4]
                if m.get("ape_m") is not None and mb.get("ape_m") is not None:
                    deltas.append(mb["ape_m"] - m["ape_m"])
        return deltas

    for label, (a, b, var, ds) in {
        "SLAM vs odometry": ("odom", "slam", None, None),
        "async SLAM vs sync SLAM (KITTI)": ("slam", "slam_async", None, "kitti"),
        "CPU SLAM vs GPU SLAM (TUM)": ("slam", "slam_cpu", None, "tum"),
    }.items():
        d = pair_delta(a, b, var, ds)
        lines.append(f"| {label} | {len(d)} | {fmt(mean(d), 4)} |")

    # IMU delta: euroc inertial vs stereo, same kind (converged pairs only)
    for kind in ("odom", "slam"):
        deltas = []
        for n, (d, s, v, k, m) in by_run.items():
            if d == "euroc" and v == "stereo" and k == kind and n in conv_names:
                twin = n.replace("_stereo_", "_inertial_")
                if twin in by_run and twin in conv_names:
                    mb = by_run[twin][4]
                    if m.get("ape_m") is not None and mb.get("ape_m") is not None:
                        deltas.append(mb["ape_m"] - m["ape_m"])
        lines.append(f"| IMU on vs off (EuRoC {kind}) | {len(deltas)} | {fmt(mean(deltas), 4)} |")
    lines.append("")

    # QoR: instrumented vs baseline
    qor = collect(os.path.join(args.root, "qor_tagged"))
    if qor:
        lines.append("## QoR: instrumented wheel vs baseline (same configs)")
        lines.append("")
        lines.append("| run | APE m baseline | APE m instrumented | Δ |")
        lines.append("|---|---|---|---|")
        for name, ds, seq, var, kind, m in qor:
            base = by_run.get(name)
            if base and base[4].get("ape_m") is not None and m.get("ape_m") is not None:
                d = m["ape_m"] - base[4]["ape_m"]
                lines.append(f"| {name} | {fmt(base[4]['ape_m'])} | {fmt(m.get('ape_m'))} | {d:+.4f} |")
        lines.append("")

    # per-sequence full dump
    lines.append("## All runs (per-sequence)")
    lines.append("")
    lines.append("| run | APE m | avgRTE % | avgRE ° | RPE@1s m/° |")
    lines.append("|---|---|---|---|---|")
    for name, ds, seq, var, kind, m in rows:
        lines.append(f"| {name} | {fmt(m.get('ape_m'))} | {fmt(m.get('rte_pct'))} "
                     f"| {fmt(m.get('re_deg'), 2)} | {fmt(m.get('rpe1s_m'))}/{fmt(m.get('rpe1s_deg'), 2)} |")

    md_path = os.path.join(args.out, "accuracy_matrix_report.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[✓] {csv_path}")
    print(f"[✓] {md_path}")
    for ln in lines[:40]:
        print(ln)


if __name__ == "__main__":
    main()
