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

    lines = ["# Accuracy matrix vs the cuVSLAM paper (arXiv:2506.04359)", ""]
    lines.append(f"{len(rows)} runs evaluated against ground truth "
                 f"(full sequences, baseline wheel, RTX 2000 Ada).")
    lines.append("")

    # mode averages vs Table 2
    lines.append("## Mode averages vs paper Table 2")
    lines.append("")
    lines.append("APE is the definition-stable comparison; avgRTE/avgRE use "
                 "our segment definitions (KITTI 100–800 m; EuRoC 8/16/32 m; "
                 "TUM 1/2/4 m) vs the paper's unsegmented Table-2 values — "
                 "directional only.")
    lines.append("")
    lines.append("| dataset/variant/mode | n | APE m (ours) | APE m (paper) | Δ | avgRTE % (ours/paper) | avgRE ° (ours/paper) |")
    lines.append("|---|---|---|---|---|---|---|")
    for (ds, var, kind), paper in PAPER_T2.items():
        sel = [m for (_n, d, s, v, k, m) in rows
               if d == ds and v == var and k == kind
               and not (ds == "euroc" and var == "stereo" and "V2_03" in s)]
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

    # feature-toggle deltas within our own runs (definition-consistent)
    lines.append("## Feature-toggle deltas (ours vs ours — definition-consistent)")
    lines.append("")
    lines.append("| toggle | pairs | mean ΔAPE m (toggle − base) |")
    lines.append("|---|---|---|")
    by_run = {n: (d, s, v, k, m) for (n, d, s, v, k, m) in rows}

    def pair_delta(kind_a, kind_b, var=None, ds=None):
        deltas = []
        for n, (d, s, v, k, m) in by_run.items():
            if k != kind_a or (var and v != var) or (ds and d != ds):
                continue
            twin = n.replace(f"_{kind_a}", f"_{kind_b}")
            if twin in by_run:
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

    # IMU delta: euroc inertial vs stereo, same kind
    for kind in ("odom", "slam"):
        deltas = []
        for n, (d, s, v, k, m) in by_run.items():
            if d == "euroc" and v == "stereo" and k == kind:
                twin = n.replace("_stereo_", "_inertial_")
                if twin in by_run:
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
