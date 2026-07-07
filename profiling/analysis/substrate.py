#!/usr/bin/env python3
"""substrate.py — heterogeneous-substrate candidacy + its dynamics.

Consumes the per-kernel classification tables the profiling regime produces
(classification.csv — one per workload/config: DAMOV-GPU class, persistence,
PiM affinity, and the raw decision features) and answers the placement
question the characterization exists for:

  1. VERDICT   per kernel × workload: which substrate suits it best —
                 GPU-keep            (compute/latency bound, well-coalesced)
                 GPU+layout-fix      (scatter that a data-layout change fixes)
                 PiM-near-bank       (bandwidth-bound streaming over DRAM)
                 PiM-scatter         (irregular gather/scatter engines)
                 ISP/near-storage    (cold-persistent scans, e.g. keyframe DB)
                 CPU/host            (tiny, low-occupancy kernels not worth a GPU launch)
     — a transparent RULE over (class, persistence, affinity, features), with
     the driving feature recorded per verdict.

  2. DYNAMICS  the same kernel across workloads/configs/sequences: does the
     verdict FLIP when the dataset or a cuVSLAM feature toggle changes the
     dynamic metrics (DRAM SoL, sectors/request, footprint, time share)?
     Flips are reported with the feature that moved most — these are exactly
     the kernels whose placement must be decided at runtime or per-product.

  3. MIX       time-weighted substrate mix per workload — how much of the GPU
     second would move to each substrate.

Inputs: any number of classification.csv paths (device reports `data/` dirs,
regime `derived/` dirs), plus optionally a campaign class_agreement.csv for
per-sequence class stability. Outputs (CSV): substrate_verdicts, substrate_mix,
substrate_flips (+ a printed summary).

Usage:
  python3 profiling/analysis/substrate.py profiling/reports/*/data/classification.csv \
      --agreement profiling/reports/2026-07-04_campaign/class_agreement.csv \
      --out reports/<date>_substrate/
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import defaultdict

FEATURES = ["dram_sol_pct", "sectors_per_req", "lfmr_gpu", "mpki_gpu",
            "occupancy_pct", "ai_flop_per_byte", "time_ms"]


def fnum(v, default=math.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def verdict(row):
    """(substrate, driving_reason) from one classification row."""
    aff = (row.get("pim_affinity") or "").strip()
    sub = (row.get("substrate") or "").lower()
    pers = (row.get("persistence") or "").strip()
    sec = fnum(row.get("sectors_per_req"))
    sol = fnum(row.get("dram_sol_pct"))
    occ = fnum(row.get("occupancy_pct"))
    t = fnum(row.get("time_ms"))

    if aff == "strong":
        if "isp" in sub or "storage" in sub or pers == "cold-persistent":
            return "ISP/near-storage", f"strong affinity, {pers} data (DB-scan shape)"
        return "PiM-near-bank", f"strong affinity, DRAM SoL {sol:.0f}%"
    if aff == "conditional":
        if sec == sec and sec >= 12:
            return "PiM-scatter", f"{sec:.0f} sectors/request (4 = coalesced)"
        if "layout" in sub:
            return "GPU+layout-fix", "scatter fixable by data layout"
        return "PiM-near-bank", f"conditional affinity, DRAM SoL {sol:.0f}%"
    # affinity none/weak → GPU unless the kernel is too small to earn a launch
    if occ == occ and t == t and occ < 8 and t < 1.0:
        return "CPU/host", f"occupancy {occ:.0f}%, {t:.2f} ms total — launch overhead territory"
    return "GPU-keep", "compute/latency bound or well-coalesced"


def load(paths):
    """[(workload_tag, {kernel: row})] — tag derived from the csv's location."""
    out = []
    for p in paths:
        tag = re.sub(r"^\d{4}-\d{2}-\d{2}_", "",
                     os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(p)))))
        rows = {}
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                rows[r["kernel"]] = r
        out.append((tag, rows))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("classifications", nargs="+", help="classification.csv path(s)")
    ap.add_argument("--agreement", default=None,
                    help="campaign class_agreement.csv for per-sequence stability")
    ap.add_argument("--out", default="reports/substrate")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    data = load(args.classifications)

    # ── 1. verdicts ──────────────────────────────────────────────────────────
    verdicts = defaultdict(dict)     # kernel -> workload -> (substrate, why, row)
    for tag, rows in data:
        for k, r in rows.items():
            verdicts[k][tag] = (*verdict(r), r)
    with open(os.path.join(args.out, "substrate_verdicts.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kernel", "workload", "substrate", "driving_reason"] + FEATURES)
        for k in sorted(verdicts):
            for tag, (s, why, r) in sorted(verdicts[k].items()):
                w.writerow([k, tag, s, why] + [r.get(f, "") for f in FEATURES])

    # ── 2. dynamics: verdict flips across workloads ──────────────────────────
    flips = []
    for k, per in sorted(verdicts.items()):
        subs = {s for s, _w, _r in per.values()}
        if len(per) >= 2 and len(subs) > 1:
            # which feature moved the most between the flipping workloads?
            tags = sorted(per)
            drive, dmax = "", 0.0
            for f in FEATURES[:-1]:            # time_ms is a share, not a driver
                vals = [fnum(per[t][2].get(f)) for t in tags]
                vals = [v for v in vals if v == v]
                if len(vals) >= 2 and min(vals) > 0:
                    rel = max(vals) / min(vals)
                    if rel > dmax:
                        dmax, drive = rel, f
            flips.append({"kernel": k,
                          "verdicts": " | ".join(f"{t}:{per[t][0]}" for t in tags),
                          "most_moved_feature": drive,
                          "max_over_min": f"{dmax:.1f}x" if dmax else ""})
    with open(os.path.join(args.out, "substrate_flips.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["kernel", "verdicts", "most_moved_feature", "max_over_min"])
        w.writeheader()
        w.writerows(flips)

    # ── 3. time-weighted mix per workload ────────────────────────────────────
    with open(os.path.join(args.out, "substrate_mix.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["workload", "substrate", "time_ms", "time_pct"])
        for tag, rows in data:
            mix = defaultdict(float)
            for k, r in rows.items():
                mix[verdicts[k][tag][0]] += fnum(r.get("time_ms"), 0)
            total = sum(mix.values()) or 1
            for s, t in sorted(mix.items(), key=lambda kv: -kv[1]):
                w.writerow([tag, s, f"{t:.1f}", f"{t / total * 100:.1f}"])

    # ── optional: per-sequence class stability from the campaign ─────────────
    if args.agreement:
        stable = flipping = 0
        with open(args.agreement, newline="") as fh:
            for r in csv.DictReader(fh):
                if (r.get("agreement") or "") == "unanimous":
                    stable += 1
                else:
                    flipping += 1
        with open(os.path.join(args.out, "NOTES.txt"), "w") as fh:
            fh.write(f"class stability across sequences: {stable} unanimous, "
                     f"{flipping} with per-sequence variation (see class_agreement.csv)\n")

    # ── summary ──────────────────────────────────────────────────────────────
    n_k = len(verdicts)
    print(f"[substrate] {n_k} kernels across {len(data)} workloads -> {args.out}")
    print(f"[substrate] verdict flips across workloads: {len(flips)}")
    for f in flips[:10]:
        print(f"  flip: {f['kernel'][:44]:44} {f['verdicts']}"
              f"  (moved: {f['most_moved_feature']} {f['max_over_min']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
