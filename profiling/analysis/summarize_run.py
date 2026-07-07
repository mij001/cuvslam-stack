#!/usr/bin/env python3
"""summarize_run.py — emit THE standard evidence schema for one profiled run.

Every adapter's workload, profiled by the harness, reduces to one
`summary.json` — the contract between the profiler and every consumer
(the dashboard's evidence explorer, cross-run substrate dynamics, papers):

{
  "workload":  "<tag>",            "device": "<hw>",
  "adapter":   "cuvslam|command",  "source": "<report/results dir>",
  "qor":       {...} | null,       # accuracy / quality-of-result, if measured
  "stages":    [{"name", "time_ms", "share_pct", "n_kernels"}],
  "kernels":   [{"name", "stage", "time_ms", "share_pct",
                 "limiter",              # what bounds it (taxonomy class)
                 "substrate",            # best-substrate verdict (GPU/CPU/PiM/ISP)
                 "pim_affinity", "rationale",
                 "evidence": {"dram_sol_pct", "sectors_per_req", "lfmr",
                              "mpki", "occupancy_pct", "ai", "dominant_stall"},
                 "roofline": {"ai", "gflops"} | null}]
}

The EVIDENCE fields are the same numbers a human used to reach the study's
conclusions (screen -> classify -> attribute -> verdict); the dashboard
renders them against the decision thresholds so the reasoning is inspectable
per kernel.

Inputs (either):
  --legacy  profiling/reports/<device report>     (data/classification.csv +
            data/dag_stages.csv + data/roofline.csv — the initial studies)
  --results profiling/results/<run dir>           (derived/ of a harness run)

Usage:
  python3 profiling/analysis/summarize_run.py --all-legacy
  python3 profiling/analysis/summarize_run.py --results profiling/results/<run>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fnum(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def build_summary(data_dir, label_dir, adapter="cuvslam", qor=None):
    """Reshape classification/dag/dag_stages/roofline CSVs into the schema."""
    cls = read_csv(os.path.join(data_dir, "classification.csv"))
    if not cls:
        return None
    stages = read_csv(os.path.join(data_dir, "dag_stages.csv"))
    roof = {r["kernel"]: r for r in read_csv(os.path.join(data_dir, "roofline.csv"))}
    # TIME shares come from the nsys timeline (dag.csv) — the whole-run truth.
    # classification.csv times are per-capture-window (ncu) and NOT comparable
    # across kernels captured in different windows.
    dag = {r["kernel"]: r for r in read_csv(os.path.join(data_dir, "dag.csv"))}

    def timeline(kernel, field, fallback):
        row = dag.get(kernel)
        return fnum(row.get(field), fallback) if row else fallback

    kernels = []
    for r in cls:
        rf = roof.get(r["kernel"])
        kernels.append({
            "name": r["kernel"], "stage": r.get("stage", "?"),
            "time_ms": round(timeline(r["kernel"], "total_ms", fnum(r["time_ms"], 0)), 3),
            "share_pct": round(timeline(r["kernel"], "pct_gpu_time", 0.0), 2),
            "limiter": r.get("class", "?"),
            "substrate": r.get("substrate", "?"),
            "pim_affinity": r.get("pim_affinity", "?"),
            "rationale": r.get("rationale", ""),
            "evidence": {
                "dram_sol_pct": fnum(r.get("dram_sol_pct")),
                "sectors_per_req": fnum(r.get("sectors_per_req")),
                "lfmr": fnum(r.get("lfmr_gpu")),
                "mpki": fnum(r.get("mpki_gpu")),
                "occupancy_pct": fnum(r.get("occupancy_pct")),
                "ai": fnum(r.get("ai_flop_per_byte")),
                "dominant_stall": r.get("dominant_stall", ""),
            },
            "roofline": ({"ai": fnum(rf.get("ai_dram")), "gflops": fnum(rf.get("gflops"))}
                         if rf else None),
        })
    kernels.sort(key=lambda k: -k["share_pct"])

    name = os.path.basename(label_dir.rstrip("/"))
    return {
        "workload": name,
        "device": name.split("_")[-1] if "_" in name else "?",
        "adapter": adapter,
        "source": os.path.relpath(label_dir, REPO),
        "qor": qor,
        "stages": [{
            "name": s["stage"],
            "time_ms": round(fnum(s.get("total_ms"), 0), 3),
            "share_pct": round(fnum(s.get("pct_gpu_time"), 0), 2),
            "n_kernels": len((s.get("kernels") or "").split()),
        } for s in stages],
        "kernels": kernels,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legacy", help="one device-report dir (uses its data/)")
    ap.add_argument("--results", help="one harness results dir (uses its derived/)")
    ap.add_argument("--all-legacy", action="store_true",
                    help="every profiling/reports/* with data/classification.csv")
    args = ap.parse_args()

    todo = []
    if args.legacy:
        todo.append((os.path.join(args.legacy, "data"), args.legacy))
    if args.results:
        todo.append((os.path.join(args.results, "derived"), args.results))
    if args.all_legacy:
        for d in sorted(glob.glob(os.path.join(REPO, "profiling/reports/*"))):
            if os.path.isfile(os.path.join(d, "data", "classification.csv")):
                todo.append((os.path.join(d, "data"), d))
    if not todo:
        ap.error("give --legacy, --results, or --all-legacy")

    for data_dir, label_dir in todo:
        s = build_summary(data_dir, label_dir)
        if not s:
            print(f"[skip] {label_dir}: no classification data")
            continue
        out = os.path.join(label_dir, "summary.json")
        json.dump(s, open(out, "w"), indent=1)
        print(f"[✓] {os.path.relpath(out, REPO)}  "
              f"({len(s['kernels'])} kernels, {len(s['stages'])} stages)")


if __name__ == "__main__":
    main()
