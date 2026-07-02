#!/usr/bin/env python3
"""bandwidth.py — per-stage DRAM traffic and achieved bandwidth.

Combines the ncu per-kernel DRAM byte counters with the stage taxonomy to
produce the "slide-2 chart": which pipeline stages move the most DRAM data,
and at what fraction of the hardware's bandwidth ceiling they run.

Two views:
  * per-stage DRAM bytes per profiled kernel-launch window (what moves the data)
  * per-kernel achieved GB/s vs the hw ceiling (who saturates the bus)

If the nsys run of the same workload is supplied, per-frame traffic is
extrapolated: bytes/launch (ncu) × launches/frame (nsys).

Emits:  bandwidth.csv, fig_stage_bytes.svg, fig_kernel_gbps.svg

Usage:  python -m analysis.bandwidth <ncu_results_dir> --hw hw/<gpu>.toml
            [--nsys <nsys_results_dir>] [--out DIR]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import build_dag, common, stages, svgfig  # noqa: E402

TIME = "gpu__time_duration.sum"
DR, DW = "dram__bytes_read.sum", "dram__bytes_write.sum"


def aggregate(launches):
    per_kernel = {}
    for lk in launches:
        r = per_kernel.setdefault(lk.kernel, {"kernel": lk.kernel,
                                              "stage": stages.stage_of(lk.kernel),
                                              "launches": 0, "time_s": 0.0,
                                              "read": 0.0, "write": 0.0})
        r["launches"] += 1
        for key, metric in (("time_s", TIME), ("read", DR), ("write", DW)):
            v = lk.m(metric)
            if v == v:
                r[key] += v
    rows = sorted(per_kernel.values(), key=lambda r: -(r["read"] + r["write"]))
    for r in rows:
        total = r["read"] + r["write"]
        r["bytes"] = total
        r["gbps"] = total / r["time_s"] / 1e9 if r["time_s"] else float("nan")
        r["bytes_per_launch"] = total / r["launches"] if r["launches"] else float("nan")
        r["rw_ratio"] = r["read"] / r["write"] if r["write"] else float("inf")
    return rows


def emit(rows, hw, out_dir, nsys_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    written = []

    # optional per-frame extrapolation from the nsys run
    launches_per_frame = {}
    if nsys_dir:
        try:
            dag = build_dag.build(nsys_dir)
            frames = dag["frames"]
            if frames:
                for k in dag["per_kernel"]:
                    launches_per_frame[k["kernel"]] = k["instances"] / frames
        except SystemExit:
            pass

    headers = ["kernel", "stage", "launches", "time_ms", "dram_read_bytes",
               "dram_write_bytes", "rw_ratio", "achieved_gbps",
               "bytes_per_launch", "est_bytes_per_frame"]
    table = []
    for r in rows:
        est = ""
        lpf = launches_per_frame.get(r["kernel"])
        if lpf and r["bytes_per_launch"] == r["bytes_per_launch"]:
            est = f"{r['bytes_per_launch'] * lpf:.6g}"
            r["est_bytes_per_frame"] = r["bytes_per_launch"] * lpf
        table.append([r["kernel"], r["stage"], r["launches"],
                      round(r["time_s"] * 1e3, 4),
                      f"{r['read']:.6g}", f"{r['write']:.6g}",
                      round(r["rw_ratio"], 2) if r["rw_ratio"] == r["rw_ratio"] and r["rw_ratio"] != float("inf") else "inf",
                      round(r["gbps"], 2) if r["gbps"] == r["gbps"] else "",
                      f"{r['bytes_per_launch']:.6g}", est])
    p = os.path.join(out_dir, "bandwidth.csv")
    common.write_csv(p, headers, table)
    written.append(p)

    # per-stage bytes
    per_stage = {}
    for r in rows:
        per_stage[r["stage"]] = per_stage.get(r["stage"], 0.0) + r["bytes"]
    st = [(s, per_stage[s]) for s in stages.ORDER if per_stage.get(s)]
    if st:
        labels = [s for s, _ in st]
        vals = [b / 1e6 for _, b in st]
        colors = [svgfig.stage_color(s, stages.ORDER) for s in labels]
        ann = [f"{v:.1f} MB ({stages.persistence_of(s)})" for (s, _), v in zip(st, vals)]
        p = os.path.join(out_dir, "fig_stage_bytes.svg")
        svgfig.hbar(p, "DRAM traffic by stage (profiled launch window)",
                    labels, vals, unit="MB", colors=colors, annotations=ann)
        written.append(p)

    # per-kernel achieved GB/s vs ceiling
    memc = hw.get("memory", {})
    bw = float(memc.get("dram_gbps_measured") or 0.0) or \
        float(memc.get("dram_gbps_theoretical") or 0.0)
    top = [r for r in rows if r["gbps"] == r["gbps"]][:14]
    if top and bw:
        labels = [r["kernel"][:34] for r in top]
        vals = [r["gbps"] for r in top]
        colors = [svgfig.stage_color(r["stage"], stages.ORDER) for r in top]
        ann = [f"{v:.1f} GB/s = {100*v/bw:.0f}% of {bw:g}" for v in vals]
        p = os.path.join(out_dir, "fig_kernel_gbps.svg")
        svgfig.hbar(p, "Achieved DRAM bandwidth per kernel (vs hw ceiling)",
                    labels, vals, unit="GB/s", colors=colors, annotations=ann)
        written.append(p)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--hw", required=True)
    ap.add_argument("--nsys", default=None, help="matching nsys results dir (per-frame extrapolation)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    csv_path = common.find_derived(args.run_dir, "ncu_metrics.csv")
    if not csv_path:
        raise SystemExit(f"no ncu_metrics.csv under {args.run_dir}/derived")
    rows = aggregate(common.load_ncu_csv(csv_path))
    out = args.out or os.path.join(args.run_dir, "derived")
    for p in emit(rows, common.load_hw(args.hw), out, nsys_dir=args.nsys):
        print(f"[✓] {p}")


if __name__ == "__main__":
    main()
