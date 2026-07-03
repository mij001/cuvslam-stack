#!/usr/bin/env python3
"""transfers.py — host↔device data movement from the nsys memop summaries.

DAMOV's function-level CPU view underweights a class of data movement that is
first-order on GPUs: explicit host↔device copies (Adapting_DAMOV_to_GPU §9).
For a V-SLAM pipeline this is the per-frame image upload — traffic that a
near-sensor / ISP substrate would eliminate entirely, so it belongs in the
PiM evidence, not a footnote.

Parses the `cuda_gpu_mem_time_sum` and `cuda_gpu_mem_size_sum` tables that
profile.py's nsys post-processing already writes into derived/nsys_stats.txt
(no re-export needed; works on committed text), joins them per operation, and
relates transfer time to total kernel time from the kernel summary.

Emits: transfers.csv + a summary dict (used by make_report §5).

Usage:  python3 -m analysis.transfers <nsys_results_dir> [--out DIR]
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import build_dag, common  # noqa: E402

_ROW = re.compile(r"^\s*([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)\s+.*\[(CUDA [^\]]+)\]\s*$")


def _num(s):
    return float(s.replace(",", ""))


def _parse_table(lines, start):
    """Rows until the first blank line; returns {operation: (col1, col2, col3)}."""
    out = {}
    for line in lines[start:]:
        if not line.strip():
            break
        m = _ROW.match(line)
        if m:
            out[m.group(4)] = (_num(m.group(1)), _num(m.group(2)), _num(m.group(3)))
    return out


def parse(run_dir: str) -> dict:
    stats = common.find_derived(run_dir, "nsys_stats.txt")
    if not stats:
        raise SystemExit(f"no nsys_stats.txt under {run_dir}/derived")
    lines = open(stats, errors="replace").read().splitlines()
    time_tab, size_tab = {}, {}
    for i, line in enumerate(lines):
        if "MemOps Summary (by Time)" in line:
            time_tab = _parse_table(lines, i + 4)
        elif "MemOps Summary (by Size)" in line:
            size_tab = _parse_table(lines, i + 4)

    ops = {}
    for op, (pct, total_ns, count) in time_tab.items():
        ops[op] = {"op": op, "time_ms": total_ns / 1e6, "count": int(count),
                   "mb": float("nan")}
    for op, (total_mb, count, _avg) in size_tab.items():
        ops.setdefault(op, {"op": op, "time_ms": float("nan"),
                            "count": int(count), "mb": float("nan")})
        ops[op]["mb"] = total_mb

    dag = build_dag.build(run_dir)
    frames = dag["frames"]
    kernel_ms = dag["total_gpu_ms"]
    total_xfer_ms = sum(o["time_ms"] for o in ops.values()
                        if o["time_ms"] == o["time_ms"])
    h2d = ops.get("CUDA memcpy Host-to-Device", {})
    return {
        "run_dir": run_dir, "frames": frames, "ops": list(ops.values()),
        "kernel_time_ms": kernel_ms, "transfer_time_ms": total_xfer_ms,
        "transfer_vs_kernel_pct": 100.0 * total_xfer_ms / kernel_ms if kernel_ms else float("nan"),
        "h2d_mb_per_frame": (h2d.get("mb", float("nan")) / frames) if frames and h2d else float("nan"),
    }


def emit(t: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for o in sorted(t["ops"], key=lambda o: -(o["time_ms"] if o["time_ms"] == o["time_ms"] else 0)):
        rows.append([o["op"], round(o["time_ms"], 2), o["count"],
                     round(o["mb"], 2) if o["mb"] == o["mb"] else "",
                     round(o["mb"] / t["frames"], 4) if t["frames"] and o["mb"] == o["mb"] else ""])
    p = os.path.join(out_dir, "transfers.csv")
    common.write_csv(p, ["operation", "total_time_ms", "count", "total_mb",
                         "mb_per_frame"], rows)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    t = parse(args.run_dir)
    p = emit(t, args.out or os.path.join(args.run_dir, "derived"))
    print(f"[✓] {p}")
    print(f"transfers {t['transfer_time_ms']:.1f} ms vs kernels {t['kernel_time_ms']:.1f} ms "
          f"({t['transfer_vs_kernel_pct']:.0f}%); H2D {t['h2d_mb_per_frame']:.2f} MB/frame")


if __name__ == "__main__":
    main()
