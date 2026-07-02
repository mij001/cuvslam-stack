#!/usr/bin/env python3
"""screen.py — the DAMOV-GPU Step-1 memory-bound screen + stall breakdown.

Consumes a results dir from `profile.py --profiler ncu` (derived/ncu_metrics.csv)
and classifies every kernel: memory-bound / compute-leaning / mixed /
latency-underutilized, using the adapted DAMOV Step-1 rule from
Adapting_DAMOV_to_GPU.md — keep as memory-bound the kernels with
Memory-SoL ≫ Compute-SoL, OR a dominant memory stall (long-scoreboard).

Emits:
  screen.csv           per-kernel: SoL%, hit rates, stalls, coalescing, verdict
  fig_screen.svg       Mem-SoL vs Compute-SoL per kernel
  fig_stalls.svg       stall-reason breakdown per kernel (if stall metrics present)

Aggregation is time-weighted across launches of the same kernel (pyramid levels
etc. collapse into one row; per-launch spread is kept as min/max Mem-SoL).

Usage:  python -m analysis.screen <ncu_results_dir> [--out DIR]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common, stages, svgfig  # noqa: E402

M = {
    "time": "gpu__time_duration.sum",
    "mem_sol": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "comp_sol": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_sol": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_hit": "l1tex__t_sector_hit_rate.pct",
    "l2_hit": "lts__t_sector_hit_rate.pct",
    "occ": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sect_ld": "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "sect_st": "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio",
}

# stall metrics (warps stalled per issue-active cycle); ordered for the chart
STALLS = [
    ("long_scoreboard", "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", "memory (DRAM/L2 latency)"),
    ("short_scoreboard", "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio", "shared-mem latency"),
    ("lg_throttle", "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio", "LSU queue full"),
    ("mio_throttle", "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio", "MIO queue full"),
    ("tex_throttle", "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio", "TEX queue full"),
    ("math_throttle", "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", "math pipe busy"),
    ("barrier", "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio", "__syncthreads"),
    ("wait", "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio", "fixed-latency dep"),
    ("not_selected", "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio", "eligible, not picked"),
]

MEMORY_STALLS = {"long_scoreboard", "lg_throttle", "mio_throttle", "tex_throttle"}


def _tw(vals_times):
    """Time-weighted mean, skipping NaNs."""
    num = den = 0.0
    for v, t in vals_times:
        if v == v and t == t and t > 0:
            num += v * t
            den += t
    return num / den if den else float("nan")


def aggregate(launches):
    by = {}
    for lk in launches:
        by.setdefault(lk.kernel, []).append(lk)
    rows = []
    for kernel, lks in by.items():
        t = sum(l.m(M["time"], 0.0) for l in lks)
        row = {"kernel": kernel, "stage": stages.stage_of(kernel),
               "launches": len(lks), "time_s": t}
        for key in ("mem_sol", "comp_sol", "dram_sol", "l1_hit", "l2_hit",
                    "occ", "sect_ld", "sect_st"):
            row[key] = _tw([(l.m(M[key]), l.m(M["time"], 0.0)) for l in lks])
        mems = [l.m(M["mem_sol"]) for l in lks if l.m(M["mem_sol"]) == l.m(M["mem_sol"])]
        row["mem_sol_min"] = min(mems) if mems else float("nan")
        row["mem_sol_max"] = max(mems) if mems else float("nan")
        for name, metric, _ in STALLS:
            row[f"stall_{name}"] = _tw([(l.m(metric), l.m(M["time"], 0.0)) for l in lks])
        rows.append(row)
    rows.sort(key=lambda r: -r["time_s"])
    return rows


def verdict(row) -> str:
    mem, comp = row["mem_sol"], row["comp_sol"]
    stall_items = [(n, row.get(f"stall_{n}", float("nan"))) for n, _, _ in STALLS]
    stall_items = [(n, v) for n, v in stall_items if v == v]
    dominant = max(stall_items, key=lambda x: x[1])[0] if stall_items else None
    mem_stall_dominant = dominant in MEMORY_STALLS if dominant else False
    if mem != mem or comp != comp:
        return "no-data"
    if mem >= 40 and mem >= 1.5 * comp:
        return "memory-bound"
    if comp >= 40 and comp >= 1.5 * mem:
        return "compute-leaning"
    if mem < 40 and comp < 40:
        return "memory-latency" if mem_stall_dominant else "underutilized"
    return "mixed"


def emit(rows, out_dir, hw=None):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    headers = ["kernel", "stage", "verdict", "launches", "time_ms", "mem_sol_pct",
               "comp_sol_pct", "dram_sol_pct", "l1_hit_pct", "l2_hit_pct",
               "occupancy_pct", "sectors_per_req_ld", "sectors_per_req_st"] + \
              [f"stall_{n}" for n, _, _ in STALLS]
    table = []
    for r in rows:
        r["verdict"] = verdict(r)
        table.append([r["kernel"], r["stage"], r["verdict"], r["launches"],
                      round(r["time_s"] * 1e3, 4)] +
                     [round(r[k], 2) if r[k] == r[k] else "" for k in
                      ("mem_sol", "comp_sol", "dram_sol", "l1_hit", "l2_hit",
                       "occ", "sect_ld", "sect_st")] +
                     [round(r[f"stall_{n}"], 3) if r[f"stall_{n}"] == r[f"stall_{n}"] else ""
                      for n, _, _ in STALLS])
    p = os.path.join(out_dir, "screen.csv")
    common.write_csv(p, headers, table)
    written.append(p)

    # Mem vs Compute SoL chart (grouped bars per kernel, top 14 by time)
    top = rows[:14]
    labels = [r["kernel"][:34] for r in top]
    mem = [r["mem_sol"] for r in top]
    ann = [f'{r["mem_sol"]:.0f}% mem / {r["comp_sol"]:.0f}% comp → {r["verdict"]}'
           if r["mem_sol"] == r["mem_sol"] else "no data" for r in top]
    colors = [svgfig.stage_color(r["stage"], stages.ORDER) for r in top]
    p = os.path.join(out_dir, "fig_screen.svg")
    svgfig.hbar(p, "DAMOV Step-1 screen — Memory SoL% (label: verdict)",
                labels, mem, unit="%", colors=colors, annotations=ann)
    written.append(p)

    # stall breakdown (only if stall metrics were collected)
    have_stalls = [n for n, _, _ in STALLS
                   if any(r[f"stall_{n}"] == r[f"stall_{n}"] for r in rows)]
    if have_stalls:
        matrix = [[r[f"stall_{n}"] if r[f"stall_{n}"] == r[f"stall_{n}"] else 0.0
                   for n in have_stalls] for r in top]
        p = os.path.join(out_dir, "fig_stalls.svg")
        svgfig.stacked_hbar(p, "Warp-stall breakdown (warps stalled per issue-active cycle)",
                            labels, have_stalls, matrix)
        written.append(p)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    csv_path = common.find_derived(args.run_dir, "ncu_metrics.csv")
    if not csv_path:
        raise SystemExit(f"no ncu_metrics.csv under {args.run_dir}/derived")
    rows = aggregate(common.load_ncu_csv(csv_path))
    out = args.out or os.path.join(args.run_dir, "derived")
    for p in emit(rows, out):
        print(f"[✓] {p}")


if __name__ == "__main__":
    main()
