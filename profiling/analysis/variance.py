#!/usr/bin/env python3
"""variance.py — run-to-run variability across repeated captures.

Single-run numbers are not publishable: top-tier reviewers expect repeats and
a dispersion statistic. This module takes N results dirs of the SAME workload
and reports, per kernel, the coefficient of variation (CoV = stddev/mean):

  * nsys dirs → CoV of total kernel time and of instance counts (also catches
    workload nondeterminism: if instance counts vary, the workload itself is
    not replaying identically);
  * ncu dirs  → CoV of kernel time and of Memory-SoL% (counter stability).

Emits variance.csv + a one-line verdict per input set. Rule of thumb used in
the report: CoV ≤ 5% supports two-significant-figure claims; kernels above
10% are flagged.

Usage:
  python3 -m analysis.variance <dir1> <dir2> [...] [--out DIR]
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common, screen  # noqa: E402


def _cov(vals):
    vals = [v for v in vals if v == v]
    if len(vals) < 2 or not statistics.mean(vals):
        return float("nan")
    return statistics.stdev(vals) / statistics.mean(vals)


def collect(dirs):
    """kernel -> {metric -> [one value per run]} plus which runs saw it.

    Records both absolute time and per-run TIME SHARE. On hosts that cannot
    lock clocks (laptops), absolute kernel time tracks the DVFS state, not the
    kernel — measured 3.4× swings across identical back-to-back runs on the
    MX450 — while shares cancel the global clock factor. Shares are the
    laptop-defensible statistic; absolute times need locked clocks.
    """
    per_kernel: dict[str, dict] = {}
    kind = None
    for d in dirs:
        ncu_csv = common.find_derived(d, "ncu_metrics.csv")
        ks_csv = common.find_derived(d, "kern_sum.csv") or \
            common.find_derived(d, "cuda_gpu_kern_sum.csv")
        if ncu_csv:
            kind = kind or "ncu"
            rows = screen.aggregate(common.load_ncu_csv(ncu_csv))
            total = sum(r["time_s"] for r in rows if r["time_s"] == r["time_s"]) or 1.0
            for r in rows:
                e = per_kernel.setdefault(r["kernel"], {"time": [], "share": [],
                                                        "mem_sol": [], "instances": [],
                                                        "runs": 0})
                e["time"].append(r["time_s"])
                e["share"].append(r["time_s"] / total)
                e["mem_sol"].append(r["mem_sol"])
                e["instances"].append(r["launches"])
                e["runs"] += 1
        elif ks_csv:
            kind = kind or "nsys"
            ks = common.load_kern_sum(ks_csv)
            total = sum(k.total_ns for k in ks) or 1.0
            for k in ks:
                e = per_kernel.setdefault(k.kernel, {"time": [], "share": [],
                                                     "mem_sol": [], "instances": [],
                                                     "runs": 0})
                e["time"].append(k.total_ns / 1e9)
                e["share"].append(k.total_ns / total)
                e["instances"].append(k.instances)
                e["runs"] += 1
        else:
            raise SystemExit(f"{d}: no derived kernel table found")
    return per_kernel, kind


def emit(per_kernel, kind, n_runs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for kernel, e in sorted(per_kernel.items(),
                            key=lambda kv: -statistics.mean(kv[1]["time"] or [0])):
        rows.append([kernel, e["runs"],
                     round(statistics.mean(e["time"]) * 1e3, 4),
                     round(100 * _cov(e["time"]), 2) if _cov(e["time"]) == _cov(e["time"]) else "",
                     round(100 * statistics.mean(e["share"]), 3),
                     round(100 * _cov(e["share"]), 2) if _cov(e["share"]) == _cov(e["share"]) else "",
                     round(100 * _cov(e["instances"]), 2) if _cov(e["instances"]) == _cov(e["instances"]) else "",
                     round(100 * _cov(e["mem_sol"]), 2) if kind == "ncu" and _cov(e["mem_sol"]) == _cov(e["mem_sol"]) else ""])
    p = os.path.join(out_dir, "variance.csv")
    common.write_csv(p, ["kernel", "runs_seen", "mean_time_ms", "time_cov_pct",
                         "mean_share_pct", "share_cov_pct",
                         "instances_cov_pct", "mem_sol_cov_pct"], rows)

    time_covs = [r[3] for r in rows if isinstance(r[3], float)]
    share_covs = [r[5] for r in rows if isinstance(r[5], float)]
    inst_covs = [r[6] for r in rows if isinstance(r[6], float)]
    missing = [r[0] for r in rows if r[1] != n_runs]
    summary = {
        "kind": kind, "n_runs": n_runs, "n_kernels": len(rows),
        "median_time_cov_pct": round(statistics.median(time_covs), 2) if time_covs else None,
        "max_time_cov_pct": round(max(time_covs), 2) if time_covs else None,
        "median_share_cov_pct": round(statistics.median(share_covs), 2) if share_covs else None,
        "max_share_cov_pct": round(max(share_covs), 2) if share_covs else None,
        "kernels_share_cov_gt10pct": [r[0] for r in rows if isinstance(r[5], float) and r[5] > 10],
        "deterministic_instance_counts": bool(inst_covs) and max(inst_covs) == 0.0,
        "max_instances_cov_pct": round(max(inst_covs), 2) if inst_covs else None,
        "kernels_not_in_every_run": missing,
    }
    return p, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=".")
    args = ap.parse_args(argv)
    per_kernel, kind = collect(args.dirs)
    p, s = emit(per_kernel, kind, len(args.dirs), args.out)
    print(f"[✓] {p}")
    print(f"{s['kind']}: {s['n_runs']} runs, {s['n_kernels']} kernels")
    print(f"  absolute time CoV: median {s['median_time_cov_pct']}%, max "
          f"{s['max_time_cov_pct']}%   (tracks DVFS on unlocked clocks)")
    print(f"  time-SHARE  CoV: median {s['median_share_cov_pct']}%, max "
          f"{s['max_share_cov_pct']}%   (clock-invariant statistic)")
    print(f"  instance counts deterministic: {s['deterministic_instance_counts']}"
          + ("" if s["deterministic_instance_counts"]
             else f" (max CoV {s['max_instances_cov_pct']}%)"))
    if s["kernels_share_cov_gt10pct"]:
        print(f"[!] share CoV>10%: {', '.join(s['kernels_share_cov_gt10pct'][:8])}")
    if s["kernels_not_in_every_run"]:
        print(f"[!] not in every run: {', '.join(s['kernels_not_in_every_run'][:8])}")


if __name__ == "__main__":
    main()
