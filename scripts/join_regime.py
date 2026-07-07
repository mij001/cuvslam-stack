#!/usr/bin/env python3
"""join_regime.py — parallel JOIN stage of the validation regime.

Takes the per-cell ledger the campaign driver wrote and fans out over all
CPU cores to merge every cell's artifacts into analysis-ready tables:

  regime_matrix.csv        config × mode -> APE / delta / status
                           (wide: one row per config, one column set per mode)
  kernel_metrics_long.csv  one row per (config, kernel) from every ncu cell's
                           derived/ncu_metrics.csv — the feature table the
                           substrate-dynamics analysis consumes
  nvbit_traces.csv         index of captured memory traces (config -> trace,
                           size, launch window) for the locality analyses

The heavy lifting (parsing ncu CSVs, stat-ing traces, reading evals) is
per-cell independent, so it runs on a process pool (--jobs 0 = all cores).

Usage:
  python3 scripts/join_regime.py --ledger /mnt/data/validation_regime_out/REGIME.tsv \
      --out /mnt/data/validation_regime_out [--jobs 0]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_ledger(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


# ── per-cell workers (process pool) ──────────────────────────────────────────
def join_ncu_cell(cell):
    """(config, results_dir) -> list of kernel-metric rows."""
    tag, rdir = cell
    path = os.path.join(REPO, rdir, "derived", "ncu_metrics.csv")
    if not os.path.isfile(path):
        # results dir may be recorded absolute
        path = os.path.join(rdir, "derived", "ncu_metrics.csv")
        if not os.path.isfile(path):
            return []
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            r["config"] = tag
            rows.append(r)
    return rows


def join_nvbit_cell(cell):
    """(config, results_dir) -> trace index row (or None)."""
    tag, rdir = cell
    base = rdir if os.path.isabs(rdir) else os.path.join(REPO, rdir)
    hits = glob.glob(os.path.join(base, "raw", "mem_trace.*"))
    if not hits:
        return None
    t = hits[0]
    return {"config": tag, "trace": t, "size_bytes": os.path.getsize(t),
            "results_dir": rdir}


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=0, help="0 = all cores")
    args = ap.parse_args()
    jobs = args.jobs or os.cpu_count() or 4
    rows = read_ledger(args.ledger)

    # 1) wide accuracy matrix: config × mode
    by_cfg: dict[str, dict] = {}
    modes = ["plain", "nsys", "ncu", "nvbit"]
    for r in rows:
        c = by_cfg.setdefault(r["config"], {"config": r["config"]})
        m = r["mode"]
        c[f"{m}_APE_m"] = r["mode_APE_m"]
        c[f"{m}_delta_m"] = r["delta_m"]
        c[f"{m}_status"] = r["status"]
    matrix_path = os.path.join(args.out, "regime_matrix.csv")
    cols = ["config"] + [f"{m}_{k}" for m in modes for k in ("APE_m", "delta_m", "status")]
    with open(matrix_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for c in sorted(by_cfg):
            w.writerow(by_cfg[c])

    # 2) kernel metrics long table (ncu cells) — parallel parse
    ncu_cells = [(r["config"], r["results_dir"]) for r in rows
                 if r["mode"] == "ncu" and r.get("results_dir", "-") not in ("-", "")]
    kern_rows = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for chunk in ex.map(join_ncu_cell, ncu_cells, chunksize=4):
            kern_rows.extend(chunk)
    klong_path = os.path.join(args.out, "kernel_metrics_long.csv")
    if kern_rows:
        cols = ["config"] + [k for k in kern_rows[0] if k != "config"]
        with open(klong_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(kern_rows)

    # 3) nvbit trace index — parallel stat
    nvb_cells = [(r["config"], r["results_dir"]) for r in rows
                 if r["mode"] == "nvbit" and r.get("results_dir", "-") not in ("-", "")]
    traces = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for t in ex.map(join_nvbit_cell, nvb_cells, chunksize=4):
            if t:
                traces.append(t)
    tr_path = os.path.join(args.out, "nvbit_traces.csv")
    if traces:
        with open(tr_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(traces[0]))
            w.writeheader()
            w.writerows(traces)

    print(f"[join] {len(by_cfg)} configs -> regime_matrix.csv; "
          f"{len(kern_rows)} kernel rows -> kernel_metrics_long.csv "
          f"({len(ncu_cells)} ncu cells, {jobs} workers); "
          f"{len(traces)} traces -> nvbit_traces.csv")


if __name__ == "__main__":
    main()
