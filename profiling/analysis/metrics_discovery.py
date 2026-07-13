#!/usr/bin/env python3
"""metrics_discovery.py — hunting the next LFMR.

DAMOV's decisive move was INVENTING a metric (LFMR) because existing ones
could not separate its classes — and its class definitions also leaned on a
RESPONSE metric (how LFMR trends under the core sweep). This script does the
same hunt on the GPU population:

candidate metrics (beyond the classifier's current inputs):
  S_core, S_mem   the clock-domain response (the sweep IS a metric — promoted
                  to a first-class feature, as the user's mandate suggests)
  DSA             domain-sensitivity angle atan2(S_core-1, S_mem-1): one number
                  for WHICH wall a kernel leans on
  QSR             queue-stall ratio (lg_throttle+mio_throttle)/long_scoreboard:
                  request-CONCURRENCY-bound vs latency-bound — the distinction
                  the population's L2-scatter finding exposed inside "memory"
  L2AMP           dram_bytes/L2_bytes: how much traffic the L2 filters
                  (the hierarchical-reuse axis; ~1 = L2 transparent, ~0 = L2
                  absorbs everything)

evaluation, DAMOV-style (does a metric EARN its place?):
  * one-way F-statistic per feature across the assigned classes
    (between-class variance / within-class variance — higher = separates)
  * k-means ablation at k=8: purity/silhouette with BASE features vs
    BASE+static-candidates vs BASE+response vs ALL

Usage:
  python3 -m analysis.metrics_discovery --data <population_out> \
      [--out ../reports/2026-07-12_gpu_damov_population]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import cluster, common  # noqa: E402
from analysis.population import read_kernsum, short  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

M = {
    "mem_sol": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "comp_sol": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_sol": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_hit": "lts__t_sector_hit_rate.pct",
    "occ": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sect_ld": "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "sect_st": "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio",
    "dram_rd": "dram__bytes_read.sum",
    "dram_wr": "dram__bytes_write.sum",
    "lts_bytes": "lts__t_bytes.sum",
    "st_long": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "st_lg": "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "st_mio": "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "time": "gpu__time_duration.sum",
}

BASE_F = ["mem_sol", "comp_sol", "dram_sol", "lfmr", "occ", "log_sect"]
NEW_F = ["qsr", "l2amp"]
RESP_F = ["s_core", "s_mem"]


def agg_features(ncu_csv):
    """per-kernel time-weighted feature row from raw ncu launches."""
    launches = common.load_ncu_csv(ncu_csv)
    by = {}
    for lk in launches:
        by.setdefault(short(lk.kernel), []).append(lk)
    out = {}
    for k, lks in by.items():
        def wavg(metric):
            vs = [(l.m(M[metric]), l.m(M["time"])) for l in lks]
            vs = [(v, t) for v, t in vs if v == v and t == t and t > 0]
            return sum(v * t for v, t in vs) / sum(t for _, t in vs) if vs else float("nan")

        def tot(metric):
            vs = [l.m(M[metric]) for l in lks if l.m(M[metric]) == l.m(M[metric])]
            return sum(vs) if vs else float("nan")
        l2h = wavg("l2_hit")
        sect = max(wavg("sect_ld") or 0, wavg("sect_st") or 0)
        dram = tot("dram_rd") + tot("dram_wr")
        lts = tot("lts_bytes")
        long_sb, lg, mio = wavg("st_long"), wavg("st_lg"), wavg("st_mio")
        out[k] = {
            "mem_sol": wavg("mem_sol"), "comp_sol": wavg("comp_sol"),
            "dram_sol": wavg("dram_sol"),
            "lfmr": 1 - l2h / 100 if l2h == l2h else float("nan"),
            "occ": wavg("occ"),
            "log_sect": math.log10(max(sect, 1e-2)) if sect == sect else float("nan"),
            "qsr": ((lg if lg == lg else 0) + (mio if mio == mio else 0))
                   / (long_sb + 0.05) if long_sb == long_sb else float("nan"),
            "l2amp": dram / lts if dram == dram and lts == lts and lts > 0 else float("nan"),
        }
    return out


def load(data_dir):
    rows = []
    idx = os.path.join(data_dir, "population_index.tsv")
    for rec in csv.DictReader(open(idx), delimiter="\t"):
        app = rec["app"]
        capp = os.path.join(data_dir, "cls", app)
        ncu = os.path.join(capp, "ncu_metrics.csv")
        clsf = os.path.join(capp, "classification.csv")
        if not (os.path.isfile(ncu) and os.path.isfile(clsf)):
            continue
        feats = agg_features(ncu)
        base, _ = read_kernsum(os.path.join(data_dir, "sweep", f"{app}_base_cuda_gpu_kern_sum.csv"))
        lowc, _ = read_kernsum(os.path.join(data_dir, "sweep", f"{app}_lowcore_cuda_gpu_kern_sum.csv"))
        lowm, _ = read_kernsum(os.path.join(data_dir, "sweep", f"{app}_lowmem_cuda_gpu_kern_sum.csv"))
        for r in csv.DictReader(open(clsf)):
            k = short(r["kernel"])
            if k not in feats or r["class"] == "G0-nosignal":
                continue
            f = dict(feats[k])
            t1 = base.get(k)
            f["s_core"] = lowc[k] / t1 if t1 and k in lowc else float("nan")
            f["s_mem"] = lowm[k] / t1 if t1 and k in lowm else float("nan")
            f["dsa"] = math.degrees(math.atan2((f["s_core"] or 1) - 1, (f["s_mem"] or 1) - 1)) \
                if f["s_core"] == f["s_core"] and f["s_mem"] == f["s_mem"] else float("nan")
            rows.append({"app": app, "kernel": k, "class": r["class"], **f})
    return rows


def fstat(rows, feat):
    """one-way F: between-class variance / within-class variance."""
    groups = {}
    for r in rows:
        v = r.get(feat)
        if v is not None and v == v:
            groups.setdefault(r["class"], []).append(v)
    groups = {c: g for c, g in groups.items() if len(g) >= 2}
    if len(groups) < 2:
        return float("nan")
    allv = [v for g in groups.values() for v in g]
    gm = sum(allv) / len(allv)
    ssb = sum(len(g) * (sum(g) / len(g) - gm) ** 2 for g in groups.values())
    ssw = sum(sum((v - sum(g) / len(g)) ** 2 for v in g) for g in groups.values())
    dfb, dfw = len(groups) - 1, len(allv) - len(groups)
    return (ssb / dfb) / (ssw / dfw) if ssw > 0 and dfw > 0 else float("inf")


def zmatrix(rows, feats):
    X, keep = [], []
    for r in rows:
        v = [r.get(f) for f in feats]
        if all(x is not None and x == x for x in v):
            X.append(list(v)); keep.append(r)
    for j in range(len(feats)):
        col = [x[j] for x in X]
        mu = sum(col) / len(col)
        sd = (sum((c - mu) ** 2 for c in col) / len(col)) ** 0.5 or 1.0
        for x in X:
            x[j] = (x[j] - mu) / sd
    return X, [r["class"] for r in keep]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "reports/2026-07-12_gpu_damov_population"))
    args = ap.parse_args(argv)
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(out, exist_ok=True)

    rows = load(args.data)
    feats_all = BASE_F + NEW_F + RESP_F + ["dsa"]
    with open(os.path.join(out, "metrics_features.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["app", "kernel", "class"] + feats_all)
        for r in rows:
            w.writerow([r["app"], r["kernel"], r["class"]]
                       + [round(r[f], 4) if r.get(f) == r.get(f) and r.get(f) is not None
                          else "" for f in feats_all])

    ranking = sorted(((f, fstat(rows, f)) for f in feats_all),
                     key=lambda kv: -(kv[1] if kv[1] == kv[1] else -1))
    with open(os.path.join(out, "feature_ranking.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feature", "F_statistic (between/within class variance)"])
        for f, s in ranking:
            w.writerow([f, round(s, 2) if s == s else "n/a"])

    ablations = [("BASE (current classifier inputs)", BASE_F),
                 ("BASE + QSR + L2AMP (new static)", BASE_F + NEW_F),
                 ("BASE + response (S_core,S_mem)", BASE_F + RESP_F),
                 ("ALL", feats_all)]
    ab_rows = [["feature set", "n kernels", "kmeans purity@k=8", "silhouette"]]
    for name, feats in ablations:
        X, labels = zmatrix(rows, feats)
        if len(X) < 16:
            ab_rows.append([name, len(X), "n/a", "n/a"]); continue
        assign = cluster.kmeans(X, 8)
        ab_rows.append([name, len(X),
                        round(cluster.purity(assign, labels), 3),
                        round(cluster.silhouette(X, assign), 3)])
    with open(os.path.join(out, "metric_ablation.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows(ab_rows)

    lines = ["# Metric discovery — does anything separate the classes better?",
             "",
             f"{len(rows)} population kernels. Per-feature one-way F-statistic "
             "(higher = separates the classes more):", "",
             "| feature | F |", "|---|---|"]
    lines += [f"| {f} | {round(s, 1) if s == s else 'n/a'} |" for f, s in ranking]
    lines += ["", "k-means ablation (does adding candidates sharpen unsupervised recovery "
              "of the classes?):", "",
              "| feature set | n | purity@k=8 | silhouette |", "|---|---|---|---|"]
    lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in ab_rows[1:]]
    open(os.path.join(out, "METRICS_DISCOVERY.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
