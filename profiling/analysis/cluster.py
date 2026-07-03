#!/usr/bin/env python3
"""cluster.py — do the G-classes fall out of the data? (DAMOV-style validation)

DAMOV derived its bottleneck classes from k-means over the metric vectors and
the GPU adaptation study prescribes the same (step 9): the decision tree in
classify.py is the *labeling*; clustering is the *validation*. This module
runs stdlib k-means (multi-restart, k chosen by silhouette) over the same
per-kernel features the tree uses, then reports how well the unsupervised
clusters agree with the assigned G-classes (purity + adjusted Rand index).

Interpretation guardrails, stated in the output: agreement on ONE dataset is a
consistency check; the validation claim needs the multi-dataset matrix. Tiny
GPU-time kernels can be excluded (--min-time-ms) so the launch-tax noise
doesn't dominate the geometry.

Stdlib only (47-kernel scale needs no sklearn).

Usage:
  python3 -m analysis.cluster <ncu_results_dir|report_data_dir> [more...]
      --hw profiling/hw/<gpu>.toml [--min-time-ms 0.05] [--out DIR]
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import classify, common  # noqa: E402

# feature -> (extractor, use_log)
FEATURES = [
    ("mem_sol", lambda r: r.get("mem_sol"), False),
    ("comp_sol", lambda r: r.get("comp_sol"), False),
    ("dram_sol", lambda r: r.get("dram_sol"), False),
    ("lfmr", lambda r: r.get("lfmr"), False),
    ("occupancy", lambda r: r.get("occ"), False),
    ("sectors_req", lambda r: max(r.get("sect_ld") or 0, r.get("sect_st") or 0) or float("nan"), True),
    ("stall_long_sb", lambda r: r.get("stall_long_scoreboard"), True),
    ("stall_wait", lambda r: r.get("stall_wait"), True),
]


def feature_matrix(rows):
    """(kernel_names, X, used_feature_names) with z-scored columns; NaN -> column mean."""
    names, vecs = [], []
    for r in rows:
        v = []
        for _, fn, use_log in FEATURES:
            x = fn(r)
            x = float(x) if x is not None and x == x else float("nan")
            if use_log and x == x:
                x = math.log10(max(x, 1e-3))
            v.append(x)
        names.append(r["kernel"])
        vecs.append(v)
    nfeat = len(FEATURES)
    # column means over non-NaN, impute, z-score
    for j in range(nfeat):
        col = [v[j] for v in vecs if v[j] == v[j]]
        mu = sum(col) / len(col) if col else 0.0
        sd = (sum((x - mu) ** 2 for x in col) / len(col)) ** 0.5 if col else 1.0
        sd = sd or 1.0
        for v in vecs:
            v[j] = ((v[j] if v[j] == v[j] else mu) - mu) / sd
    return names, vecs, [f for f, _, _ in FEATURES]


def _dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def kmeans(X, k, restarts=20, iters=100, seed=0):
    rng = random.Random(seed)
    best, best_inertia = None, float("inf")
    for _ in range(restarts):
        centers = [list(X[i]) for i in rng.sample(range(len(X)), k)]
        assign = [0] * len(X)
        for _ in range(iters):
            new_assign = [min(range(k), key=lambda c: _dist2(x, centers[c])) for x in X]
            if new_assign == assign:
                break
            assign = new_assign
            for c in range(k):
                members = [X[i] for i in range(len(X)) if assign[i] == c]
                if members:
                    centers[c] = [sum(col) / len(members) for col in zip(*members)]
        inertia = sum(_dist2(X[i], centers[assign[i]]) for i in range(len(X)))
        if inertia < best_inertia:
            best, best_inertia = assign, inertia
    return best


def silhouette(X, assign):
    n = len(X)
    if len(set(assign)) < 2:
        return -1.0
    d = [[math.sqrt(_dist2(X[i], X[j])) for j in range(n)] for i in range(n)]
    scores = []
    for i in range(n):
        own = [d[i][j] for j in range(n) if j != i and assign[j] == assign[i]]
        a = sum(own) / len(own) if own else 0.0
        bs = []
        for c in set(assign):
            if c == assign[i]:
                continue
            others = [d[i][j] for j in range(n) if assign[j] == c]
            if others:
                bs.append(sum(others) / len(others))
        b = min(bs) if bs else 0.0
        scores.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return sum(scores) / n


def adjusted_rand(labels_a, labels_b):
    """Adjusted Rand index between two labelings (stdlib)."""
    def comb2(x):
        return x * (x - 1) // 2
    n = len(labels_a)
    pairs = Counter(zip(labels_a, labels_b))
    a_cnt, b_cnt = Counter(labels_a), Counter(labels_b)
    sum_ij = sum(comb2(c) for c in pairs.values())
    sum_a = sum(comb2(c) for c in a_cnt.values())
    sum_b = sum(comb2(c) for c in b_cnt.values())
    expected = sum_a * sum_b / comb2(n) if comb2(n) else 0.0
    max_index = (sum_a + sum_b) / 2
    return (sum_ij - expected) / (max_index - expected) if max_index != expected else 1.0


def purity(clusters, classes):
    total = len(classes)
    by_cluster = {}
    for c, g in zip(clusters, classes):
        by_cluster.setdefault(c, []).append(g)
    return sum(Counter(gs).most_common(1)[0][1] for gs in by_cluster.values()) / total


def run(paths, hw, min_time_ms=0.05):
    rows = classify.run(paths, hw)
    kept = [r for r in rows if r["time_s"] == r["time_s"] and r["time_s"] * 1e3 >= min_time_ms]
    names, X, feats = feature_matrix(kept)
    classes = [r["class"] for r in kept]
    results = {}
    for k in range(2, min(9, len(kept))):
        assign = kmeans(X, k)
        results[k] = {"assign": assign, "silhouette": silhouette(X, assign),
                      "ari": adjusted_rand(assign, classes),
                      "purity": purity(assign, classes)}
    best_k = max(results, key=lambda k: results[k]["silhouette"])
    return {"kernels": names, "classes": classes, "kept": kept,
            "features": feats, "results": results, "best_k": best_k,
            "n_excluded": len(rows) - len(kept)}


def emit(out, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    k = out["best_k"]
    r = out["results"][k]
    rows = [[n, c, r["assign"][i]] for i, (n, c) in
            enumerate(zip(out["kernels"], out["classes"]))]
    p = os.path.join(out_dir, "clusters.csv")
    common.write_csv(p, ["kernel", "tree_class", f"kmeans_cluster_k{k}"], rows)
    sweep = [[k_, round(v["silhouette"], 3), round(v["ari"], 3), round(v["purity"], 3)]
             for k_, v in sorted(out["results"].items())]
    p2 = os.path.join(out_dir, "cluster_sweep.csv")
    common.write_csv(p2, ["k", "silhouette", "ari_vs_tree", "purity_vs_tree"], sweep)
    return [p, p2]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--hw", required=True)
    ap.add_argument("--min-time-ms", type=float, default=0.05)
    ap.add_argument("--out", default=".")
    args = ap.parse_args(argv)
    hw = common.load_hw(args.hw)
    out = run(args.paths, hw, args.min_time_ms)
    for p in emit(out, args.out):
        print(f"[✓] {p}")
    print(f"\n{len(out['kernels'])} kernels clustered ({out['n_excluded']} excluded "
          f"below {args.min_time_ms} ms); features: {', '.join(out['features'])}")
    print(common.md_table(["k", "silhouette", "ARI vs tree", "purity vs tree"],
                          [[k, round(v['silhouette'], 3), round(v['ari'], 3),
                            round(v['purity'], 3)]
                           for k, v in sorted(out['results'].items())]))
    k = out["best_k"]
    print(f"\nbest k by silhouette: {k} — ARI {out['results'][k]['ari']:.3f}, "
          f"purity {out['results'][k]['purity']:.3f} vs the decision-tree classes")
    print("(single-dataset run = consistency check; the validation claim needs the matrix)")


if __name__ == "__main__":
    main()
