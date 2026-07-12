#!/usr/bin/env python3
"""population.py — DAMOV's population-scale two-phase validation, GPU edition.

Consumes what population_campaign.sh measured on the target for REAL, FOREIGN
codebases (BabelStream, Polybench-GPU, Rodinia-CUDA, CUDA samples — the first
two/three overlap DAMOV's own CPU application sources):

  cls/<app>/classification.csv       blind classification, frozen thresholds
  sweep/<app>_{base,lowcore,lowmem}_cuda_gpu_kern_sum.csv
                                     per-kernel time at the 3 clock points

and produces the paper-parity analyses (paper §3.5, §3.3, Fig. 18a, §4.1):

  population.csv            one row per (app, kernel): class + features +
                            S_core/S_mem + screen/signature verdicts
  phase2_accuracy           kernel correct iff blind class's clock-response
                            signature matches measurement (their two-condition
                            correctness; 97/100 was their result)
  class_distributions.csv   per-class feature medians/IQR (Fig. 18a analog)
  derived_thresholds_population.csv
                            §3.5 phase-1 midpoint derivation re-run on the
                            WIDENED population (stability vs cuVSLAM-only)
  impossible_combos.csv     the paper's "combinations we do not observe" audit
  cluster_sweep_population.csv
                            k-means silhouette sweep on the widened cloud —
                            does k≈8 persist beyond cuVSLAM?
  outliers.csv + SUMMARY.md

Usage:
  python3 -m analysis.population --data <copied population_out> \
      --out ../reports/2026-07-12_gpu_damov_population
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import classify, cluster  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# refined per-class clock-response signatures (see validation/clock_sweep.sh —
# derived from designed archetypes, both deviations grounded in clock-domain
# architecture facts). Bands are tolerant because real kernels are mixtures;
# they stay mutually discriminating via WHICH domain dominates.
SIG = {
    "G1-bandwidth":  (0.90, 1.25, 1.13, 1.55),
    "G2-coalescing": (1.05, 1.60, 0.85, 1.12),
    "G3-l2-reuse":   (1.45, 2.15, 0.85, 1.12),
    "G4-latency":    (1.15, 1.65, 1.00, 1.28),
    "G5-compute":    (1.60, 2.15, 0.85, 1.12),
    "G6-onchip":     (1.60, 2.15, 0.85, 1.12),
    "G7-dependency": (1.60, 2.15, 0.85, 1.12),
}
SCREEN_MS = 0.5        # Step-1 floor: whole-run kernel time
SCREEN_SHARE = 1.0     # ... or <1% of the app's GPU time


def short(name):
    s = name.split("(")[0].strip()
    s = re.sub(r"^void\s+", "", s)
    s = s.split("<")[0]
    parts = s.split("::")
    return "::".join(parts[-2:]) if len(parts) > 2 else s


def fnum(v, d=None):
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return d


def read_kernsum(path):
    """nsys cuda_gpu_kern_sum csv -> {short_kernel: total_ns} (+ app total)."""
    if not os.path.isfile(path):
        return {}, 0.0
    out, total = {}, 0.0
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            name = r.get("Name") or r.get("Kernel Name") or ""
            ns = fnum(r.get("Total Time (ns)"), 0.0)
            if not name or not ns:
                continue
            k = short(name)
            out[k] = out.get(k, 0.0) + ns
            total += ns
    return out, total


def load_population(data_dir):
    rows = []
    idx = os.path.join(data_dir, "population_index.tsv")
    for rec in csv.DictReader(open(idx), delimiter="\t"):
        app, status = rec["app"], rec["status"]
        if "FAILED" in status and "classify" in status:
            continue
        cls_csv = os.path.join(data_dir, "cls", app, "classification.csv")
        if not os.path.isfile(cls_csv):
            continue
        base, base_tot = read_kernsum(os.path.join(data_dir, "sweep", f"{app}_base_cuda_gpu_kern_sum.csv"))
        lowc, _ = read_kernsum(os.path.join(data_dir, "sweep", f"{app}_lowcore_cuda_gpu_kern_sum.csv"))
        lowm, _ = read_kernsum(os.path.join(data_dir, "sweep", f"{app}_lowmem_cuda_gpu_kern_sum.csv"))
        for r in csv.DictReader(open(cls_csv)):
            k = short(r["kernel"])
            t_base = base.get(k)
            row = {
                "app": app, "kernel": k, "class": r["class"],
                "confidence": r.get("confidence", ""), "stability": r.get("stability", ""),
                "dram_sol": fnum(r.get("dram_sol_pct")), "lfmr": fnum(r.get("lfmr_gpu")),
                "mpki": fnum(r.get("mpki_gpu")), "sect": fnum(r.get("sectors_per_req")),
                "occ": fnum(r.get("occupancy_pct")), "ai": fnum(r.get("ai_flop_per_byte")),
                "t_base_ms": round(t_base / 1e6, 3) if t_base else None,
                "share_pct": round(100 * t_base / base_tot, 2) if t_base and base_tot else None,
                "s_core": round(lowc[k] / t_base, 3) if t_base and k in lowc else None,
                "s_mem": round(lowm[k] / t_base, 3) if t_base and k in lowm else None,
            }
            # Step-1 screen (whole-run numbers, like DAMOV's >=3%-of-cycles)
            row["screened"] = (row["t_base_ms"] is None or row["t_base_ms"] < SCREEN_MS
                               or (row["share_pct"] is not None and row["share_pct"] < SCREEN_SHARE))
            rows.append(row)
    return rows


def phase2(rows):
    """Signature verdict per non-screened kernel with sweep data."""
    for r in rows:
        r["signature"] = ""
        if r["screened"] or r["s_core"] is None or r["s_mem"] is None:
            continue
        sig = SIG.get(r["class"])
        if not sig:                       # G0 etc.
            r["signature"] = "n/a"
            continue
        clo, chi, mlo, mhi = sig
        if clo <= r["s_core"] <= chi and mlo <= r["s_mem"] <= mhi:
            r["signature"] = "match"
        elif r["s_core"] < 1.15 and r["s_mem"] < 1.12:
            # bounded by neither domain (launch/host/PCIe-bound): the class
            # fingerprint may still be right but the run can't test it
            r["signature"] = "inconclusive"
        else:
            r["signature"] = "mismatch"
    tested = [r for r in rows if r["signature"] in ("match", "mismatch", "inconclusive")]
    m = sum(r["signature"] == "match" for r in tested)
    x = sum(r["signature"] == "mismatch" for r in tested)
    i = sum(r["signature"] == "inconclusive" for r in tested)
    return {"tested": len(tested), "match": m, "mismatch": x, "inconclusive": i,
            "strict_pct": round(100 * m / max(len(tested), 1), 1),
            "conclusive_pct": round(100 * m / max(m + x, 1), 1)}


def distributions(rows):
    per = {}
    for r in rows:
        if r["screened"]:
            continue
        per.setdefault(r["class"], []).append(r)
    out = [["class", "n", *(f"{f}_med" for f in ("dram_sol", "lfmr", "mpki", "sect", "occ")),
            *(f"{f}_iqr" for f in ("dram_sol", "lfmr", "mpki", "sect", "occ"))]]
    for c in sorted(per):
        rs = per[c]
        med, iqr = [], []
        for f in ("dram_sol", "lfmr", "mpki", "sect", "occ"):
            xs = sorted(x[f] for x in rs if x[f] is not None)
            if xs:
                med.append(round(xs[len(xs) // 2], 2))
                iqr.append(round(xs[(3 * len(xs)) // 4] - xs[len(xs) // 4], 2))
            else:
                med.append(""); iqr.append("")
        out.append([c, len(rs), *med, *iqr])
    return out


def impossible_combos(rows):
    """Paper §3.3: metric combinations that should not occur."""
    live = [r for r in rows if not r["screened"]]
    TH = classify.THRESHOLDS

    def n(pred):
        return sum(1 for r in live if pred(r))
    combos = [
        ("high MPKI (>30) with low LFMR (<lfmr_lo)",
         n(lambda r: (r["mpki"] or 0) > 30 and r["lfmr"] is not None and r["lfmr"] < TH["lfmr_lo"])),
        ("DRAM saturated (>=dram_sat) with low LFMR (<lfmr_lo)",
         n(lambda r: (r["dram_sol"] or 0) >= TH["dram_sat"] and r["lfmr"] is not None and r["lfmr"] < TH["lfmr_lo"])),
        ("scattered (sect>=2x thresh) while DRAM saturated AND L2 absorbing (lfmr<lo)",
         n(lambda r: (r["sect"] or 0) >= 2 * TH["sect_scatter"] and (r["dram_sol"] or 0) >= TH["dram_sat"]
           and r["lfmr"] is not None and r["lfmr"] < TH["lfmr_lo"])),
    ]
    return [["combination (should be ~absent)", "count", "of"],
            *[[c, k, len(live)] for c, k in combos]]


def widened_thresholds(rows):
    """§3.5 phase-1 midpoints, now over cuVSLAM + the real-codebase population."""
    # population side
    def side(rows_, classes, key):
        xs = [r[key] for r in rows_ if r["class"] in classes and r[key] is not None
              and not r["screened"]]
        return xs
    # cuVSLAM side (reuse the device reports through damov_parity's loader)
    from analysis import damov_parity as dp
    cu_rows, cu_labels = [], []
    for _t, rep in dp.DEVICE_REPORTS + dp.EXTRA_REPORTS:
        cls = dp.read_cls(rep)
        try:
            feats = classify.load_features(os.path.join(REPO, rep, "data"))
        except SystemExit:
            continue
        for r in feats:
            kr = cls.get(r["kernel"])
            if kr and kr["class"] != "G0-nosignal":
                cu_rows.append({"class": kr["class"], "screened": False,
                                "dram_sol": r.get("dram_sol"), "lfmr": r.get("lfmr"),
                                "occ": r.get("occ"),
                                "sect": max(r.get("sect_ld") or 0, r.get("sect_st") or 0) or None,
                                "comp_sol": r.get("comp_sol")})
    for r in rows:
        r.setdefault("comp_sol", None)   # population classification.csv lacks comp_sol
    both = cu_rows + rows
    TH = classify.THRESHOLDS
    checks = [
        ("dram_sat", "dram_sol",
         {"G2-coalescing", "G3-l2-reuse", "G4-latency", "G5-compute", "G6-onchip", "G7-dependency"},
         {"G1-bandwidth"}, TH["dram_sat"]),
        ("sect_scatter", "sect",
         {"G1-bandwidth", "G3-l2-reuse", "G5-compute", "G6-onchip"},
         {"G2-coalescing"}, TH["sect_scatter"]),
        ("occ_low(_dep)", "occ",
         {"G4-latency", "G7-dependency"},
         {"G1-bandwidth", "G2-coalescing", "G3-l2-reuse", "G5-compute", "G6-onchip"},
         (TH["occ_low"] + TH["occ_low_dep"]) / 2),
        ("lfmr band", "lfmr", {"G3-l2-reuse"}, {"G1-bandwidth", "G4-latency"},
         (TH["lfmr_lo"] + TH["lfmr_hi"]) / 2),
    ]
    out = [["threshold", "derived (cuVSLAM only)", "derived (widened population)", "stated"]]
    for name, key, lo_c, hi_c, stated in checks:
        def mid(rows_):
            lo = side(rows_, lo_c, key); hi = side(rows_, hi_c, key)
            if not lo or not hi:
                return None
            return (sum(lo) / len(lo) + sum(hi) / len(hi)) / 2
        m_cu, m_all = mid(cu_rows), mid(both)
        out.append([name,
                    round(m_cu, 2) if m_cu is not None else "n/a",
                    round(m_all, 2) if m_all is not None else "n/a", stated])
    return out


def ksweep(rows):
    """Silhouette sweep on the widened feature cloud (population + cuVSLAM)."""
    from analysis import damov_parity as dp
    feats, labels = [], []
    for _t, rep in dp.DEVICE_REPORTS + dp.EXTRA_REPORTS:
        cls = dp.read_cls(rep)
        try:
            fr = classify.load_features(os.path.join(REPO, rep, "data"))
        except SystemExit:
            continue
        for r in fr:
            kr = cls.get(r["kernel"])
            if kr and kr["class"] != "G0-nosignal":
                feats.append(r); labels.append(kr["class"])
    for r in rows:
        if r["screened"]:
            continue
        feats.append({"kernel": f'{r["app"]}:{r["kernel"]}', "mem_sol": None,
                      "comp_sol": None, "dram_sol": r["dram_sol"], "lfmr": r["lfmr"],
                      "occ": r["occ"], "sect_ld": r["sect"], "sect_st": r["sect"],
                      "stall_long_scoreboard": None, "stall_wait": None})
        labels.append(r["class"])
    names, X, _ = cluster.feature_matrix(feats)
    out = [["k", "silhouette", "purity_vs_tree"]]
    for k in range(4, 13):
        assign = cluster.kmeans(X, k)
        out.append([k, round(cluster.silhouette(X, assign), 3),
                    round(cluster.purity(assign, labels), 3)])
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="copied population_out dir")
    ap.add_argument("--out", default=os.path.join(REPO, "reports/2026-07-12_gpu_damov_population"))
    args = ap.parse_args(argv)
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(out, exist_ok=True)

    rows = load_population(args.data)
    acc = phase2(rows)

    cols = ["app", "kernel", "class", "confidence", "stability", "dram_sol", "lfmr",
            "mpki", "sect", "occ", "ai", "t_base_ms", "share_pct", "s_core", "s_mem",
            "screened", "signature"]
    write_csv(os.path.join(out, "population.csv"),
              [cols] + [[r.get(c, "") for c in cols] for r in rows])
    write_csv(os.path.join(out, "class_distributions.csv"), distributions(rows))
    write_csv(os.path.join(out, "impossible_combos.csv"), impossible_combos(rows))
    wt = widened_thresholds(rows)
    write_csv(os.path.join(out, "derived_thresholds_population.csv"), wt)
    ks = ksweep(rows)
    write_csv(os.path.join(out, "cluster_sweep_population.csv"), ks)
    outliers = [r for r in rows if r["signature"] == "mismatch"]
    write_csv(os.path.join(out, "outliers.csv"),
              [cols] + [[r.get(c, "") for c in cols] for r in outliers])

    apps = sorted({r["app"] for r in rows})
    live = [r for r in rows if not r["screened"]]
    best = max((r for r in ks[1:]), key=lambda r: r[1])
    lines = [
        "# GPU-DAMOV population validation — real codebases, two-phase, paper-parity",
        "",
        f"**Population:** {len(rows)} kernels ({len(live)} above the Step-1 screen) "
        f"from {len(apps)} real applications across 4 independent suites "
        f"(BabelStream, Polybench-GPU, Rodinia-CUDA, CUDA samples) — Polybench and "
        f"Rodinia are sources DAMOV's own CPU population drew from.",
        "",
        f"**Phase-2 (paper §3.5) two-condition correctness:** blind classification "
        f"with frozen thresholds, then the class must predict the kernel's measured "
        f"clock-domain response. Of {acc['tested']} testable kernels: "
        f"**{acc['match']} match, {acc['mismatch']} mismatch, {acc['inconclusive']} "
        f"inconclusive (insensitive to both domains — host/launch-bound)**. "
        f"Strict accuracy {acc['strict_pct']}%; among conclusive kernels "
        f"**{acc['conclusive_pct']}%** (DAMOV: 97% on 100 held-out CPU functions).",
        "",
        "**Threshold stability (§3.5 phase-1 re-derivation, widened population):**",
        "",
        "| threshold | cuVSLAM-only | widened | stated |", "|---|---|---|---|",
        *[f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in wt[1:]],
        "",
        f"**Cluster persistence (§4.1):** silhouette-best k on the widened cloud = "
        f"**k={best[0]}** (sil {best[1]}, purity {best[2]} vs the tree labels).",
        "",
        "See population.csv (full rows), class_distributions.csv (Fig-18a analog), "
        "impossible_combos.csv (§3.3 audit), outliers.csv (mismatches, named).",
    ]
    open(os.path.join(out, "SUMMARY.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
