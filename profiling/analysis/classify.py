#!/usr/bin/env python3
"""classify.py — GPU-adapted DAMOV bottleneck classes + PiM/ISP candidacy.

Implements the revised GPU taxonomy from `Adapting_DAMOV_to_GPU.md` §6 as an
NCU-counter decision tree (the doc's §3 "first-cut label straight from NCU").
DAMOV's CPU classes [Oliveira21] don't port unchanged — GPUs hide latency with
occupancy, cache in two levels, and live or die on coalescing — so kernels are
placed in:

  G1  DRAM-bandwidth-bound      high DRAM-SoL / MemSoL with LFMR_gpu≈1     → PiM: strong
  G2  coalescing-bound          sectors/request ≫ 4 while memory-limited   → PiM: conditional (scatter-capable PiM, or a data-layout fix first)
  G3  L2-capacity/reuse-bound   memory-bound with LOW LFMR (L2 is earning
                                its keep) — deep hierarchy helps            → PiM: weak (bigger cache wins; PiM removes the cache that works)
  G4  latency/occupancy-bound   long-scoreboard dominant at low occupancy,
                                DRAM far from saturated                     → PiM: strong when the working set also defeats caches (LFMR high / set ≫ L2), else fix occupancy first
  G5  compute-bound             Compute-SoL dominant or math-pipe throttle  → PiM: none (keep on host GPU)
  G6  shared-memory/MIO-bound   MIO/short-scoreboard dominant, on-chip      → PiM: none (on-chip problem)
  G7  dependency/ILP-bound      'wait'/barrier/short-scoreboard dominant at
                                low occupancy, memory NOT the wall          → PiM: none — raise occupancy/ILP, then re-screen
  G0  underutilized/no-signal   nothing dominant (tiny kernels, launch tax) → n/a

G7 was not in the §6 hypothesis table; it emerged from the first cuVSLAM data
(the RGBD matcher:: kernels stall on fixed-latency dependencies at 16%
occupancy) — exactly the "classes should fall out of the data" step the
adaptation doc prescribes.

Inputs are architecture-measured proxies, all present in the `characterize`
metric set:  LFMR_gpu = 1 − L2-hit (≈ L2misses/L1misses), MPKI_gpu = DRAM
sectors per kilo warp-instruction, DRAM-SoL%, sectors/request (coalescing),
occupancy, stall taxonomy, AI (FLOP/DRAM-byte), DRAM bytes/launch vs L2 size.

The verdict combines the class with the stage's persistence hypothesis
(stages.py) to name a substrate: near-sensor SRAM (streaming), DRAM-PiM
(hot-persistent), ISP/near-storage (cold-persistent).

HONEST LIMITS (stated in the emitted table): this is single-point profiling.
It cannot see the LFMR-vs-#SM trend (G3 vs G1 at scale), divergence, or true
reuse distance — those need the gated Slice-3 NVBit/Accel-Sim track, which
refines (not replaces) these labels. Reads only derived CSVs → runs with no
GPU and no dataset.

Usage:
  python3 -m analysis.classify <report_data_dir | ncu_results_dir> [more dirs...]
      --hw profiling/hw/<gpu>.toml [--nsys <nsys_results_dir>] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common, roofline, screen, stages, svgfig  # noqa: E402

CLASSES = {
    "G1-bandwidth":  ("#cc3311", "DRAM bandwidth-bound"),
    "G2-coalescing": ("#ee7733", "coalescing/scatter-bound"),
    "G3-l2-reuse":   ("#0077bb", "L2 capacity/reuse-bound"),
    "G4-latency":    ("#aa3377", "latency-bound at low occupancy"),
    "G5-compute":    ("#228833", "compute-bound"),
    "G6-onchip":     ("#999933", "shared-mem/MIO-bound (on-chip)"),
    "G7-dependency": ("#66ccee", "dependency/ILP-bound at low occupancy"),
    "G0-nosignal":   ("#bbbbbb", "underutilized / no dominant signal"),
}

MEMORY_STALLS = screen.MEMORY_STALLS


def _dominant_stall(r):
    items = [(n, r.get(f"stall_{n}", float("nan"))) for n, _, _ in screen.STALLS]
    items = [(n, v) for n, v in items if v == v]
    if not items:
        return None, float("nan")
    return max(items, key=lambda x: x[1])


# Decision-tree thresholds. DAMOV calibrated its single 30% cutoff empirically;
# ours are stated here once and STRESS-TESTED: classify_kernel is re-run at
# ×0.75 and ×1.25 of every threshold, and kernels whose class flips are flagged
# 'borderline' in the output (see sensitivity()).
THRESHOLDS = {
    "sol_hi": 40.0,        # SoL% considered 'high' (bound-ness)
    "sol_ratio": 1.5,      # dominance ratio between Mem/Comp SoL
    "dram_sat": 50.0,      # DRAM-SoL% (of ncu's theoretical peak) = saturated
    "lfmr_hi": 0.4,        # LFMR_gpu above which the L2 is 'not helping'
    "lfmr_lo": 0.35,       # LFMR_gpu below which the L2 'earns its keep'
    "sect_scatter": 8.0,   # sectors/request marking scattered access (4 = ideal)
    "occ_low": 25.0,       # occupancy% below which latency can't be hidden
    "occ_low_dep": 30.0,   # occupancy% bound for the dependency class
    "stall_dom": 1.0,      # warps/issue-active for a stall to count as dominant
}


def classify_kernel(r: dict, hw: dict, th: dict = THRESHOLDS) -> dict:
    """r: merged feature row (screen ∪ roofline fields). Returns class row."""
    l2_bytes = float(hw.get("memory", {}).get("l2_bytes") or 0)
    mem, comp = r.get("mem_sol", float("nan")), r.get("comp_sol", float("nan"))
    dram_sol = r.get("dram_sol", float("nan"))
    lfmr = r.get("lfmr", float("nan"))
    occ = r.get("occ", float("nan"))
    sect = max(r.get("sect_ld", float("nan")) or 0, r.get("sect_st", float("nan")) or 0)
    wset = r.get("dram_bytes_per_launch", float("nan"))
    ai = r.get("ai_dram", float("nan"))
    dom, dom_v = _dominant_stall(r)

    cls, conf, why = "G0-nosignal", "low", []
    sol_hi, ratio = th["sol_hi"], th["sol_ratio"]
    mem_limited = (mem == mem and mem >= sol_hi) or \
        (dom in MEMORY_STALLS and dom_v == dom_v and dom_v >= th["stall_dom"])

    if comp == comp and comp >= sol_hi and (mem != mem or comp >= ratio * mem):
        cls, conf = "G5-compute", "high"
        why.append(f"CompSoL {comp:.0f}% dominant" + (f", AI {ai:.1f} FLOP/B" if ai == ai else ""))
    elif dom in ("mio_throttle", "short_scoreboard") and dom_v >= th["stall_dom"] \
            and not (dram_sol == dram_sol and dram_sol >= sol_hi):
        cls, conf = "G6-onchip", "medium"
        why.append(f"dominant stall {dom} ({dom_v:.1f} warps) with DRAM unsaturated")
    elif dram_sol == dram_sol and dram_sol >= th["dram_sat"]:
        cls = "G1-bandwidth"
        conf = "high" if lfmr == lfmr and lfmr >= th["lfmr_hi"] else "medium"
        why.append(f"DRAM-SoL {dram_sol:.0f}%")
        if lfmr == lfmr:
            why.append(f"LFMR {lfmr:.2f}" + (" (L2 not helping)" if lfmr >= th["lfmr_hi"]
                                             else " (L2 absorbing reuse)"))
    elif mem_limited and sect == sect and sect >= th["sect_scatter"]:
        cls, conf = "G2-coalescing", "high" if sect >= 2 * th["sect_scatter"] else "medium"
        why.append(f"{sect:.0f} sectors/request (4 = coalesced)")
    elif mem_limited and lfmr == lfmr and lfmr < th["lfmr_lo"]:
        cls, conf = "G3-l2-reuse", "medium"
        why.append(f"memory-limited but LFMR {lfmr:.2f} — the L2 is earning its keep")
    elif dom == "long_scoreboard" and occ == occ and occ < th["occ_low"] \
            and not (dram_sol == dram_sol and dram_sol >= sol_hi):
        cls = "G4-latency"
        conf = "high" if (wset == wset and l2_bytes and wset > l2_bytes) else "medium"
        why.append(f"long-scoreboard dominant at {occ:.0f}% occupancy, DRAM-SoL {dram_sol:.0f}%")
        if wset == wset and l2_bytes and wset > l2_bytes:
            why.append(f"working set {wset/1e6:.1f} MB/launch ≫ L2 {l2_bytes/1e6:.1f} MB")
    elif (dom in ("wait", "short_scoreboard", "barrier", "not_selected")
          and occ == occ and occ < th["occ_low_dep"]
          and not (mem == mem and mem >= sol_hi) and not (comp == comp and comp >= sol_hi)):
        cls, conf = "G7-dependency", "medium"
        why.append(f"'{dom}' stall dominant at {occ:.0f}% occupancy; memory is not the wall "
                   f"(MemSoL {mem:.0f}%, DRAM-SoL {dram_sol:.0f}%)")
        if sect == sect and sect >= th["sect_scatter"]:
            why.append(f"note: scattered access ({sect:.0f} sect/req) — re-screen for PiM once occupancy is fixed")
    elif mem_limited:
        cls, conf = "G1-bandwidth", "low"
        why.append(f"memory-limited (MemSoL {mem:.0f}%) without a sharper signature")
    else:
        why.append("no dominant bottleneck signal (short/latency-tax kernels)")

    return {"class": cls, "confidence": conf, "why": "; ".join(why),
            "dom_stall": dom or "", "sect": sect, "lfmr": lfmr,
            "dram_sol": dram_sol, "wset": wset}


def sensitivity(r: dict, hw: dict) -> str:
    """Stress the thresholds ±25%: 'stable' if the class survives, else the
    set of classes it flips among. DAMOV's analog: calibrating the 30% cutoff."""
    seen = set()
    for scale in (0.75, 1.0, 1.25):
        th = {k: v * scale for k, v in THRESHOLDS.items()}
        seen.add(classify_kernel(r, hw, th)["class"])
    if len(seen) == 1:
        return "stable"
    return "borderline:" + "↔".join(sorted(seen))


def pim_affinity(cls: str, persistence: str, r: dict) -> tuple[str, str]:
    """(affinity, substrate) from the bottleneck class × persistence hypothesis."""
    if cls in ("G5-compute", "G6-onchip"):
        return "none", "host GPU"
    if cls == "G7-dependency":
        return "none", "host GPU — raise occupancy/ILP first, then re-screen"
    if cls == "G0-nosignal":
        return "n/a", "—"
    cold = "cold" in persistence
    hot = "hot" in persistence
    streaming = "streaming" in persistence
    scattered = r.get("sect", 0) == r.get("sect", 0) and r.get("sect", 0) >= 8
    big_wset = r.get("wset", float("nan")) == r.get("wset", float("nan")) and r.get("wset", 0) > 4e6

    if cold and (cls in ("G4-latency", "G2-coalescing", "G1-bandwidth")) and big_wset:
        return "strong", "ISP / near-storage scan engine"
    if cls == "G1-bandwidth":
        if streaming:
            return "strong", "near-sensor SRAM (consume before DRAM)"
        return "strong", "DRAM-PiM (bank-level bandwidth)"
    if cls == "G2-coalescing":
        return "conditional", "scatter-capable PiM — or a data-layout fix first"
    if cls == "G4-latency":
        if r.get("lfmr", 0) == r.get("lfmr", 0) and r.get("lfmr", 0) >= 0.4 or big_wset:
            return "strong", "near-memory compute (latency, uncacheable set)"
        return "conditional", "raise occupancy first; PiM if the set defeats caches"
    if cls == "G3-l2-reuse":
        return "weak", "bigger/persisted L2 wins; PiM would forfeit the reuse" if hot else "weak, cache-friendly"
    return "n/a", "—"


# ── feature loading: from a report data dir (CSVs) or an ncu results dir ─────

def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_features(path: str) -> list[dict]:
    """Accepts a results dir (has derived/ncu_metrics.csv) or a report data dir
    (has screen.csv [+ roofline.csv]). Returns merged per-kernel feature rows."""
    ncu_csv = common.find_derived(path, "ncu_metrics.csv") if os.path.isdir(os.path.join(path, "derived")) else None
    if ncu_csv:
        launches = common.load_ncu_csv(ncu_csv)
        rows = screen.aggregate(launches)
        roof = {r["kernel"]: r for r in roofline.aggregate(launches)}
        for r in rows:
            r["ai_dram"] = roof.get(r["kernel"], {}).get("ai_dram", float("nan"))
        return rows

    screen_csv = os.path.join(path, "screen.csv")
    if not os.path.isfile(screen_csv):
        raise SystemExit(f"{path}: neither an ncu results dir nor a report data dir (no screen.csv)")
    rows = []
    for r in csv.DictReader(open(screen_csv)):
        row = {"kernel": r["kernel"], "stage": r["stage"], "verdict": r.get("verdict", ""),
               "launches": int(r.get("launches") or 0), "time_s": _f(r.get("time_ms")) / 1e3,
               "mem_sol": _f(r.get("mem_sol_pct")), "comp_sol": _f(r.get("comp_sol_pct")),
               "dram_sol": _f(r.get("dram_sol_pct")), "l1_hit": _f(r.get("l1_hit_pct")),
               "l2_hit": _f(r.get("l2_hit_pct")), "occ": _f(r.get("occupancy_pct")),
               "sect_ld": _f(r.get("sectors_per_req_ld")), "sect_st": _f(r.get("sectors_per_req_st")),
               "lfmr": _f(r.get("lfmr_gpu")), "mpki": _f(r.get("mpki_gpu")),
               "dram_bytes_per_launch": _f(r.get("dram_bytes_per_launch")),
               "ai_dram": float("nan")}
        for n, _, _ in screen.STALLS:
            row[f"stall_{n}"] = _f(r.get(f"stall_{n}"))
        rows.append(row)
    roof_csv = os.path.join(path, "roofline.csv")
    if os.path.isfile(roof_csv):
        roof = {r["kernel"]: _f(r.get("ai_dram")) for r in csv.DictReader(open(roof_csv))}
        for r in rows:
            r["ai_dram"] = roof.get(r["kernel"], float("nan"))
    return rows


def run(paths: list[str], hw: dict) -> list[dict]:
    merged: dict[str, dict] = {}
    for p in paths:
        for r in load_features(p):
            # keep the row with the larger profiled time for duplicate kernels
            k = r["kernel"]
            if k not in merged or r["time_s"] > merged[k]["time_s"]:
                r["source"] = os.path.basename(os.path.normpath(p))
                merged[k] = r
    out = []
    for r in merged.values():
        c = classify_kernel(r, hw)
        # small-sample guard: DAMOV-style claims need more than a couple of
        # launches; cap the confidence and say so
        n = r.get("launches") or 0
        if 0 < n < 5 and c["confidence"] != "low":
            c["confidence"] = "low"
            c["why"] += f"; only n={n} profiled launches — small sample"
        c["stability"] = sensitivity(r, hw)
        if c["stability"] != "stable" and c["confidence"] == "high":
            c["confidence"] = "medium"
        pers = stages.persistence_of(r["stage"])
        aff, substrate = pim_affinity(c["class"], pers, c)
        out.append({**r, **c, "persistence": pers, "pim": aff, "substrate": substrate})
    out.sort(key=lambda r: -(r["time_s"] if r["time_s"] == r["time_s"] else 0))
    return out


def emit(rows: list[dict], hw: dict, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    p = os.path.join(out_dir, "classification.csv")
    common.write_csv(p, ["kernel", "stage", "persistence", "class", "confidence",
                         "stability", "pim_affinity", "substrate", "time_ms", "lfmr_gpu",
                         "mpki_gpu", "dram_sol_pct", "sectors_per_req", "occupancy_pct",
                         "ai_flop_per_byte", "dram_bytes_per_launch", "dominant_stall",
                         "rationale", "source"],
                     [[r["kernel"], r["stage"], r["persistence"], r["class"],
                       r["confidence"], r["stability"], r["pim"], r["substrate"],
                       round(r["time_s"] * 1e3, 3) if r["time_s"] == r["time_s"] else "",
                       round(r["lfmr"], 3) if r["lfmr"] == r["lfmr"] else "",
                       round(r.get("mpki", float("nan")), 2) if r.get("mpki", float("nan")) == r.get("mpki", float("nan")) else "",
                       round(r["dram_sol"], 1) if r["dram_sol"] == r["dram_sol"] else "",
                       round(r["sect"], 1) if r["sect"] == r["sect"] else "",
                       round(r["occ"], 1) if r["occ"] == r["occ"] else "",
                       round(r["ai_dram"], 3) if r["ai_dram"] == r["ai_dram"] else "",
                       f"{r['wset']:.6g}" if r["wset"] == r["wset"] else "",
                       r["dom_stall"], r["why"], r.get("source", "")] for r in rows])
    written.append(p)

    # classification scatter: LFMR (does the hierarchy help?) vs DRAM-SoL (is
    # bandwidth the wall?), sized by time, colored by class
    total_t = sum(r["time_s"] for r in rows if r["time_s"] == r["time_s"]) or 1.0
    pts = []
    for r in rows:
        if r["lfmr"] != r["lfmr"] or r["dram_sol"] != r["dram_sol"]:
            continue
        rad = 4 + 12 * math.sqrt(max(r["time_s"], 0) / total_t)
        pts.append((f'{r["kernel"]} [{r["class"]}]', r["lfmr"], r["dram_sol"],
                    CLASSES[r["class"]][0], rad))
    legend = [(f"{k} — {v[1]}", v[0]) for k, v in CLASSES.items()
              if any(r["class"] == k for r in rows)]
    p = os.path.join(out_dir, "fig_classification.svg")
    svgfig.scatter(p, "GPU-DAMOV placement (size ∝ GPU-time share)",
                   pts, "LFMR_gpu = 1 − L2 hit rate  (→1: caches useless)",
                   "DRAM throughput (% of peak)", x_max=1.0, y_max=100.0,
                   legend=legend)
    written.append(p)
    return written


def synthesis(rows: list[dict]) -> list[list]:
    """stage → dominant class → PiM/ISP verdict (time-weighted)."""
    by_stage: dict[str, list[dict]] = {}
    for r in rows:
        by_stage.setdefault(r["stage"], []).append(r)
    out = []
    for s in stages.ORDER:
        rs = by_stage.get(s)
        if not rs:
            continue
        t = {}
        for r in rs:
            if r["time_s"] == r["time_s"]:
                t[r["class"]] = t.get(r["class"], 0.0) + r["time_s"]
        dom_cls = max(t, key=t.get) if t else "G0-nosignal"
        dom_rows = [r for r in rs if r["class"] == dom_cls]
        aff = max(dom_rows, key=lambda r: r["time_s"] if r["time_s"] == r["time_s"] else 0)
        out.append([s, stages.persistence_of(s), dom_cls,
                    f"{100*t.get(dom_cls,0)/ (sum(t.values()) or 1):.0f}% of stage time",
                    aff["pim"], aff["substrate"]])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help="report data dir(s) and/or ncu results dir(s); merged")
    ap.add_argument("--hw", required=True)
    ap.add_argument("--out", default=".")
    args = ap.parse_args(argv)
    hw = common.load_hw(args.hw)
    rows = run(args.paths, hw)
    for p in emit(rows, hw, args.out):
        print(f"[✓] {p}")
    print()
    print(common.md_table(["stage", "persistence", "dominant class", "share",
                           "PiM affinity", "substrate"], synthesis(rows)))


if __name__ == "__main__":
    main()
