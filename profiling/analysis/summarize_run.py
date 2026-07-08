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
import re

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


# ── cross-study context, joined into every summary ──────────────────────────
# These are the workload-independent per-kernel findings of the initial
# studies: the two-pass attribution join (NVTX + TaggedAllocator -> memory
# spaces + data-structure tag; "DRAM scratch" = register-spill traffic), the
# taxonomy stability across 27 sequences, and the k-means validation.
ATTR_CSV = "profiling/reports/2026-07-05_attribution_campaign/attribution_consistency.csv"
AGREE_CSV = "profiling/reports/2026-07-04_campaign/class_agreement.csv"
CLUST_CSV = "profiling/reports/2026-07-04_campaign/clusters.csv"
SWEEP_CSV = "profiling/reports/2026-07-04_campaign/cluster_sweep.csv"


def study_context():
    attr = {r["kernel"]: {
        "shared_pct": fnum(r.get("med_shared_pct")),
        "spill_pct": fnum(r.get("med_local_spill_pct")),
        "global_pct": fnum(r.get("med_global_pct")),
        "data_structure": r.get("modal_top_global_tag", ""),
        "tag_agreement_pct": fnum(r.get("agreement_pct")),
    } for r in read_csv(os.path.join(REPO, ATTR_CSV))}
    agree = {r["kernel"]: {
        "modal_class": r.get("modal_class", ""),
        "n_seq": r.get("n_seq", ""),
        "verdict": r.get("agreement", ""),
    } for r in read_csv(os.path.join(REPO, AGREE_CSV))}
    clust = {r["kernel"]: r.get("kmeans_cluster_k8", "")
             for r in read_csv(os.path.join(REPO, CLUST_CSV))}
    sweep = read_csv(os.path.join(REPO, SWEEP_CSV))
    best_k = max(sweep, key=lambda r: fnum(r.get("silhouette"), 0))["k"] if sweep else "?"

    def for_kernel(name):
        out = {}
        if name in attr:
            out["attribution"] = attr[name]
        if name in agree:
            out["taxonomy_stability"] = agree[name]
        if name in clust:
            out["kmeans"] = {"cluster_k8": clust[name], "sweep_best_k": best_k}
        return out or None
    return for_kernel


def device_peaks(label):
    """Roofline ceilings for the run's device, from its hw descriptor."""
    hw_map = {"mx450": "mx450_sm75.toml", "rtx2000ada": "rtx2000ada_sm89.toml",
              "dellworkstation": "dellworkstation_sm89.toml",
              "sm89": "dellworkstation_sm89.toml", "orin": "jetson_orin_sm87.toml"}
    hw = next((v for k, v in hw_map.items() if k in label.lower()), None)
    if not hw:
        return None
    text = open(os.path.join(REPO, "profiling", "hw", hw)).read()

    def g(key):
        m = re.search(rf"(?m)^{key}\s*=\s*([0-9.]+)", text)
        return float(m.group(1)) if m else None
    dram = g("dram_gbps_measured") or g("dram_gbps_theoretical")
    gflops = g("fp32_gflops_measured") or (
        (g("fp32_tflops_theoretical") or 0) * 1000 or None)
    if not (dram and gflops):
        return None
    return {"dram_gbps": dram, "peak_gflops": gflops, "hw": hw,
            "basis": "measured" if g("dram_gbps_measured") else "theoretical"}


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
    # screen.csv carries the raw SoL pair + working set the decision tree used
    scr = {r["kernel"]: r for r in read_csv(os.path.join(data_dir, "screen.csv"))}

    def timeline(kernel, field, fallback):
        row = dag.get(kernel)
        return fnum(row.get(field), fallback) if row else fallback

    study = study_context()
    kernels = []
    for r in cls:
        rf = roof.get(r["kernel"])
        sc = scr.get(r["kernel"], {})
        kernels.append({
            "name": r["kernel"], "stage": r.get("stage", "?"),
            "study": study(r["kernel"]),
            "time_ms": round(timeline(r["kernel"], "total_ms", fnum(r["time_ms"], 0)), 3),
            "share_pct": round(timeline(r["kernel"], "pct_gpu_time", 0.0), 2),
            "limiter": r.get("class", "?"),
            "substrate": r.get("substrate", "?"),
            "pim_affinity": r.get("pim_affinity", "?"),
            "rationale": r.get("rationale", ""),
            "confidence": r.get("confidence", ""),
            "stability": r.get("stability", ""),
            "persistence": r.get("persistence", ""),
            "evidence": {
                "dram_sol_pct": fnum(r.get("dram_sol_pct")),
                "mem_sol_pct": fnum(sc.get("mem_sol_pct")),
                "comp_sol_pct": fnum(sc.get("comp_sol_pct")),
                "sectors_per_req": fnum(r.get("sectors_per_req")),
                "lfmr": fnum(r.get("lfmr_gpu")),
                "mpki": fnum(r.get("mpki_gpu")),
                "occupancy_pct": fnum(r.get("occupancy_pct")),
                "ai": fnum(r.get("ai_flop_per_byte")),
                "wset_bytes_per_launch": fnum(r.get("dram_bytes_per_launch")),
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
        "device_peaks": device_peaks(name),
        "qor": qor,
        "stages": [{
            "name": s["stage"],
            "time_ms": round(fnum(s.get("total_ms"), 0), 3),
            "share_pct": round(fnum(s.get("pct_gpu_time"), 0), 2),
            "n_kernels": len((s.get("kernels") or "").split()),
        } for s in stages],
        "kernels": kernels,
    }


def shorten_kernel(demangled):
    """ncu's demangled name -> the studies' short kernel key."""
    s = demangled.split("(")[0].strip()
    s = re.sub(r"^void\s+", "", s)
    s = s.split("<")[0]
    parts = s.split("::")
    return "::".join(parts[-2:]) if len(parts) > 2 else s


def summarize_regime(ledger_path, out_dir, repo_prefix=""):
    """Every campaign ncu cell in the ledger -> one summary.json per config.

    Evidence here is the quick ncu set (duration + compute/memory SoL inside
    the capture window); the deep per-kernel verdicts/attribution are joined
    from the studies by kernel name. QoR = the ledger's accuracy cells.
    """
    rows = read_csv(ledger_path) if ledger_path.endswith(".csv") else \
        list(csv.DictReader(open(ledger_path), delimiter="\t"))
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(r["config"], {})[r["mode"]] = r
    study = study_context()
    verd = {}
    for r in read_csv(os.path.join(REPO, "reports/2026-07-07_substrate/substrate_verdicts.csv")):
        verd.setdefault(r["kernel"], []).append(r["substrate"])
    modal_verdict = {k: max(set(v), key=v.count) for k, v in verd.items()}

    # campaign ncu names come demangled WITHOUT the namespace the study keys
    # carry (photometric_kernel vs matcher::photometric_kernel) — index the
    # study lookups by last name component too
    by_suffix = {k.split("::")[-1]: k for k in modal_verdict}

    def study_join(kname):
        full = kname if kname in modal_verdict else by_suffix.get(kname.split("::")[-1])
        return (modal_verdict.get(full, "?"), study(full) if full else None)

    os.makedirs(out_dir, exist_ok=True)
    n = 0
    DUR = "gpu__time_duration.sum"
    SM = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    MEM = "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
    for cfg, cells in sorted(by_cfg.items()):
        ncu = cells.get("ncu", {})
        rdir = ncu.get("results_dir", "-")
        if rdir in ("-", ""):
            continue
        mpath = os.path.join(repo_prefix or REPO, rdir, "derived", "ncu_metrics.csv")
        met = read_csv(mpath)
        if not met:
            continue
        agg = {}
        for r in met:
            raw = r.get("Kernel Name", r.get("Name", ""))
            d = fnum(r.get(DUR))
            if not raw or d is None:      # header echo / units row / broken line
                continue
            kname = shorten_kernel(raw)
            a = agg.setdefault(kname, {"dur": 0.0, "sm": [], "mem": []})
            a["dur"] += d                  # gpu__time_duration.sum is in µs
            for key, dst in ((SM, "sm"), (MEM, "mem")):
                v = fnum(r.get(key))
                if v is not None:
                    a[dst].append(v)
        total = sum(a["dur"] for a in agg.values()) or 1.0
        kernels = []
        for kname, a in sorted(agg.items(), key=lambda kv: -kv[1]["dur"]):
            sm = sum(a["sm"]) / len(a["sm"]) if a["sm"] else None
            mem = sum(a["mem"]) / len(a["mem"]) if a["mem"] else None
            # A bare mem>=sm comparison mislabels near-idle kernels: 1.5% mem
            # vs 1.1% compute is not "memory-bound", it's neither-bound (the
            # wall is elsewhere — occupancy/dependencies, invisible to this
            # quick 3-metric window). Require BOTH throughputs to clear the
            # same "high" floor classify.py uses (sol_hi=40%) before calling
            # a direction at all; below that, say so explicitly.
            SOL_HI = 40.0
            if sm is None and mem is None:
                limiter, why = "?", "quick ncu window: no SoL data captured"
            elif (mem or 0) < SOL_HI and (sm or 0) < SOL_HI:
                limiter = "low-utilization"
                why = (f"quick ncu window: mem SoL {mem:.1f}% and compute SoL {sm:.1f}% "
                       "are both under 40% — neither is 'bound'; the wall is likely "
                       "occupancy/dependency (invisible to this quick metric set) — "
                       "see the deep verdict below, not this tag")
            elif (mem or 0) >= (sm or 0):
                limiter = "memory-leaning"
                why = f"quick ncu window: mem SoL {mem:.1f}% ≥ compute SoL {sm:.1f}%"
            else:
                limiter = "compute-leaning"
                why = f"quick ncu window: compute SoL {sm:.1f}% > mem SoL {mem:.1f}%"
            verdict, ctx = study_join(kname)
            kernels.append({
                "name": kname, "stage": "?",
                "time_ms": round(a["dur"] / 1e3, 3),      # µs -> ms
                "share_pct": round(100 * a["dur"] / total, 2),
                "limiter": limiter,
                "substrate": verdict,
                "pim_affinity": "?",
                "rationale": why + " — deep verdict joined from the studies by kernel name",
                "study": ctx,
                # the quick set measures the MEMORY-PIPELINE SoL, not DRAM BW
                # (dram__throughput is only in the deep sets) — label honestly
                "evidence": {"dram_sol_pct": None, "mem_sol_pct": mem,
                             "comp_sol_pct": sm, "sectors_per_req": None,
                             "lfmr": None, "mpki": None, "occupancy_pct": None,
                             "ai": None, "wset_bytes_per_launch": None,
                             "dominant_stall": ""},
                "sm_sol_pct": sm,
                "roofline": None,
            })
        qor = {}
        for mode in ("plain", "nsys", "ncu", "nvbit"):
            c = cells.get(mode)
            if c and c.get("mode_APE_m", "-") != "-":
                qor[f"{mode}_ape_m"] = fnum(c["mode_APE_m"])
        s = {"workload": cfg, "device": "dellworkstation_sm89", "adapter": "cuvslam",
             "source": rdir, "device_peaks": device_peaks("dellworkstation"),
             "qor": qor or None, "note": "campaign cell (ncu quick window)",
             "stages": [], "kernels": kernels}
        out = os.path.join(out_dir, f"{cfg}.summary.json")
        json.dump(s, open(out, "w"), indent=1)
        n += 1
    print(f"[✓] {n} campaign summaries -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legacy", help="one device-report dir (uses its data/)")
    ap.add_argument("--results", help="one harness results dir (uses its derived/)")
    ap.add_argument("--all-legacy", action="store_true",
                    help="every profiling/reports/* with data/classification.csv")
    ap.add_argument("--regime", help="validation-regime ledger (REGIME.tsv) -> "
                                     "one summary per campaign config")
    ap.add_argument("--regime-out", default="reports/2026-07-08_campaign_runs",
                    help="output dir for --regime summaries")
    args = ap.parse_args()

    if args.regime:
        summarize_regime(args.regime, os.path.join(REPO, args.regime_out))
        return

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
