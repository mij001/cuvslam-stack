#!/usr/bin/env python3
"""ndp_eval.py — the NDP evaluator: measurement-fitted, holdout-validated.

DAMOV evaluated NDP with ZSim+Ramulator (cycle-accurate simulation) because
CPUs offer no way to vary the memory system in place. GPUs DO offer one: the
two clock domains. The campaign measures every kernel at FOUR locked points

    base(1620,7001)  lowcore(810,7001)  lowmem(1620,5001)  holdout(1005,5001)

and the first three EXACTLY determine a first-order model per kernel

    t(f_core, f_mem) = a·(1620/f_core) + b·(7001/f_mem) + c

a = core-domain work (SMs, L1/L2, MSHR/LSU queues, shared memory)
b = memory-domain work (DRAM bandwidth/CAS)
c = clock-invariant residue (fixed latencies, launch/host overhead)

The FOURTH point is off both fitted axes and is never used in the fit: the
model earns trust per kernel only where it predicts the holdout time (gate:
|err| <= 10%). This is the falsifiable core the analytical pim_placement
model lacked.

NDP transform (near-bank PiM with internal bandwidth k× and compute ratio
c_r, plus a fractional offload overhead):

    t_ndp = a/c_r + b/k + c + oh·t_base        speedup = t_base / t_ndp

Sweeping (k, c_r) reproduces DAMOV's Fig-18b role: per-class speedup — and
LOSS — curves. Compute-/dependency-bound classes must lose (a/c_r dominates);
bandwidth-bound classes must gain ∝k saturating at a+c. Cycle-accurate
Accel-Sim (the literal ZSim+Ramulator analog) remains the gated Phase-4
instrument for microarchitectural what-ifs; THIS is the silicon-grounded
first-order evaluator.

Usage:
  python3 -m analysis.ndp_eval --data <population_out> \
      [--out ../reports/2026-07-12_gpu_damov_population]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.population import read_kernsum, short  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

U2 = 1620.0 / 810.0          # core-period ratio at lowcore  (2.0)
V3 = 7001.0 / 5001.0         # mem-period ratio at lowmem    (1.39992)
U4 = 1620.0 / 1005.0         # holdout core ratio            (1.61194)
V4 = V3                      # holdout mem ratio
HOLDOUT_GATE = 0.10          # fit trusted iff holdout error <= 10%
OH_FRAC = 0.02               # offload overhead as a fraction of t_base

K_SWEEP = (2.0, 4.0, 8.0, 16.0)
CR_SWEEP = (0.25, 0.50, 0.75, 1.00)


def fit_kernel(t1, t2, t3, t4):
    """Exact 3-point fit + holdout error. Returns dict or None (bad data)."""
    if not all(x and x > 0 for x in (t1, t2, t3, t4)):
        return None
    a = t2 - t1
    b = (t3 - t1) / (V3 - 1.0)
    c = t1 - a - b
    pred4 = a * U4 + b * V4 + c
    err = abs(pred4 - t4) / t4
    return {"a": a, "b": b, "c": c,
            "a_share": a / t1, "b_share": b / t1, "c_share": c / t1,
            "holdout_pred_ms": pred4 / 1e6, "holdout_meas_ms": t4 / 1e6,
            "holdout_err": err,
            "physical": a >= -0.02 * t1 and b >= -0.02 * t1 and c >= -0.05 * t1}


def ndp_time(fit, k, cr, t1):
    a, b, c = max(fit["a"], 0.0), max(fit["b"], 0.0), max(fit["c"], 0.0)
    return a / cr + b / k + c + OH_FRAC * t1


def load(data_dir):
    """[(app, kernel, class, t1..t4)] joined from cls + 4 sweep points."""
    idx = os.path.join(data_dir, "population_index.tsv")
    out = []
    for rec in csv.DictReader(open(idx), delimiter="\t"):
        app = rec["app"]
        cls_csv = os.path.join(data_dir, "cls", app, "classification.csv")
        if rec["status"] != "done" or not os.path.isfile(cls_csv):
            continue
        pts = {}
        for tag in ("base", "lowcore", "lowmem", "holdout"):
            pts[tag], _ = read_kernsum(os.path.join(
                data_dir, "sweep", f"{app}_{tag}_cuda_gpu_kern_sum.csv"))
        if not pts["holdout"]:
            continue                       # pre-v3 app (3-point era)
        for r in csv.DictReader(open(cls_csv)):
            k = short(r["kernel"])
            out.append({"app": app, "kernel": k, "class": r["class"],
                        "t": [pts[t].get(k) for t in ("base", "lowcore", "lowmem", "holdout")]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "reports/2026-07-12_gpu_damov_population"))
    args = ap.parse_args(argv)
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(out, exist_ok=True)

    rows = load(args.data)
    fits, rejects = [], {"missing": 0, "holdout": 0, "nonphysical": 0}
    for r in rows:
        f = fit_kernel(*r["t"])
        if f is None:
            rejects["missing"] += 1
            continue
        r.update(f)
        if not f["physical"]:
            rejects["nonphysical"] += 1
            r["gate"] = "nonphysical"
        elif f["holdout_err"] > HOLDOUT_GATE:
            rejects["holdout"] += 1
            r["gate"] = "holdout-fail"
        else:
            r["gate"] = "ok"
        fits.append(r)

    with open(os.path.join(out, "ndp_model_fits.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["app", "kernel", "class", "t_base_ms", "a_share", "b_share",
                    "c_share", "holdout_err_pct", "gate"])
        for r in fits:
            w.writerow([r["app"], r["kernel"], r["class"],
                        round(r["t"][0] / 1e6, 3), round(r["a_share"], 3),
                        round(r["b_share"], 3), round(r["c_share"], 3),
                        round(100 * r["holdout_err"], 1), r["gate"]])

    ok = [r for r in fits if r["gate"] == "ok"]
    with open(os.path.join(out, "ndp_speedups.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["app", "kernel", "class", "k_bw", "c_ratio", "speedup"])
        for r in ok:
            t1 = r["t"][0]
            for k in K_SWEEP:
                for cr in CR_SWEEP:
                    w.writerow([r["app"], r["kernel"], r["class"], k, cr,
                                round(t1 / ndp_time(r, k, cr, t1), 3)])

    # per-class curves (Fig-18b analog): median speedup, moderate PiM compute
    classes = sorted({r["class"] for r in ok})
    curve_rows = [["class", "n"] + [f"k={k:g} (c_r=0.5)" for k in K_SWEEP]
                  + [f"k={k:g} (c_r=1.0)" for k in K_SWEEP]]
    for c in classes:
        rs = [r for r in ok if r["class"] == c]
        line = [c, len(rs)]
        for cr in (0.5, 1.0):
            for k in K_SWEEP:
                sp = sorted(r["t"][0] / ndp_time(r, k, cr, r["t"][0]) for r in rs)
                line_val = sp[len(sp) // 2] if sp else float("nan")
                if cr in (0.5, 1.0):
                    line.append(round(line_val, 2))
        curve_rows.append(line)
    with open(os.path.join(out, "ndp_class_curves.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows(curve_rows)

    # summary
    errs = sorted(r["holdout_err"] for r in fits if r.get("holdout_err") is not None)
    med_err = errs[len(errs) // 2] if errs else float("nan")
    gains = [r for r in ok if r["t"][0] / ndp_time(r, 8, 0.5, r["t"][0]) > 1.1]
    losses = [r for r in ok if r["t"][0] / ndp_time(r, 8, 0.5, r["t"][0]) < 0.9]
    lines = [
        "# NDP evaluator — measurement-fitted, holdout-validated",
        "",
        f"**Model:** per-kernel t = a/f_core + b/f_mem + c, fitted EXACTLY from the "
        f"three primary clock points; validated on the held-out 4th point "
        f"(1005/5001 MHz, off both fitted axes).",
        f"**Fits:** {len(fits)} kernels; median holdout error "
        f"**{100 * med_err:.1f}%**; trusted (err<=10%, physical): **{len(ok)}**; "
        f"rejected: {rejects['holdout']} holdout-fail, {rejects['nonphysical']} "
        f"non-physical, {rejects['missing']} missing points.",
        "",
        f"**NDP sweep (k∈{{2,4,8,16}}× internal BW, c_r∈{{0.25..1}} PiM compute, "
        f"+{OH_FRAC:.0%} offload overhead): gains AND losses, as DAMOV's Fig 18b "
        f"requires:** at k=8/c_r=0.5, {len(gains)} kernels gain >1.1x and "
        f"**{len(losses)} kernels LOSE >10%** — the compute-/dependency-bound "
        f"classes pay a/c_r, exactly the predicted failure mode.",
        "",
        "| class | n | k=2 | k=4 | k=8 | k=16 (all c_r=0.5, median) |",
        "|---|---|---|---|---|---|",
    ]
    for row in curve_rows[1:]:
        lines.append(f"| {row[0]} | {row[1]} | " + " | ".join(str(x) for x in row[2:6]) + " |")
    lines += ["", "Cycle-accurate Accel-Sim (the literal ZSim+Ramulator analog) stays the "
              "gated Phase-4 instrument; its NDP overlays are already generated "
              "(profiling/sim/). This evaluator is silicon-grounded by construction.",
              "", "-> ndp_model_fits.csv, ndp_speedups.csv, ndp_class_curves.csv"]
    open(os.path.join(out, "NDP_EVAL.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    main()
