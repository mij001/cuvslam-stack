#!/usr/bin/env python3
"""compare_ndp.py — the three-rung NDP agreement table (the capstone check).

Three independent instruments now speak about the same question — what does a
near-bank NDP substrate do to each bottleneck class?

  rung 1  TAXONOMY   the class's predicted direction (G1/G2 gain, G3/G5/G6/G7
                     lose or flat, G4 latency-helped) — from the decision tree
  rung 2  FITTED     the measurement-fitted first-order model (a/f_core +
          MODEL      b/f_mem + c from the 3-point clock sweep on real silicon)
                     with the NDP transform t = a/c_r + b/k + c
  rung 3  ACCEL-SIM  cycle-level trace-driven simulation (the ZSim+Ramulator
                     role): baseline vs NDP overlay configs on real SASS traces

If the three do not tell one story per class, something is wrong with the
class, the model, or the sim — that is the point of having all three.

Scenario mapping (identical knobs across rung 2 and 3):
  conservative  k=4,  c_r=0.50      moderate  k=8, c_r=0.75

Usage:
  python3 profiling/sim/compare_ndp.py \
      --sweep reports/2026-07-09_damov_validation/clock_sweep.csv \
      --sim   <ndp_sim_results.csv> \
      --out   reports/2026-07-13_ndp_three_rungs
"""
from __future__ import annotations

import argparse
import csv
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

V3 = 7001.0 / 5001.0
SCEN = {"ndp_cons": (4.0, 0.50), "ndp_mod": (8.0, 0.75)}
CLASS_OF = {"g1_triad": "G1", "g2_gather": "G2", "g3_l2": "G3", "g4_chase": "G4",
            "g5_fma": "G5", "g6_shared": "G6", "g7_dep": "G7", "g0_tiny": "G0"}
# rung-1 predictions: expected NDP direction per class (near-bank substrate)
TAXO = {"G1": "gain", "G2": "gain (concurrency near banks)", "G3": "lose (loses its L2)",
        "G4": "gain (latency shrinks near banks)", "G5": "lose (pays c_r)",
        "G6": "lose/flat (on-chip, pays c_r)", "G7": "lose/flat (dep chain, pays c_r)",
        "G0": "n/a (screened)"}


def model_speedup(t1, t2, t3, k, cr):
    a = t2 - t1
    b = (t3 - t1) / (V3 - 1.0)
    c = t1 - a - b
    a, b, c = max(a, 0.0), max(b, 0.0), max(c, 0.0)
    return t1 / (a / cr + b / k + c + 0.02 * t1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default="reports/2026-07-09_damov_validation/clock_sweep.csv")
    ap.add_argument("--sim", required=True)
    ap.add_argument("--out", default="reports/2026-07-13_ndp_three_rungs")
    args = ap.parse_args(argv)
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(out, exist_ok=True)

    sweep = {}
    for r in csv.DictReader(open(os.path.join(REPO, args.sweep))):
        sweep[(r["archetype"], int(r["gfx_mhz"]), int(r["mem_mhz"]))] = \
            float(r["mean_kernel_ms"])
    sim = {}
    for r in csv.DictReader(open(args.sim)):
        if r["cycles"]:
            sim[(r["archetype"], r["config"])] = float(r["cycles"])

    rows = [["archetype", "class", "rung1 taxonomy",
             "rung2 model cons", "rung3 sim cons",
             "rung2 model mod", "rung3 sim mod", "story"]]
    agree = n = 0
    for arch, g in CLASS_OF.items():
        if g == "G0":
            continue
        t1 = sweep.get((arch, 1620, 7001))
        t2 = sweep.get((arch, 810, 7001))
        t3 = sweep.get((arch, 1620, 5001))
        base = sim.get((arch, "baseline"))
        line = [arch, g, TAXO[g]]
        sims, mods = [], []
        for scen, (k, cr) in SCEN.items():
            m = model_speedup(t1, t2, t3, k, cr) if all((t1, t2, t3)) else None
            # cycles are counted in the CORE clock domain: a half-speed PiM
            # core needs the same cycles twice as long — convert to TIME
            # (t = cycles/f_core, f_ndp = c_r*f_base) before comparing
            s = (base / sim[(arch, scen)]) * cr if base and (arch, scen) in sim else None
            mods.append(m); sims.append(s)
        line[3:3] = []  # keep order
        line = [arch, g, TAXO[g],
                f"{mods[0]:.2f}x" if mods[0] else "-",
                f"{sims[0]:.2f}x" if sims[0] else "-",
                f"{mods[1]:.2f}x" if mods[1] else "-",
                f"{sims[1]:.2f}x" if sims[1] else "-"]
        # one story? direction agreement at moderate scenario
        story = "?"
        if mods[1] and sims[1]:
            dm = "gain" if mods[1] > 1.05 else ("lose" if mods[1] < 0.95 else "flat")
            ds = "gain" if sims[1] > 1.05 else ("lose" if sims[1] < 0.95 else "flat")
            taxo_dir = TAXO[g].split()[0].rstrip("/flat")
            story = "AGREE" if (dm == ds and (taxo_dir.startswith(dm)
                     or (taxo_dir == "lose" and dm in ("lose", "flat"))
                     or (taxo_dir == "gain" and dm == "gain"))) else f"model={dm} sim={ds}"
            n += 1
            agree += story == "AGREE"
        line.append(story)
        rows.append(line)

    with open(os.path.join(out, "ndp_three_rungs.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    md = ["# NDP, three independent rungs — one story per class?", "",
          "| archetype | class | taxonomy says | model k4/c.5 | SIM k4/c.5 | "
          "model k8/c.75 | SIM k8/c.75 | verdict |",
          "|---|---|---|---|---|---|---|---|"]
    md += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows[1:]]
    md += ["", f"**{agree}/{n} classes tell one story across taxonomy, fitted model, "
           f"and cycle-level simulation.**"]
    open(os.path.join(out, "THREE_RUNGS.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
