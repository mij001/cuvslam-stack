#!/usr/bin/env python3
"""migrate_g2_guard.py — apply the G2 DRAM-visibility guard to COMMITTED tables.

The 2026-07-12 real-codebase population exposed L2-RESIDENT scatter: kernels
with sectors/request >= the scatter threshold whose LFMR < lfmr_lo — the L2
absorbs the gather, the clock response is pure core-domain, and the correct
class is G3 (DAMOV's high-temporal-locality cache-capacity family), not G2.
classify.py now guards the G2 rule accordingly.

The committed device-report classification.csvs were produced by the original
multi-window pipeline and CANNOT be regenerated from data/screen.csv alone
(the SLAM-window kernels — st_track etc. — were merged from dedicated
captures; a naive re-run drops them and substitutes window times: discovered
the hard way, reverted). This migration applies the SAME rule change to those
tables IN PLACE, deterministically from their own columns, preserving every
row, time, and provenance field:

    class == G2-coalescing  AND  lfmr_gpu < lfmr_lo (0.35)
      ->  class = G3-l2-reuse (confidence medium)
          rationale amended; pim_affinity/substrate recomputed via
          classify.pim_affinity(G3, persistence, features)

Usage:  python3 -m analysis.migrate_g2_guard [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import classify  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS = [
    "profiling/reports/2026-07-02_tum_office_mx450",
    "profiling/reports/2026-07-03_kitti06_rtx2000ada",
    "profiling/reports/2026-07-03_tum_office_rtx2000ada",
    "profiling/reports/2026-07-03_tumvi_corridor1_rtx2000ada",
]


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    lfmr_lo = classify.THRESHOLDS["lfmr_lo"]

    total = 0
    for rep in REPORTS:
        p = os.path.join(REPO, rep, "data", "classification.csv")
        rows = list(csv.DictReader(open(p)))
        fields = rows[0].keys()
        flips = []
        for r in rows:
            lf = fnum(r.get("lfmr_gpu"))
            if r["class"] == "G2-coalescing" and lf == lf and lf < lfmr_lo:
                feats = {"sect": fnum(r.get("sectors_per_req")), "lfmr": lf,
                         "wset": fnum(r.get("dram_bytes_per_launch"))}
                aff, sub = classify.pim_affinity("G3-l2-reuse",
                                                 r.get("persistence", ""), feats)
                r["class"] = "G3-l2-reuse"
                r["confidence"] = "medium"
                r["pim_affinity"] = aff
                r["substrate"] = sub
                r["rationale"] = (f"memory-limited but LFMR {lf:.2f} — the L2 absorbs even the "
                                  f"{feats['sect']:.0f} sect/req scatter "
                                  "[migrated: G2 DRAM-visibility guard, 2026-07-12 population]")
                flips.append(r["kernel"])
        total += len(flips)
        print(f"{rep.split('/')[-1]}: {len(flips)} G2->G3"
              + (f"  ({', '.join(k.split('::')[-1] for k in flips[:4])}"
                 f"{'…' if len(flips) > 4 else ''})" if flips else ""))
        if flips and not args.dry_run:
            with open(p, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
    print(f"[{'DRY' if args.dry_run else '✓'}] {total} rows migrated across {len(REPORTS)} reports")


if __name__ == "__main__":
    main()
