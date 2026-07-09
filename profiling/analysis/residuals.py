#!/usr/bin/env python3
"""residuals.py — the Layer-3 target: which kernels' DRAM traffic is still
UNMAPPED, and how much is at stake.

The three-layer attribution (attribution.py) resolves most global traffic to a
named data structure via the TaggedAllocator journal. What it can't name is
traffic to allocations it never saw: `__device__` module globals, texture-path
reads, driver-internal buffers — these land in the `unmapped` / `untagged_driver`
tags. Layer-3 (kernel-argument correlation, THESIS G5 / Path C3) resolves them
by matching each launch's pointer ARGUMENTS to the live allocation set.

This script is the GROUNDWORK for that: it reads the committed campaign
attribution (attribution_by_sequence.csv), aggregates each kernel's unmapped +
untagged-driver share of its own DRAM traffic (mean across sequences, so it is a
stable property not a one-run artifact), and ranks the kernels Layer-3 must
resolve — with the traffic at stake. It turns "~40% is unmapped somewhere" into
a named, prioritized target list, and it runs NOW on committed data (no GPU).

  python3 -m analysis.residuals            # print the ranked target list
  python3 -m analysis.residuals --out reports/<date>_residuals

── Layer-3 join interface (pinned here so the future NVBit tool has a target) ──
The NVBit kernel-arg tool must emit `kernel_args.csv`:
    grid_launch_id, kernel, arg_index, arg_ptr, arg_bytes_hint
Then the resolver reuses attribution.py's LiveSet: for each (launch, arg_ptr),
`LiveSet.find(arg_ptr)` at that launch id → the owning allocation's tag. A
kernel's currently-`unmapped` sectors are re-attributed to whichever arg's
allocation contains the faulting address range. No new join math — the same
address→allocation map, keyed by the arguments instead of the trace stream.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BY_SEQ = "profiling/reports/2026-07-05_attribution_campaign/attribution_by_sequence.csv"
UNRESOLVED = {"unmapped", "untagged_driver"}


def fnum(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def load(path):
    p = os.path.join(REPO, path)
    if not os.path.isfile(p):
        raise SystemExit(f"no attribution data at {path} — run the attribution campaign first")
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def rank(rows):
    """kernel -> (mean unmapped-share %, mean total sectors, n sequences seen)."""
    per_kernel_seq = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))  # k -> seq -> [unmapped_pct, sectors]
    for r in rows:
        k, seq = r["kernel"], r["sequence"]
        pct = fnum(r.get("pct_kernel_traffic"))
        sect = fnum(r.get("sectors"))
        if r.get("tag") in UNRESOLVED:
            per_kernel_seq[k][seq][0] += pct
        per_kernel_seq[k][seq][1] += sect
    out = []
    for k, seqs in per_kernel_seq.items():
        unmapped = [v[0] for v in seqs.values()]
        sectors = [v[1] for v in seqs.values()]
        mean_unmapped = sum(unmapped) / len(unmapped) if unmapped else 0.0
        mean_sectors = sum(sectors) / len(sectors) if sectors else 0.0
        out.append({
            "kernel": k, "n_sequences": len(seqs),
            "mean_unmapped_pct": round(mean_unmapped, 1),
            "mean_total_sectors": round(mean_sectors, 1),
            # traffic at stake = the unmapped fraction of this kernel's sectors
            "unmapped_sectors": round(mean_sectors * mean_unmapped / 100.0, 1),
        })
    # rank by absolute unmapped traffic, then by share
    out.sort(key=lambda r: (-r["unmapped_sectors"], -r["mean_unmapped_pct"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=10.0,
                    help="only list kernels with mean unmapped share >= this %% (default 10)")
    ap.add_argument("--out", default=None, help="also write residuals.csv here")
    args = ap.parse_args(argv)

    ranked = rank(load(BY_SEQ))
    targets = [r for r in ranked if r["mean_unmapped_pct"] >= args.threshold]

    print(f"Layer-3 targets — kernels with >= {args.threshold:g}% unmapped DRAM traffic "
          f"(mean across sequences):\n")
    print(f"  {'kernel':40} {'unmapped%':>9} {'unmapped sectors':>17} {'seqs':>5}")
    for r in targets:
        print(f"  {r['kernel'][:40]:40} {r['mean_unmapped_pct']:>9} "
              f"{r['unmapped_sectors']:>17} {r['n_sequences']:>5}")
    total_unmapped = sum(r["unmapped_sectors"] for r in ranked)
    hit = sum(r["unmapped_sectors"] for r in targets)
    print(f"\n  {len(targets)} kernels carry {hit:.0f} of {total_unmapped:.0f} "
          f"unmapped sectors ({100*hit/total_unmapped:.0f}% of the Layer-3 target) — "
          f"resolve these first.")

    if args.out:
        os.makedirs(os.path.join(REPO, args.out), exist_ok=True)
        p = os.path.join(REPO, args.out, "residuals.csv")
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ranked[0]))
            w.writeheader()
            w.writerows(ranked)
        print(f"\n[✓] {os.path.relpath(p, REPO)}")


if __name__ == "__main__":
    main()
