#!/usr/bin/env python3
"""compare.py — does the classification generalize across datasets?

Joins per-kernel GPU-DAMOV classifications from multiple workloads (the
characterization matrix: e.g. TUM RGBD indoor / KITTI stereo outdoor / TUM-VI
fisheye building) and reports: which kernels appear where, whether each keeps
its class, and the overall agreement — the evidence that the characterization
describes cuVSLAM rather than one configuration of it. Kernel sets legitimately
differ per mode (RGBD has matchers, stereo has LK-horizontal); agreement is
computed over the kernels shared by ≥2 workloads.

Inputs are `classification.csv` files (from analysis.classify / make_report),
so this runs from committed report data with no GPU and no datasets.

Usage:
  python3 -m analysis.compare LABEL=path/to/classification.csv [...] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common  # noqa: E402


def load(spec: str):
    label, path = spec.split("=", 1)
    rows = {}
    for r in csv.DictReader(open(path)):
        rows[r["kernel"]] = {"class": r["class"], "pim": r["pim_affinity"],
                             "conf": r["confidence"],
                             "stability": r.get("stability", ""),
                             "time_ms": float(r["time_ms"] or 0)}
    return label, rows


def run(specs):
    data = dict(load(s) for s in specs)
    labels = list(data)
    kernels = sorted({k for rows in data.values() for k in rows})
    table, n_shared, n_agree = [], 0, 0
    for k in kernels:
        classes = {lab: data[lab][k]["class"] for lab in labels if k in data[lab]}
        present = list(classes.values())
        agree = len(set(present)) == 1
        if len(present) >= 2:
            n_shared += 1
            n_agree += agree
        table.append([k] + [classes.get(lab, "—") for lab in labels] +
                     ["✓" if agree and len(present) >= 2 else
                      ("✗ " + "/".join(sorted(set(present))) if len(present) >= 2 else "n/a")])
    return labels, table, n_shared, n_agree


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("specs", nargs="+", help="LABEL=classification.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    labels, table, n_shared, n_agree = run(args.specs)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        p = os.path.join(args.out, "class_agreement.csv")
        common.write_csv(p, ["kernel"] + labels + ["agreement"], table)
        print(f"[✓] {p}")
    print(common.md_table(["kernel"] + labels + ["agree"], table))
    print(f"\nagreement: {n_agree}/{n_shared} kernels shared by ≥2 workloads keep "
          f"their class ({100*n_agree/max(n_shared,1):.0f}%)")


if __name__ == "__main__":
    main()
