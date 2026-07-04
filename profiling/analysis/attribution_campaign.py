#!/usr/bin/env python3
"""attribution_campaign.py — cross-sequence synthesis of the attribution joins.

Reads every sequence directory under the campaign output root (each holding
join_steady_state/, join_st_scans/ and optional join_gapfill_*/ produced by
ws_attribution_all.sh + ws_attribution_gapfill.sh) and answers the campaign
question: is the per-kernel data-structure composition a property of the
KERNEL, or of the sequence?

Merging rule: within one sequence a kernel may appear in several joins
(overlapping windows); the join with the most warp accesses for that kernel
wins — maximum coverage, no double counting.

Outputs (--out):
  attribution_by_sequence.csv   kernel × sequence × tag long table (merged)
  attribution_consistency.csv   per kernel: modal top-GLOBAL-tag, agreement %,
                                median space split (global/local/shared)
  coverage.csv                  kernel × sequence capture matrix
  CAMPAIGN_ATTRIBUTION.md       the summary tables

Stdlib-only, GPU-free: runs anywhere against the campaign CSVs.

Usage: python3 -m analysis.attribution_campaign ROOT --out DIR
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common  # noqa: E402

SPACE_TAGS = ("shared_onchip", "local_spill")


def load_joins(seq_dir):
    """kernel -> {tag: (warp_accesses, sectors)} merged across the joins."""
    per_join = []                     # [(join_name, {kernel: {tag: (acc, sect)}})]
    for jd in sorted(glob.glob(os.path.join(seq_dir, "join_*"))):
        p = os.path.join(jd, "attribution.csv")
        if not os.path.isfile(p):
            continue
        kt = defaultdict(dict)
        with open(p) as fh:
            for r in csv.DictReader(fh):
                kt[r["kernel"]][r["tag"]] = (int(r["warp_accesses"]),
                                             int(r["sectors"]))
        per_join.append((os.path.basename(jd), kt))
    merged = {}
    for _name, kt in per_join:
        for kernel, tags in kt.items():
            acc = sum(a for a, _s in tags.values())
            if kernel not in merged or acc > merged[kernel][0]:
                merged[kernel] = (acc, tags)
    return {k: tags for k, (_a, tags) in merged.items()}


def composition(tags):
    """(space_split, global_tags) as sector fractions of the kernel's traffic."""
    total = sum(s for _a, s in tags.values()) or 1
    space = {t: tags.get(t, (0, 0))[1] / total for t in SPACE_TAGS}
    glob_total = sum(s for t, (_a, s) in tags.items() if t not in SPACE_TAGS)
    gtags = {t: s / glob_total for t, (_a, s) in tags.items()
             if t not in SPACE_TAGS and glob_total}
    return space, gtags


def synthesize(root):
    seqs = {}
    for d in sorted(glob.glob(os.path.join(root, "*", ""))):
        name = os.path.basename(d.rstrip("/"))
        merged = load_joins(d)
        if merged:
            seqs[name] = merged
    return seqs


def emit(seqs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    long_rows, cons_rows, cov_rows = [], [], []
    kernels = sorted({k for m in seqs.values() for k in m})
    seq_names = sorted(seqs)

    for kernel in kernels:
        top_tags, spaces = [], []
        for seq in seq_names:
            tags = seqs[seq].get(kernel)
            if not tags:
                continue
            space, gtags = composition(tags)
            spaces.append(space)
            total = sum(s for _a, s in tags.values()) or 1
            for tag, (acc, sect) in sorted(tags.items(), key=lambda x: -x[1][1]):
                long_rows.append([kernel, seq, tag, acc, sect,
                                  round(100 * sect / total, 1)])
            if gtags:
                top_tags.append(max(gtags, key=gtags.get))
        n = len(top_tags)
        modal, agree = ("", 0.0)
        if n:
            modal, cnt = Counter(top_tags).most_common(1)[0]
            agree = cnt / n
        med_shared = sorted(s["shared_onchip"] for s in spaces)[len(spaces) // 2] if spaces else 0
        med_local = sorted(s["local_spill"] for s in spaces)[len(spaces) // 2] if spaces else 0
        cons_rows.append([kernel, len(spaces), n, modal, round(100 * agree, 1),
                          round(100 * med_shared, 1), round(100 * med_local, 1),
                          round(100 * (1 - med_shared - med_local), 1)])
        cov_rows.append([kernel] + [1 if kernel in seqs[s] else 0 for s in seq_names])

    p1 = os.path.join(out_dir, "attribution_by_sequence.csv")
    common.write_csv(p1, ["kernel", "sequence", "tag", "warp_accesses",
                          "sectors", "pct_kernel_traffic"], long_rows)
    p2 = os.path.join(out_dir, "attribution_consistency.csv")
    common.write_csv(p2, ["kernel", "n_sequences", "n_with_global",
                          "modal_top_global_tag", "agreement_pct",
                          "med_shared_pct", "med_local_spill_pct",
                          "med_global_pct"], cons_rows)
    p3 = os.path.join(out_dir, "coverage.csv")
    common.write_csv(p3, ["kernel"] + seq_names, cov_rows)

    # the summary markdown: cuVSLAM kernels with global traffic, by agreement
    md = [f"# Cross-sequence attribution synthesis — {len(seq_names)} sequences\n"]
    md.append("| kernel | seqs | modal top global tag | agree | shared | spill | global |")
    md.append("|---|---|---|---|---|---|---|")
    for r in sorted((r for r in cons_rows if r[2]), key=lambda r: (-r[4], r[0])):
        md.append(f"| {r[0]} | {r[1]} | {r[3]} | {r[4]:.0f}% "
                  f"| {r[5]:.0f}% | {r[6]:.0f}% | {r[7]:.0f}% |")
    p4 = os.path.join(out_dir, "CAMPAIGN_ATTRIBUTION.md")
    with open(p4, "w") as fh:
        fh.write("\n".join(md) + "\n")
    return [p1, p2, p3, p4], cons_rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="campaign output root (per-sequence dirs)")
    ap.add_argument("--out", default=".")
    args = ap.parse_args(argv)
    seqs = synthesize(args.root)
    if not seqs:
        sys.exit(f"no join CSVs under {args.root}")
    files, cons = emit(seqs, args.out)
    for p in files:
        print(f"[✓] {p}")
    with_global = [r for r in cons if r[2]]
    perfect = sum(1 for r in with_global if r[4] == 100.0)
    print(f"    {len(seqs)} sequences, {len(cons)} kernels; "
          f"{perfect}/{len(with_global)} kernels have a unanimous top global tag")


if __name__ == "__main__":
    main()
