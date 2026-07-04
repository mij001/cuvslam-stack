#!/usr/bin/env python3
"""plan_gapfill.py — plan launch windows that cover attribution-coverage gaps.

The campaign's pass2a/pass2b windows are slices, so kernels that fire sparsely
(keyframe refresh GFTT, SBA bursts, loop-closure scans) or a window that landed
in a degenerate stretch (front-end-only frames) can leave a sequence's joins
without rows for some kernels. This planner reads a sequence's pass1 launch map
(every launch of the full run) and the join CSVs already produced, computes the
set of cuVSLAM kernels absent from ALL joins, and emits up to --max-windows
launch windows that together cover at least one launch of every missing kernel.

Output (stdout): one "BEGIN END n_missing_covered" line per window; nothing if
the sequence has no gaps. Consumed by ws_attribution_gapfill.sh.

Usage: python3 -m campaign.plan_gapfill /mnt/data/attribution_out/<seq>
       (run from profiling/; also works as a path argument from anywhere)
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import base_kernel_name  # noqa: E402

_LAUNCH = re.compile(r"Kernel name (.+?) - grid launch id (\d+)")
# library kernels (CUB sort, cuSOLVER) ride along inside covered windows; the
# gap plan is driven by cuVSLAM's own kernels
_LIB = re.compile(r"^(cub::|trsv|dtrsv|getrf|xxtrf|copy_info|.*wo_pivot)")


def launchmap_ids(path):
    """kernel -> sorted launch ids, from a pass1 launch map."""
    ids = {}
    proc = subprocess.Popen(["zstdcat", path], stdout=subprocess.PIPE,
                            text=True, errors="replace")
    for line in proc.stdout:
        m = _LAUNCH.search(line)
        if m:
            ids.setdefault(base_kernel_name(m.group(1)), []).append(int(m.group(2)))
    proc.wait()
    return ids


def joined_kernels(seq_dir):
    ks = set()
    for p in glob.glob(os.path.join(seq_dir, "join_*", "attribution.csv")):
        with open(p) as fh:
            ks.update(r["kernel"] for r in csv.DictReader(fh))
    return ks


def plan(seq_dir, max_windows=4, margin=30, span=3000):
    lm = os.path.join(seq_dir, "pass1_launchmap.txt.zst")
    if not os.path.isfile(lm):
        return []
    ids = launchmap_ids(lm)
    have = joined_kernels(seq_dir)
    missing = {k: v for k, v in ids.items()
               if k not in have and not _LIB.match(k)}
    if not missing:
        return []
    # greedy: repeatedly place the window [id-margin, id+span) that covers the
    # most still-missing kernels; ties break on the number of missing-kernel
    # LAUNCHES inside the window — sparse kernels drift between runs, so an
    # isolated occurrence is a bad anchor while a dense cluster is robust
    windows = []
    while missing and len(windows) < max_windows:
        best, best_cover, best_hits = None, set(), 0
        for k, v in missing.items():
            for seed in v[:: max(1, len(v) // 8)]:      # subsample seeds
                b, e = max(0, seed - margin), seed + span
                cover = set()
                hits = 0
                for k2, v2 in missing.items():
                    n = sum(1 for i in v2 if b <= i < e)
                    if n:
                        cover.add(k2)
                        hits += n
                if (len(cover), hits) > (len(best_cover), best_hits):
                    best, best_cover, best_hits = (b, e), cover, hits
        if not best:
            break
        windows.append((best[0], best[1], len(best_cover)))
        for k in best_cover:
            del missing[k]
    if missing:
        print(f"# WARNING: {len(missing)} kernel(s) still uncovered after "
              f"{max_windows} windows: {sorted(missing)[:6]}", file=sys.stderr)
    return windows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("seq_dir")
    ap.add_argument("--max-windows", type=int, default=4)
    ap.add_argument("--span", type=int, default=3000,
                    help="window length in launches (default 3000)")
    args = ap.parse_args(argv)
    for b, e, n in plan(args.seq_dir, args.max_windows, span=args.span):
        print(b, e, n)


if __name__ == "__main__":
    main()
