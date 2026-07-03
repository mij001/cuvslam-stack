#!/usr/bin/env python3
"""locality.py — DAMOV Step-2 locality analysis of NVBit mem_trace streams.

Consumes the (optionally zstd-compressed) text output of NVBit's mem_trace
tool (with the launch-window patch, blocked/mem_trace_launch_window.patch) and
produces, per kernel:

  * exact working-set FOOTPRINT (unique 32 B sectors × 32) — turns the
    "traffic ≈ footprint" inference of the counter-based reports into a
    direct measurement;
  * REUSE-DISTANCE CDF vs cache capacity — for each access, the number of
    distinct sectors touched since that sector's previous access (LRU stack
    distance); the CDF evaluated at cache sizes {64 KiB … 24 MiB} is the
    architecture-independent predicted-hit-rate-vs-cache-size curve — THE
    crossover curve for the PiM/ISP argument (cf. hw descriptors' L2 sizes);
  * intra-warp SPATIAL locality — unique sectors per warp access (1 = fully
    coalesced, 32 = fully scattered), the trace-exact version of ncu's
    sectors/request;
  * inter-launch footprint OVERLAP (Jaccard) — for cold-persistent kernels:
    do successive scans re-touch the same data (streaming-friendly, ISP-able)
    or hop randomly?

Granularity follows the GPU adaptation study §4: per-warp issue order within a
launch, 32 B sectors, global memory only (mem_trace already excludes shared/
local). Reuse distance uses the classic last-access + Fenwick-tree algorithm
(O(log n)/access); accesses beyond --max-accesses per launch are skipped with
a note (bounded memory/time on TB-scale traces).

Stdlib only. zstd files are read via the system `zstdcat`.

Usage:
  python3 -m analysis.locality TRACE[.zst] [--kernel REGEX] [--max-launches N]
      [--max-accesses M] [--out DIR]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common  # noqa: E402

SECTOR = 32                     # bytes; GPU cache sector granularity
CAPACITIES = [64 * 1024, 128 * 1024, 512 * 1024, 1 << 20, 6 << 20, 12 << 20,
              24 << 20, 48 << 20]

_LAUNCH = re.compile(r"MEMTRACE: .* - LAUNCH - .*Kernel name (\S+) - grid launch id (\d+)")
_ACCESS = re.compile(r"MEMTRACE: CTX \S+ - grid_launch_id (\d+) - CTA \S+ - warp \d+ - (\S+) - (.*)")


class Fenwick:
    def __init__(self, n):
        self.n = n
        self.t = [0] * (n + 1)

    def add(self, i, v):
        i += 1
        while i <= self.n:
            self.t[i] += v
            i += i & (-i)

    def prefix(self, i):
        i += 1
        s = 0
        while i > 0:
            s += self.t[i]
            i -= i & (-i)
        return s


class KernelStats:
    __slots__ = ("name", "launches", "footprints", "reuse_hist", "spatial",
                 "n_accesses", "skipped", "last_fp", "overlaps")

    def __init__(self, name):
        self.name = name
        self.launches = 0
        self.footprints = []          # unique sectors per launch
        self.reuse_hist = Counter()   # log2-bucketed stack distance (sectors)
        self.spatial = Counter()      # unique sectors per warp access (1..32)
        self.n_accesses = 0
        self.skipped = 0
        self.last_fp = None           # previous launch's footprint set (overlap)
        self.overlaps = []            # Jaccard vs previous launch


class LaunchState:
    def __init__(self, max_accesses):
        self.max_accesses = max_accesses
        self.n = 0
        self.last_seen = {}           # sector -> access index
        self.fen = Fenwick(max_accesses + 1)
        self.sectors = set()
        self.overflow = False

    def access(self, sector, stats: KernelStats):
        self.sectors.add(sector)
        if self.n >= self.max_accesses:
            self.overflow = True
            stats.skipped += 1
            return
        prev = self.last_seen.get(sector)
        if prev is None:
            stats.reuse_hist[-1] += 1              # cold / compulsory
        else:
            dist = self.fen.prefix(self.n) - self.fen.prefix(prev)
            stats.reuse_hist[max(0, dist).bit_length()] += 1
            self.fen.add(prev, -1)
        self.fen.add(self.n, 1)
        self.last_seen[sector] = self.n
        self.n += 1
        stats.n_accesses += 1


def open_trace(path):
    if path.endswith(".zst"):
        proc = subprocess.Popen(["zstdcat", path], stdout=subprocess.PIPE, text=True,
                                errors="replace")
        return proc.stdout
    return open(path, errors="replace")


def analyze(path, kernel_rx=None, max_launches=None, max_accesses=5_000_000):
    launch_names: dict[int, str] = {}
    kernels: dict[str, KernelStats] = {}
    state: dict[int, LaunchState] = {}
    launches_seen: dict[str, int] = {}
    skipped_gids: set[int] = set()

    for line in open_trace(path):
        m = _LAUNCH.search(line)
        if m:
            name = common.base_kernel_name(m.group(1))
            launch_names[int(m.group(2))] = name
            continue
        m = _ACCESS.search(line)
        if not m:
            continue
        gid = int(m.group(1))
        if gid in skipped_gids:
            continue
        name = launch_names.get(gid, f"launch{gid}")
        if kernel_rx and not kernel_rx.search(name):
            continue
        if gid not in state:
            if max_launches and launches_seen.get(name, 0) >= max_launches:
                skipped_gids.add(gid)
                continue
            launches_seen[name] = launches_seen.get(name, 0) + 1
            # close out any previous launch of this kernel
            for g, st in list(state.items()):
                if launch_names.get(g) == name:
                    _finish_launch(kernels[name], st)
                    del state[g]
            state[gid] = LaunchState(max_accesses)
            ks = kernels.setdefault(name, KernelStats(name))
            ks.launches += 1
        st = state[gid]
        ks = kernels[name]
        addrs = [int(a, 16) for a in m.group(3).split() if a != "0x0"]
        if not addrs:
            continue
        sectors = [a // SECTOR for a in addrs]
        ks.spatial[len(set(sectors))] += 1
        for s in sectors:
            st.access(s, ks)

    for g, st in state.items():
        name = launch_names.get(g, f"launch{g}")
        if name in kernels:
            _finish_launch(kernels[name], st)
    return kernels


def _finish_launch(ks: KernelStats, st: LaunchState):
    ks.footprints.append(len(st.sectors))
    if ks.last_fp is not None:
        union = len(ks.last_fp | st.sectors) or 1
        ks.overlaps.append(len(ks.last_fp & st.sectors) / union)
    ks.last_fp = st.sectors


def hit_cdf(ks: KernelStats):
    """Predicted hit fraction at each CAPACITIES size from the reuse histogram."""
    numeric = {k: v for k, v in ks.reuse_hist.items() if isinstance(k, int) and k >= 0}
    cold = ks.reuse_hist.get(-1, 0)
    total = sum(numeric.values()) + cold
    if not total:
        return {}
    out = {}
    for cap in CAPACITIES:
        cap_sectors_log = (cap // SECTOR).bit_length()
        hits = sum(v for k, v in numeric.items() if k <= cap_sectors_log)
        out[cap] = hits / total
    return out


def emit(kernels, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    cdf_rows = []
    for ks in sorted(kernels.values(), key=lambda k: -max(k.footprints or [0])):
        fp = max(ks.footprints or [0]) * SECTOR
        spatial_total = sum(ks.spatial.values()) or 1
        mean_sect = sum(k * v for k, v in ks.spatial.items()) / spatial_total
        coalesced = sum(v for k, v in ks.spatial.items() if k <= 4) / spatial_total
        overlaps = getattr(ks, "overlaps", [])
        rows.append([ks.name, ks.launches, ks.n_accesses, ks.skipped,
                     f"{fp/1e6:.3f}", round(mean_sect, 2),
                     round(100 * coalesced, 1),
                     round(sum(overlaps) / len(overlaps), 3) if overlaps else ""])
        cdf = hit_cdf(ks)
        cdf_rows.append([ks.name] + [round(cdf.get(c, float("nan")), 4) for c in CAPACITIES])
    p1 = os.path.join(out_dir, "locality.csv")
    common.write_csv(p1, ["kernel", "launches", "accesses", "skipped_overflow",
                          "max_footprint_mb", "mean_sectors_per_warp_access",
                          "pct_warp_accesses_coalesced_le4", "interlaunch_jaccard"],
                     rows)
    p2 = os.path.join(out_dir, "reuse_cdf.csv")
    common.write_csv(p2, ["kernel"] + [f"hit_at_{c//1024}KiB" for c in CAPACITIES],
                     cdf_rows)
    return [p1, p2]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--kernel", default=None, help="regex filter on kernel name")
    ap.add_argument("--max-launches", type=int, default=None,
                    help="analyze at most N launches per kernel")
    ap.add_argument("--max-accesses", type=int, default=5_000_000,
                    help="reuse-distance cap per launch (footprint stays exact)")
    ap.add_argument("--out", default=".")
    args = ap.parse_args(argv)
    rx = re.compile(args.kernel) if args.kernel else None
    kernels = analyze(args.trace, rx, args.max_launches, args.max_accesses)
    for p in emit(kernels, args.out):
        print(f"[✓] {p}")
    for ks in kernels.values():
        cdf = hit_cdf(ks)
        pts = "  ".join(f"{c//1024//1024 if c>=1<<20 else c//1024}"
                        f"{'M' if c>=1<<20 else 'K'}:{100*v:.0f}%"
                        for c, v in cdf.items())
        print(f"{ks.name}: footprint {max(ks.footprints or [0])*SECTOR/1e6:.2f} MB "
              f"× {ks.launches} launches — hit-vs-capacity {pts}")


if __name__ == "__main__":
    main()
