#!/usr/bin/env python3
"""attribution.py — address→data-structure attribution (TaggedAllocator milestone).

Joins three artifacts captured from ONE instrumented cuVSLAM process
(patches/0002-tagged-allocator-nvtx.patch + blocked/mem_trace_alloc_events.patch):

  1. the Layer-1 journal (CUVSLAM_ALLOC_LOG): every wrapper-level device
     allocation with a host backtrace — WHO owns the buffer;
  2. the NVBit sidecar (MEM_TRACE_ALLOC_LOG): every driver-level allocation
     with the grid-launch id it precedes — WHEN the buffer is live, in the
     trace's own launch-id clock;
  3. the mem_trace address stream — WHAT each kernel actually touches.

Two subcommands:

  resolve  Turn the raw journal into a symbolized allocation table. Needs
           binutils addr2line and the exact .so files the process ran with, so
           it runs on the capture host. Backtrace PCs are rebased with the
           journal's embedded /proc/self/maps rows; the owner is the innermost
           frame that is neither the wrapper itself nor allocator/std plumbing.
             python3 -m analysis.attribution resolve journal.csv --out alloc_table.csv

  join     Stream the trace, keep the live allocation set in launch-id order
           (sidecar events applied between launches; the live set is
           non-overlapping at any instant, so containment is a bisect), and
           aggregate per (kernel × data-structure tag) sector traffic.
           Stdlib-only, GPU-free, runs anywhere.
             python3 -m analysis.attribution join trace.zst alloc_table.csv sidecar.csv --out DIR

Known blind spot (stated in the report): GPUImage is read through texture
objects; TEX-path fetches do not appear in mem_trace (global LD/ST only), so
image-tag traffic is a lower bound there.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from bisect import bisect_right
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common  # noqa: E402
from analysis.locality import open_trace, _LAUNCH, _ACCESS  # noqa: E402

SECTOR = 32

# owner frames that are allocator plumbing, not data structures — walk past them
_PLUMBING = re.compile(
    r"cuvslam::cuda::(GPUArray|GPUArrayPinned|GPUOnlyArray|GPUImage|GPUPatchData)\b"
    r"|cuvslam::cuda::LogAlloc|^std::|^__gnu|^Eigen::|^operator new"
    r"|_Sp_counted|allocator|make_unique|make_shared|uniq_ptr")

# owner call site → taxonomy tag, first match wins (vocabulary follows
# onboarding §11.2.1, trimmed to the structures cuVSLAM actually allocates;
# unmatched owners fall through to their class name so nothing is hidden)
TAG_RULES = [
    (r"GPULinearSystem|SchurComplement|GPUBundleAdjustmentProblem|sba\.cpp",
     "ba_linear_system"),
    (r"STDescriptorGpuOps|st_descriptor", "keyframe_descriptors"),
    (r"build_gpu_image_pyramid|ImagePyramid|GaussianGPUImagePyramid|prepare_levels",
     "pyramid_levels"),
    (r"build_gpu_depth_pyramid|DepthPyramid", "depth_pyramid"),
    (r"GradientPyramid|gradient", "gradient_pyramid"),
    (r"ImageContext|cast_image|image_cast", "images_raw"),
    (r"GPUICPTools|icp", "icp_state"),
    (r"GPUSelection|computeGFTT|gftt|selection", "feature_selection_scratch"),
    (r"PatchData|points_cache|st_tracker", "st_patch_cache"),
    (r"MonoSOFGPU|MultiSOF|sof_", "feature_tracks"),
    (r"Matcher|matcher", "matcher_state"),
]


# ── resolve ──────────────────────────────────────────────────────────────────

def parse_journal(path):
    """maps rows, alloc records, free records from a CUVSLAM_ALLOC_LOG."""
    maps, allocs, frees = [], [], []
    rx_map = re.compile(r"M,([0-9a-f]+)-([0-9a-f]+) r-xp ([0-9a-f]+) \S+ \S+ +(/\S+)")
    for line in open(path, errors="replace"):
        if line.startswith("M,"):
            m = rx_map.match(line.strip())
            if m:
                row = (int(m.group(1), 16), int(m.group(2), 16),
                       int(m.group(3), 16), m.group(4))
                if row not in maps:
                    maps.append(row)
        elif line.startswith("A,"):
            f = line.strip().split(",")
            allocs.append({"t_us": int(f[1]), "ptr": int(f[2], 16),
                           "bytes": int(f[3]), "kind": f[4],
                           "bt": [int(x, 16) for x in f[5:]]})
        elif line.startswith("F,"):
            f = line.strip().split(",")
            frees.append({"t_us": int(f[1]), "ptr": int(f[2], 16)})
    return maps, allocs, frees


def _rebase(pc, maps):
    for start, end, off, mod in maps:
        if start <= pc < end:
            return mod, pc - start + off
    return None, None


def symbolize(maps, allocs):
    """Batch-addr2line every unique PC; returns {pc: (func, site)}."""
    by_mod = defaultdict(set)
    where = {}
    for a in allocs:
        for pc in a["bt"]:
            mod, off = _rebase(pc, maps)
            if mod:
                by_mod[mod].add((pc, off))
                where[pc] = (mod, off)
    resolved = {}
    for mod, pcs in by_mod.items():
        pcs = sorted(pcs)
        try:
            out = subprocess.run(
                ["addr2line", "-e", mod, "-f", "-C", "-i", "-a"]
                + [hex(off) for _, off in pcs],
                capture_output=True, text=True, check=True).stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        groups = re.split(r"^0x[0-9a-f]+:?\s*$", out, flags=re.M)[1:]
        for (pc, _off), grp in zip(pcs, groups):
            lines = [ln for ln in grp.strip().splitlines() if ln]
            # -f pairs: func, file:line, [inlined-by pairs …]; innermost first
            frames = [(lines[i], lines[i + 1]) for i in range(0, len(lines) - 1, 2)]
            resolved[pc] = frames
    return resolved


def owner_of(bt, resolved):
    """Innermost non-plumbing cuvslam frame (function, file:line)."""
    fallback = None
    for pc in bt:
        for func, site in resolved.get(pc, []):
            if func == "??" or _PLUMBING.search(func):
                continue
            if "cuvslam" in func or "/cuvslam" in site:
                return func, site
            if fallback is None:
                fallback = (func, site)
    return fallback or ("unknown", "??:0")


def tag_of(func, site):
    hay = f"{func} {site}"
    for rx, tag in TAG_RULES:
        if re.search(rx, hay):
            return tag
    m = re.search(r"cuvslam::[\w:]*?(\w+)::(?:\1|Impl)?\(", func)
    return m.group(1) if m else "untagged"


def cmd_resolve(args):
    maps, allocs, frees = parse_journal(args.journal)
    if not allocs:
        sys.exit(f"no A records in {args.journal}")
    resolved = symbolize(maps, allocs)
    rows = []
    for a in allocs:
        func, site = owner_of(a["bt"], resolved)
        tag = tag_of(func, site)
        if a["kind"] in ("GPUArrayPinnedHost", "HostAllocator"):
            tag += ":host"           # zero-copy pinned host: PCIe-path traffic
        rows.append([a["t_us"], hex(a["ptr"]), a["bytes"], a["kind"], tag,
                     func.split("(")[0], site])
    for f in frees:
        rows.append([f["t_us"], hex(f["ptr"]), "", "FREE", "", "", ""])
    rows.sort(key=lambda r: r[0])
    common.write_csv(args.out, ["t_us", "ptr", "bytes", "kind", "tag",
                                "owner_func", "owner_site"], rows)
    tags = Counter(r[4] for r in rows if r[3] != "FREE")
    print(f"[✓] {args.out}  ({len(allocs)} allocs, {len(frees)} frees; "
          f"tags: {dict(tags.most_common())})")


# ── join ─────────────────────────────────────────────────────────────────────

def load_alloc_table(path):
    """Ordered alloc/free event list from a resolved table."""
    events = []
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        idx = {k: i for i, k in enumerate(header)}
        for line in fh:
            f = line.rstrip("\n").split(",")
            if f[idx["kind"]] == "FREE":
                events.append(("F", int(f[idx["ptr"]], 16), 0, ""))
            else:
                events.append(("A", int(f[idx["ptr"]], 16),
                               int(f[idx["bytes"]]), f[idx["tag"]]))
    return events


def load_sidecar(path):
    """Ordered driver alloc/free events with launch ids."""
    events = []
    for line in open(path):
        if line.startswith("ALLOC,"):
            _, gid, ptr, size = line.strip().split(",")
            events.append(("A", int(gid), int(ptr, 16), int(size)))
        elif line.startswith("FREE,"):
            _, gid, ptr = line.strip().split(",")
            events.append(("F", int(gid), int(ptr, 16), 0))
    return events


def correlate(table_events, sidecar_events):
    """Tag sidecar allocations from the Layer-1 table.

    Both logs observe the same process in the same order; Layer-1 records the
    runtime pointer, the sidecar the driver pointer — identical for these APIs.
    Match each sidecar ALLOC to the earliest unconsumed table ALLOC with the
    same (ptr, bytes); driver-internal allocations stay untagged_driver.
    """
    pending = defaultdict(list)          # (ptr, bytes) -> [tag, …] in order
    for kind, ptr, size, tag in table_events:
        if kind == "A":
            pending[(ptr, size)].append(tag)
    tagged, n_matched = [], 0
    for kind, gid, ptr, size in sidecar_events:
        if kind == "A":
            q = pending.get((ptr, size))
            if q:
                tag = q.pop(0)
                n_matched += 1
            else:
                tag = "untagged_driver"
            tagged.append(("A", gid, ptr, size, tag))
        else:
            tagged.append(("F", gid, ptr, 0, ""))
    unmatched = sum(len(v) for v in pending.values())
    return tagged, n_matched, unmatched


class LiveSet:
    """Non-overlapping live allocations; bisect containment lookups."""

    def __init__(self):
        self.bases = []
        self.entries = []                # parallel: (base, end, tag)

    def alloc(self, base, size, tag):
        i = bisect_right(self.bases, base)
        self.bases.insert(i, base)
        self.entries.insert(i, (base, base + size, tag))

    def free(self, base):
        i = bisect_right(self.bases, base) - 1
        if i >= 0 and self.entries[i][0] == base:
            del self.bases[i]
            del self.entries[i]

    def find(self, addr):
        i = bisect_right(self.bases, addr) - 1
        if i >= 0:
            base, end, tag = self.entries[i]
            if base <= addr < end:
                return tag
        return None


def cmd_join(args):
    table_events = load_alloc_table(args.alloc_table)
    sidecar_events = load_sidecar(args.sidecar)
    tagged, n_matched, unmatched = correlate(table_events, sidecar_events)

    live = LiveSet()
    ev_i = 0

    def apply_events(up_to_gid):
        nonlocal ev_i
        while ev_i < len(tagged) and tagged[ev_i][1] <= up_to_gid:
            kind, _gid, ptr, size, tag = tagged[ev_i]
            if kind == "A":
                live.alloc(ptr, size, tag)
            else:
                live.free(ptr)
            ev_i += 1

    launch_names = {}
    per_kt = defaultdict(Counter)        # kernel -> tag -> warp accesses
    sectors_kt = defaultdict(Counter)    # kernel -> tag -> sectors
    kernel_acc = Counter()               # kernel -> warp accesses (for the cap)
    capped = set()                       # kernels that reached --max-accesses-per-kernel
    cap = args.max_accesses_per_kernel   # tag fractions converge fast; a cap keeps
                                         # the join bounded on TB-scale st_ scans and
                                         # lets it early-stop once all kernels saturate
    cur_gid = -1
    for line in open_trace(args.trace):
        # hot path: fixed " - "-separated access records (billions of lines);
        # regexes are kept for LAUNCH lines and as the format oracle in tests
        parts = line.split(" - ")
        if len(parts) >= 6 and parts[1].startswith("grid_launch_id "):
            gid = int(parts[1][15:])
            opcode = parts[4]
            addr_field = parts[5]
        else:
            m = _LAUNCH.search(line)
            if m:
                launch_names[int(m.group(2))] = common.base_kernel_name(m.group(1))
                continue
            m = _ACCESS.search(line)
            if not m:
                continue
            gid = int(m.group(1))
            opcode = m.group(2)
            addr_field = m.group(3)
        if gid != cur_gid:
            cur_gid = gid
            apply_events(gid)
        kernel = launch_names.get(gid, f"launch{gid}")
        if cap is not None and kernel in capped:
            # this kernel is saturated; once every kernel we've seen is capped,
            # the remaining trace can only repeat known kernels — stop reading.
            if len(capped) == len(kernel_acc):
                break
            continue
        addrs = [int(a, 16) for a in addr_field.split() if a != "0x0"]
        if not addrs:
            continue
        # mem_trace records every memory space; only global-space ops can hit
        # a data structure. Shared is on-chip (no DRAM); local is the per-
        # thread spill window (DRAM-backed, but compiler scratch, not data).
        if opcode.startswith(("LDS", "STS", "ATOMS")):
            tag = "shared_onchip"
        elif opcode.startswith(("LDL", "STL")):
            tag = "local_spill"
        else:
            tag = live.find(addrs[0]) or "unmapped"
        per_kt[kernel][tag] += 1
        sectors_kt[kernel][tag] += len(set(a // SECTOR for a in addrs))
        if cap is not None:
            kernel_acc[kernel] += 1
            if kernel_acc[kernel] >= cap:
                capped.add(kernel)
                if len(capped) == len(kernel_acc):
                    break

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for kernel in sorted(sectors_kt, key=lambda k: -sum(sectors_kt[k].values())):
        total = sum(sectors_kt[kernel].values()) or 1
        for tag, sect in sectors_kt[kernel].most_common():
            rows.append([kernel, tag, per_kt[kernel][tag], sect,
                         sect * SECTOR, round(100 * sect / total, 1)])
    path = os.path.join(args.out, "attribution.csv")
    common.write_csv(path, ["kernel", "tag", "warp_accesses", "sectors",
                            "bytes", "pct_kernel_traffic"], rows)
    print(f"[✓] {path}")
    if capped:
        print(f"    (early-stop: {len(capped)} kernel(s) hit the "
              f"{cap:,}-access cap — fractions are from the capped prefix)")
    print(f"    sidecar allocs tagged {n_matched}/{n_matched + sum(1 for e in tagged if e[0] == 'A' and e[4] == 'untagged_driver')}"
          f" (journal leftovers: {unmatched})")
    for kernel in list(sorted(sectors_kt, key=lambda k: -sum(sectors_kt[k].values())))[:12]:
        total = sum(sectors_kt[kernel].values()) or 1
        top = ", ".join(f"{t}:{100 * s / total:.0f}%"
                        for t, s in sectors_kt[kernel].most_common(3))
        unmapped = 100 * sectors_kt[kernel].get("unmapped", 0) / total
        print(f"    {kernel}: {top}" + (f"  [unmapped {unmapped:.1f}%]" if unmapped >= 0.05 else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("resolve", help="symbolize a CUVSLAM_ALLOC_LOG journal")
    r.add_argument("journal")
    r.add_argument("--out", default="alloc_table.csv")
    r.set_defaults(fn=cmd_resolve)
    j = sub.add_parser("join", help="attribute trace traffic to data structures")
    j.add_argument("trace")
    j.add_argument("alloc_table")
    j.add_argument("sidecar")
    j.add_argument("--out", default=".")
    j.add_argument("--max-accesses-per-kernel", type=int, default=None,
                   help="stop accumulating a kernel past N warp accesses and "
                        "early-stop the read once all kernels are capped "
                        "(bounds the join on TB-scale traces; fractions converge)")
    j.set_defaults(fn=cmd_join)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
