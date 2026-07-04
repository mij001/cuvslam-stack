# Slice-3 Locality — measured reuse distance from NVBit traces

*First architecture-independent locality analysis of cuVSLAM, from real
per-warp memory-address traces (NVBit 1.8 mem_trace + the launch-window /
kernel-name-filter patch) on the RTX 2000 Ada. This replaces the counter-based
LFMR *proxy* of the earlier reports with DAMOV Step-2 measurement: exact
working-set footprint, reuse-distance-vs-cache-capacity CDF, and — new — a
divergence-vs-coalescing split. Data: `data/`.*

Provenance: driver 575.64.05 / CUDA 12.9 / linux-lts 6.12.39 (the downgrade
that unblocked NVBit); cuVSLAM 15.0 cu12 wheel rebuilt from source; clocks
locked 1620/7001. Reuse CDF within a launch is over the first 4M sector-access
events (`--max-accesses`; footprints and divergence are exact over the whole
launch).

## 1. Method note — the metrics

- **Footprint** = unique 32 B sectors touched × 32 B. Exact per launch. This is
  the *direct* measurement that the counter reports could only infer from
  traffic.
- **Reuse-distance CDF**: for each sector access, the number of distinct
  sectors touched since that sector was last seen (LRU stack distance). The CDF
  evaluated at capacities {64 KiB … 48 MiB} is the **predicted hit rate vs
  cache size** — architecture-independent (no real cache involved). A *flat*
  curve means adding cache does nothing: the misses are compulsory (cold).
- **Divergence vs coalescing**: `mean_active_lanes` (32 = fully converged warp)
  and `sectors_per_32_active_lanes` (coalescing *among the active lanes* — 1 is
  perfect). This split is the fix for a real trap: a warp with few active lanes
  looks "coalesced" on a naive sectors/access metric but is actually divergent.

## 2. Front-end (streaming) — measured cache-immune, as hypothesized

The preprocess/feature kernels' reuse CDF is **flat across the entire capacity
range** (64 KiB → 48 MiB): their hit rate is fixed by tiny-distance intra-launch
reuse, and **no amount of cache changes it**. The residual misses are
compulsory cold-start misses on streaming image data — a cache cannot remove
them, but a near-sensor "consume before DRAM" substrate eliminates them
outright. This is the streaming→SRAM case, now proven from the address stream
rather than inferred from an SoL/LFMR proxy.

| kernel | footprint (MB) | hit @64 KiB | hit @48 MiB | active lanes | sectors/32 lanes |
|---|---|---|---|---|---|
| cast_image_kernel_rgb | 2.15 | 0.51 | 0.56 | 32.0 | 4.0 |
| cast_depth_kernel | 1.84 | 0.91 | 0.91 | 32.0 | 3.0 |
| gaussian_scaling_kernel | 0.31 | 0.88 | 0.88 | 32.0 | 4.0 |
| conv_grad_x_kernel | 1.23 | 0.99 | 0.99 | 32.0 | 5.3 |
| conv_grad_y_kernel | 1.23 | 0.98 | 0.98 | 32.0 | **15.4** |
| lk_track_kernel | 0.02 | 0.99 | 0.99 | 32.0 | 2.1 |
| matcher::photometric | 0.006 | — | — | 32.0 | 3.5 |

Two clean sub-findings: (a) every front-end kernel is **fully converged (32.0
active lanes)** — so their low sectors/access is *true coalescing*, not
divergence; (b) **[superseded — see §5]** `conv_grad_y` is the one scattered
kernel (15.4 sectors per 32
lanes — the vertical-gradient stride crosses rows), a concrete, localized
data-layout target that the counter view (kernel-level "feature detect is
memory-bound") could not isolate.

## 3. Loop-closure scan (st_track_with_cache) — the trace overturns the proxy
**[superseded — the tables below blend memory spaces; §5 re-derives them
space-filtered and REVERSES the coalescing verdict. Footprint and migration
rows stand.]**

| metric | TUM (room) | KITTI 00 (street) |
|---|---|---|
| per-scan footprint | 0.467 MB | 1.096 MB |
| mean active lanes | **32.0** | **32.0** |
| sectors / 32 active lanes | **2.14** | **2.13** |
| % accesses ≤4 sectors | 99.4 % | 99.5 % |
| reuse hit @64 KiB | 99.9 % | 99.8 % |
| inter-launch Jaccard | 0.672 | 0.899 |

**The address trace contradicts the counter-based classification, and the
trace is ground truth.** The earlier reports read st_track as a *scattered
gather* (ncu `sectors/request` 18–30) and filed it G2-coalescing / ISP-scatter.
The actual per-warp addresses say the opposite: **fully converged (32.0 active
lanes — zero divergence) and tightly coalesced (2.1 sectors per warp access,
99.4 % of accesses touching ≤4 sectors).** The kernel is a *streaming scan with
strong local reuse*, not a scatter: footprint 0.47–1.10 MB, ~3300× reuse per
sector, and 99.9 % of reuse distances under 64 KiB, so each scan is
**L2-resident** on the Ada. (The ncu `sectors/request` almost certainly counts
an L1-sector-replay / lookup effect, not thread-address spread — a proxy
artifact this measurement exposes; whichever, the addresses are what a memory
system sees.)

**So the ISP/near-memory case is real but re-grounded — and *stronger* for a
streaming substrate.** It does not rest on within-scan scatter (there is none);
it rests on **session scale**: the per-scan working set grows with the map
(0.47 → 1.10 MB, room → street) and *migrates* between consecutive scans
(Jaccard 0.67 → 0.90, i.e. 10–33 % of the set turns over each scan). The union
over a long deployment — the whole keyframe database, incrementally scanned —
is what no cache holds, and because the access is coalesced and reuse-heavy, a
**near-memory *streaming* engine** (not a gather engine) is the right substrate.
KITTI's higher Jaccard (0.90) means the outdoor drive's per-scan set is more
stable frame-to-frame; the database still grows without bound with distance
travelled.

This is exactly the correction a characterization paper exists to make: a
measured address stream overturning a counter proxy, and sharpening the
hardware ask from "scatter engine" to "streaming near-memory scan over a
capacity-unbounded, slowly-migrating database."

## 4. What this closes / changes

- Replaces the LFMR/sectors *proxy* with measured reuse distance + divergence
  (PUBLISHABILITY issue 4 → resolved).
- **Overturns the G2-scatter label on the loop-closure kernel** — a
  trace-vs-counter correction; the classifier's counter-based verdict is now
  annotated with the trace ground truth.
- Adds a divergence/coalescing axis (no CPU analog) the DAMOV-GPU adaptation
  calls for; both st_track and the front-end are converged, so cuVSLAM's
  memory cost is coalesced streaming + capacity, not divergence.
- Localizes a real data-layout target the kernel view missed: `conv_grad_y`
  (15.4 sectors/32 lanes, vertical-gradient stride).

## 5. 2026-07-05 CORRECTION — space-filtered re-derivation (data_v2/)

The TaggedAllocator attribution join (reports/2026-07-04_attribution/) exposed
that mem_trace records **every memory space**, and this report's tables blended
them: shared-memory tiles (LDS/STS, on-chip) and the per-thread register-spill
window (LDL/STL, DRAM-backed compiler scratch) were counted as if they were
data accesses. `analysis/locality.py --spaces` now filters; the same traces
re-derived global-only (`data_v2/*_global/`) and, for the spill stream,
local-only (`data_v2/*_local/`).

**§3 reverses.** st_track_with_cache's *global* (data) accesses are a
**scattered gather**, exactly as the ncu counters said all along:

| metric (global-only) | TUM (room) | KITTI 00 (street) | old §3 (mixed) |
|---|---|---|---|
| sectors / warp access | **23.4** | **30.0** | 2.1 |
| % accesses ≤4 sectors | **6.0 %** | **2.3 %** | 99.4 % |
| per-scan footprint | 0.464 MB | 1.093 MB | ≈unchanged |
| inter-launch Jaccard | 0.669 | 0.899 | unchanged |

The 99%-coalesced signal was the **spill stream** (local-only: 2.0
sectors/warp, 100 % ≤4, Jaccard 1.0 — per-thread interleaving is coalesced by
construction), which carries ~94 % of the kernel's accesses (attribution join)
and drowned the data accesses in the blended table. **The counter proxy was
never wrong** — ncu's `sectors/request` (18–30) matches the corrected
global-only measurement (23–30). Two independent methods now agree; the
2026-07-04 "proxy overturned" claim is withdrawn and the classifier's original
**G2-scatter label on st_track stands**.

The corrected picture of the loop-closure scan: **a scattered gather over the
keyframe-descriptor buffer, superimposed on a coalesced spill stream that
dominates its DRAM volume.** The ISP case keeps both legs — session-scale
database growth/migration (footprint and Jaccard rows unchanged) *plus* genuine
within-scan scatter — and gains a third: most of the kernel's DRAM bandwidth is
compiler spill, i.e. a larger register file / spill-local SRAM attacks the
volume while a near-memory gather engine attacks the latency.

**§2 sharpens.** Front-end cast/gaussian/lk rows are identical (they had no
on-chip traffic) — the flat-CDF streaming→SRAM case stands. `conv_grad_x/y`'s
apparent scatter (5.3/15.4 sectors) and conv_grad_x's 92 % "reuse" were
**shared-memory tiles**: global-only both are 4.0 sectors/warp, 100 % ≤4,
reuse CDF flat at ~0 — *pure* coalesced streaming. The "conv_grad_y
data-layout target" sub-finding is withdrawn; the front-end is uniformly
stream-clean and the correct optimization target is on-chip, not layout.

Provenance: same traces (`~/slice3/*.zst`), `locality.py --spaces global`
(and `--spaces local` for the spill view), 2026-07-05.
