# Data-structure attribution — TaggedAllocator + NVTX (TUM fr3 long_office, SLAM)

**2026-07-04, RTX 2000 Ada (dell-workstation), driver 575.64.05 / CUDA 12.9.1.**
The onboarding §11.2 three-layer pipeline, built and run end-to-end. Kernel-level
claims ("st_track is memory-bound streaming") become data-structure-level ones
("the loop-closure scan reads the *st_track patch cache*") by observing which
allocation every traced access falls into. Closes PUBLISHABILITY issues **3**
(address→data-structure mapping) and **8** (NVTX stage attribution).

## Method (what was built)

| Layer | Mechanism | Artifact |
|---|---|---|
| 1 — source | `patches/0002-tagged-allocator-nvtx.patch`: every `cudaMalloc*`/`cudaFree*` in the RAII wrappers (`GPUArray`, `GPUArrayPinned`, `GPUOnlyArray`, `GPUImage`) + 3 direct sites (`GPUPatchData`, `SelectionSortTemp`, `SBARig`) journals ptr/size/kind + a host backtrace (`CUVSLAM_ALLOC_LOG`); wheel rebuilt RelWithDebInfo (keeps DWARF; Release links `-s`) | `alloc_table_full_run.csv` |
| 2 — driver | `blocked/mem_trace_alloc_events.patch`: NVBit mem_trace also logs `cuMemAlloc*`/`cuMemFree*` with the grid-launch id it precedes (`MEM_TRACE_ALLOC_LOG`) — allocation lifetimes in the trace's own clock, plus allocations outside layer 1 (CUB, cuSOLVER) | `driver_allocs_full_run.csv` |
| 3 — join | `analysis/attribution.py`: `resolve` symbolizes backtraces (addr2line vs the journal's `/proc/self/maps`; the owner is the innermost non-plumbing frame) and maps owners to a fixed tag vocabulary; `join` streams the trace with the live allocation set in launch-id order | `join_*/attribution.csv` |
| NVTX | `profiler_enable.h` typedefs flipped to `Enable` under `-DUSE_NVTX=ON` — cuVSLAM's own profiler domains emit ranges; nsys `nvtx_kern_sum` gives the measured kernel→stage table | `nvtx_kern_sum.csv` |

Captures (`campaign/ws_attribution_capture.sh`; the full-matrix version is
`campaign/ws_attribution_all.sh` + `ws_attribution_gapfill.sh`): pass 1 = full
sequence, NVBit injected, empty window (launch map + journals, near-native);
pass 2a = all-kernel steady-state window (~3 frames mid-orbit, 28.1 GiB of
addresses); pass 2b = `KERNEL_FILTER=st_` in a late loop-closure-likely window
(a whole-sequence st_ trace is TB-scale — the scan re-reads the keyframe DB
every keyframe); pass 2c = planned gap-fill windows covering any kernel the
sliced windows missed (`plan_gapfill.py` diffs the pass1 launch map against
the joins); 600-frame nsys NVTX capture. Layer-1 sanity: **274/274 allocations resolved and tagged,
0 unknown** (240 device + 34 pinned-host) — identical structure across all
runs (deterministic). Join sanity: driver sidecar 274/290 matched to the
journal; the 16 leftovers are CUB/cuSOLVER internals (`untagged_driver`).

**Memory-space discipline (the join's key correctness rule):** mem_trace
records *every* memory space. Shared (`LDS/STS`) is on-chip — no DRAM claim
attaches to it; local (`LDL/STL`) is the per-thread spill window — DRAM-backed
but compiler scratch, not a data structure. Only global-space accesses
participate in data-structure attribution. Without this split, tile/spill
traffic masquerades as 88–98% "unmapped" per kernel; with it, unmapped falls
to ≤7% (mostly ≤1%).

## Finding 1 — the GPU memory budget is static (nothing grows with the map)

240 allocations for the full ~2500-frame loop-closing sequence — the **same
240** as a 30-frame run; peak live == total allocated for every tag (no
mid-run frees). cuVSLAM pre-sizes everything at tracker init:

| tag | allocs | peak live MB |
|---|---|---|
| images_raw | 92 | 39.97 |
| pyramid_levels | 99 | 31.15 |
| ba_linear_system | 31 | 24.26 |
| keyframe_descriptors | 2 | 6.73 |
| feature_tracks | 8 | 4.27 |
| depth_pyramid | 3 | 2.21 |
| icp_state | 4 | 0.07 |
| feature_selection_scratch | 1 | ~0 |
| **total** | **240** | **108.65** |

Plus 34 pinned-host buffers (17.14 MB; dominated by `ba_linear_system:host`
at 15.4 MB) — the `GPUArrayPinned` host halves and `HostAllocator` vectors,
journaled separately because pinned memory is device-visible under UVA
(`:host` marks would-be zero-copy PCIe traffic in the join).

**Implication for the taxonomy:** the GPU-resident keyframe state is a fixed
6.7 MB descriptor buffer; the session-scale database (landmarks/keyframes,
LMDB-backed) grows **host-side only**. This independently confirms the Slice-3
re-grounding: the ISP case is the *host* database path, not GPU-side gather —
now shown from the allocator, not inferred from counters.

## Finding 2 — measured kernel→stage map (NVTX, closes issue 8)

Innermost NVTX range per kernel (600-frame capture, 46 keyframe events;
`nvtx_kern_sum.csv`):

| NVTX stage | kernels |
|---|---|
| MultiSOF:mono start (per-frame front-end) | cast_image_kernel_rgb, gaussian_scaling, conv_grad_x/y, lk_track |
| MultiSOF:mono finish (keyframe feature refresh) | gftt_values, accumulateGFTT, downsample_gftt_x8, non_max_suppression, select_features, filter_maximums (+ CUB keypoint sort) |
| RGBD VO:track (per-frame solver) | matcher::photometric/point_to_point/lift, cast_depth, sba::reduced_system_* + cuSOLVER getrf/trsv |
| SBA GPU:SBA iter (windowed BA) | sba::build_full_system_*, calc_jacobians, evaluate_cost, clear_full_system_* |
| SLAM:process vo data (keyframe ingest) | **st_build_cache_kernel** |
| SLAM:LC & optimization (loop closure) | **st_track_with_cache_kernel** |

The former regex-based stage mapping is now measured: st_track_with_cache is
*the* loop-closure kernel, st_build_cache is keyframe ingestion.

## Finding 3 — per-kernel data-structure traffic (steady-state window join)

`join_steady_state/attribution.csv` (sector-weighted shares; ~3 frames
mid-orbit, 28.1 GiB of trace addresses). Selected rows:

| kernel | traffic composition |
|---|---|
| **st_track_with_cache** (loop closure) | **local_spill 94.2%**, keyframe_descriptors 4.9%, unmapped 0.8% |
| st_build_cache (keyframe ingest) | keyframe_descriptors 93.0%, unmapped 7.0% |
| sba::reduced_system_stage_2 | **ba_linear_system 96.9%**, shared 3.1% |
| sba::reduced_system_stage_12 | ba_linear_system 56%, shared 44% |
| sba::v1T_x_M_T_x_v2 / calc_point_update | shared 88%, ba_linear_system 12% |
| conv_grad_x / conv_grad_y | shared 92–96%, **pyramid_levels** remainder |
| gftt_values | shared 98%, feature_tracks 2% |
| non_max_suppression | shared 89%, feature_tracks 11% |
| cub::DeviceMergeSort* (keypoint sort) | shared 97%, feature_tracks 3% |
| cast_image_kernel_rgb | pyramid_levels 75%, images_raw 25% |

Three data-structure-level readings:

1. **The loop-closure scan's DRAM traffic is register spill, not database
   gather — and the database gather it does perform is scattered.**
   st_track_with_cache touches the keyframe descriptor buffer for only ~5% of
   its accesses; 94% is the compiler's local-memory window (the 9-dim patch
   working set exceeds the register budget). This finding triggered the
   Slice-3 space-filtered re-derivation (FINDINGS §5): the *global* accesses
   are a scattered gather (23–30 sectors/warp), confirming the original ncu
   counter reading, while the spill stream is coalesced by construction and
   had masked it. The scan is bounded by spill bandwidth; its data side is a
   scattered gather over a fixed 6.7 MB device buffer. The ISP target is the
   host-side LMDB store; the device-side asks are a *larger register file /
   spill-local SRAM* (volume) and near-memory gather (latency).
2. **BA is the clean PiM candidate at the data-structure level.** The
   full-system SBA kernels stream `ba_linear_system` (24.3 MB device +
   15.4 MB pinned-host mirror) at 97% of their global traffic — one named,
   pre-sized, contiguous structure.
3. **Front-end kernels barely touch DRAM directly** — conv/GFTT/sort traffic
   is 89–98% shared-memory tiles; their global residue is exactly the
   pyramid/track structures the streaming (near-sensor) taxonomy class
   predicts.

The full 27-sequence generalization of this table — coverage-audited,
gap-filled to 0 missing kernels, 48/49 kernels unanimous — is
`reports/2026-07-05_attribution_campaign/`; raw traces and per-sequence joins
live on the workstation sda at `/mnt/data/attribution_out/`.

## Limitations (stated for the paper)

- `GPUImage` buffers are also read through **texture objects**; TEX-path
  fetches are invisible to mem_trace (global LD/ST only), so image-tag traffic
  is a lower bound. The pyramid/gradient tags cover the LDG-path reads.
- Driver-internal allocations (CUB sort scratch, cuSOLVER workspaces) carry
  the `untagged_driver` tag — visible, bounded, attributable to their stage
  via the launch map when needed.
- Single sequence (the canonical loop-closure workload); the campaign already
  established cross-sequence kernel-class stability (91% modal consistency).

## Reproduce

```bash
# workstation (capture + join; needs the instrumented venv + NVBit tool):
setsid nohup profiling/campaign/ws_attribution_capture.sh > ~/attribution/capture.log 2>&1 &
# anywhere (from committed CSVs): the join tables in this directory are final;
# tests: python3 profiling/tests/test_analysis.py
```
