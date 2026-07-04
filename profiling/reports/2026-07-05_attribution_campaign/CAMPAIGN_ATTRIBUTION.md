# Full-matrix data-structure attribution — 27 sequences × 4 datasets

**2026-07-05, RTX 2000 Ada (dell-workstation), instrumented cuVSLAM wheel
(`patches/0002`).** Extends the single-workload attribution
(`reports/2026-07-04_attribution/`) to the entire campaign matrix: KITTI 00–10,
EuRoC MH/V1/V2 ×11, TUM fr3 ×4, TUM-VI, all in SLAM mode. Produced by
`campaign/ws_attribution_all.sh` + `ws_attribution_gapfill.sh` (captures) and
`analysis/attribution_campaign.py` (synthesis). Raw traces + per-sequence joins:
`/mnt/data/attribution_out/` on the workstation's sda (~85 GB); the committed
CSVs here reproduce every table.

## Headline

**48 of 49 kernels have a unanimous top data-structure tag across every
sequence they appear in.** Per-kernel data-structure composition is a property
of the KERNEL, not the workload — the attribution analog of the
characterization campaign's 91% modal-class consistency, and the license to
make per-kernel data-structure claims dataset-independently.

The one non-unanimous kernel (`sba::build_full_system_2`, 68% agreement) is not
sequence-noise: it splits ~40% unmapped / ~36% ba_linear_system / ~23% shared
nearly identically in *every* sequence — a systematic residual (see below).

## Coverage (the windowed-slice audit)

Windowed traces risk missing sparse kernels, so coverage was audited per
sequence (pass-1 launch map vs the union of joins) and gap-filled iteratively:
keyframe-refresh GFTT clusters missed by 3-frame steady-state windows (12/27),
three EuRoC windows that landed in front-end-only stretches, and st_ scans
drifting out of planned windows (launch ids shift run-to-run; the planner now
anchors on dense launch clusters, `plan_gapfill.py`). After three gap-fill
iterations: **every cuVSLAM kernel of every sequence has attribution rows —
0 missing.** `coverage.csv` is the kernel × sequence matrix; kernels appearing
in fewer than 27 sequences reflect the workload itself (RGBD-only kernels in 4
TUM sequences, stereo LK in 11–12 KITTI/EuRoC ones, no loop closures in
kitti01/04 — confirmed non-looping trajectories, a useful no-loop bracket).

## The taxonomy, read off the table (consistency_table.md)

- **Streaming front-end** (cast, gaussian, conv_grad, GFTT, LK, CUB sort):
  global traffic lands on `images_raw` / `pyramid_levels` / `feature_tracks`;
  the compute-side traffic is 83–98% shared-memory tiles. Per-frame,
  bounded, near-sensor class — as the taxonomy predicts.
- **BA/solver block** (all sba::* + cuSOLVER helpers): global traffic is
  `ba_linear_system` at 96–100% for the streaming kernels of the group; the
  reduction-style kernels are shared-heavy with the same global tag. One
  named, pre-sized structure (24.3 MB device + 15.4 MB pinned-host mirror) —
  the clean DRAM-PiM target.
- **Keyframe/loop-closure pair**: `st_build_cache` → `keyframe_descriptors`
  (93–100% global) on ingest; `st_track_with_cache` → 91–95% `local_spill` +
  5–8% `keyframe_descriptors` scatter on scan, uniformly across room-scale
  (TUM/EuRoC) and street-scale (KITTI). Combined with the Slice-3 §5
  correction (the global side is a scattered gather; the DB grows host-side):
  the ISP ask is the host LMDB store; the device-side asks are spill-local
  SRAM (volume) and near-memory gather (latency).

## Identified residuals (bounded, documented)

- `sba::build_full_system_2` (~40% unmapped everywhere), `lk_track` /
  `lk_track_horizontal` / `cub::DeviceMergeSortPartition` / `getrf` (unmapped
  or `untagged_driver` global): consistent with **static module memory**
  (`__device__` globals allocated by the module loader — invisible to both the
  wrapper journal and the `cuMemAlloc` sidecar by construction; the .so's
  cubins carry thrust/static symbols) plus texture-path reads (LK reads
  pyramids via texture objects; TEX fetches don't appear in mem_trace at all,
  so LK's visible globals are its small side-buffers). Pinning these to named
  statics needs the Layer-3 kernel-arg correlation or a module-global map —
  the documented next refinement, not a hole: the residual is stable,
  bounded, and identical across sequences.

## Files

- `consistency_table.md` — per-kernel modal tag / agreement / space split
- `attribution_consistency.csv` — the same, machine-readable
- `attribution_by_sequence.csv` — the full kernel × sequence × tag long table
- `coverage.csv` — kernel × sequence capture matrix
