---
name: cuvslam-stack-profiling-state
description: cuvslam-stack research state — Slice 2 built 2026-07-02; goal is MICRO/ASPLOS-grade PiM/ISP characterization of cuVSLAM
metadata: 
  node_type: memory
  type: project
  originSessionId: 83c42e24-7c04-403d-b5f5-5135ef6002f1
---

The cuvslam-stack repo pursues goal.md: memory-centric architectures (PiM/ISP) for Physical AI, via per-kernel memory characterization of cuVSLAM (methodology: DAMOV [Oliveira21] + Cao23 NCU roofline; target venue path ISPASS/IISWC characterization → MICRO/ASPLOS architecture paper).

As of 2026-07-04: the ISPASS/IISWC-grade characterization is essentially COMPLETE. Milestones done & committed: (1) locked-clock RTX 2000 Ada production pass (CoV 0.14%, measured ceilings 205 GB/s/5445 GF); (2) FULL-SCALE 27-sequence campaign (KITTI 00-10, EuRoC MH/V1/V2 ×11, TUM fr3 ×4 incl. ablations, TUM-VI), odom+SLAM, 0 failures — reports/2026-07-04_campaign/, analysis/campaign.py; modal class consistency 91%, pooled k-means k=7-8 (validates taxonomy), 61% of GPU-time carries PiM affinity; (3) SLICE-3 locality from real NVBit traces (reports/2026-07-04_slice3_locality/) — front-end reuse CDF flat=cache-immune streaming; and the KEY result: the address trace OVERTURNED the ncu counter proxy on st_track (it's coalesced converged streaming, NOT the scattered gather the counters implied → ISP case re-grounds to session-scale DB growth, a stronger ask). PUBLISHABILITY issues 1,2,4,7 CLOSED.

NEXT UP (the plan): TaggedAllocator + NVTX from-source build for data-structure-level attribution (turns "st_track is memory-bound" into "the keyframe DB belongs in ISP" by observing which allocation each kernel reads) — the cu12 cuVSLAM source tree is on the workstation at ~/Projects/cuvslam_src_cu12. Then Accel-Sim NDP config + AccelWattch energy, then the PiM/ISP substrate design (Phase 4 = the architecture paper). Still open: LICENSE (user decision). Raw campaign derived data gitignored at profiling/results_ws_campaign/ (35MB); the committed 876KB report reproduces the synthesis. See [[workstation-access]] for the 575/CUDA-12.9/ncu-2025.2 unified stack.

**Why:** the repo docs (PROFILING_PLAN.md, WALKTHROUGH.md) carry the full plan; this memory just anchors the cross-session thread.

**How to apply:** run `profiling/run_characterization.sh` for the full pipeline; datasets fetch via `profiling/env/fetch_datasets.sh` (EuRoC's ETH server was unreachable 2026-07-02; TUM cvg.cit.tum.de worked). KITTI images were deleted from ~/Projects/cuvslam_datasets (only poses remain); disk is nearly full. Related: [[cuda-corruption-ld-library-path]].
