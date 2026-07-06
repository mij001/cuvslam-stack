---
name: cuvslam-stack-profiling-state
description: cuVSLAM PiM/ISP profiling — memory characterization + attribution complete; accuracy validation vs NVIDIA paper done 2026-07-06 (stereo/RGB-D reproduce, instrumentation neutral); next = Accel-Sim/energy (Phase 4)
metadata: 
  node_type: memory
  type: project
  originSessionId: 83c42e24-7c04-403d-b5f5-5135ef6002f1
---

Project: motivate PiM/ISP for Physical AI via per-kernel memory characterization
of cuVSLAM (goal.md; DAMOV+Cao23 methodology; ISPASS/IISWC → MICRO/ASPLOS path).

DONE (all committed+pushed, branch main): locked-clock Ada pass (CoV 0.14%);
27-seq campaign (91% modal consistency, k=7–8, 61% PiM affinity); Slice-3
locality (front-end cache-immune streaming; st_track story CORRECTED 2026-07-05
— space-filtered re-derivation, FINDINGS §5: global accesses are a scattered
gather, ncu counters were right, the 'coalesced' signal was the spill stream;
G2-scatter label stands);
and 2026-07-04 **TaggedAllocator + NVTX attribution (M7)** — commit 0e237fa,
`reports/2026-07-04_attribution/`. Instrumented cu12 wheel (RelWithDebInfo,
`patches/0002`, USE_NVTX=ON; installed in ws `cuvslam_venv_tagged`), NVBit alloc
sidecar (`blocked/mem_trace_alloc_events.patch`), `analysis/attribution.py`
resolve+join. KEY findings: GPU memory budget STATIC (240 allocs / 108.65 MB
device + 34 / 17.1 MB pinned-host over 2500 frames; keyframe state fixed
6.7 MB → DB growth is host-side LMDB); st_track_with_cache DRAM traffic = 94%
register spill + 5% keyframe descriptors; SBA = 97% ba_linear_system; front-end
89–98% shared-tile. NVTX kernel→stage map measured. PUBLISHABILITY 1,2,3,4,7,8
closed. Join rule that made it work: memory-space buckets (LDS/STS→
shared_onchip, LDL/STL→local_spill) — without them 88–98% looks "unmapped".

DONE 2026-07-05: **full-matrix attribution campaign** — all 27 SLAM sequences
captured to `/mnt/data/attribution_out/` (~85 GB, workstation sda), coverage
audited + gap-filled over 3 iterations to **0 missing kernels** (planner
anchors on dense launch clusters — isolated sparse-kernel occurrences drift
between runs). Synthesis (`analysis/attribution_campaign.py`, commit a31c13b):
**48/49 kernels have a unanimous top data-structure tag across all sequences**
(`reports/2026-07-05_attribution_campaign/`). Keyframe pair uniform at room and
street scale: st_build_cache→keyframe_descriptors; st_track→91-95% spill +
5-8% descriptor scatter. Residuals bounded: build_full_system_2 ~40% unmapped
everywhere + lk_track/getrf unmapped = static module memory + texture-path
reads → pinning them = the Layer-3 refinement (documented, optional).

Slice-3 CORRECTED same day (commit db31d30, FINDINGS §5 + data_v2/): global-only
st_track is a scattered gather (23-30 sectors/warp) — ncu counters confirmed,
"proxy overturned" withdrawn, G2-scatter stands; front-end streaming claim
strengthened (conv scatter was shared tiles). PUBLISHABILITY NEW+NEW2 closed.

DONE 2026-07-06: **accuracy validation** (`reports/2026-07-06_accuracy/`,
commits 1cdb380/031f95d + earlier f1863f8 tooling). Ran the full 104-run config
matrix (all datasets × stereo/inertial/mono/RGBD × odom/slam/sync/async/GPU/CPU)
on the workstation → `/mnt/data/accuracy_out/`; `accuracy_report.py` compares to
the cuVSLAM paper (arXiv:2506.04359) Tables 2/6 with a convergence gate
(avgRTE<5%, 62/104 converged). RESULT: the profiled modes (stereo, RGB-D SLAM)
REPRODUCE the paper (EuRoC stereo APE 0.114/0.051m vs 0.13/0.054; TUM long_office
beats it; KITTI 500m-drift 0.82% vs leaderboard 0.85%); QoR = instrumented wheel
== baseline (EuRoC bit-identical) → memory characterization is accuracy-neutral
= valid. Discrepancies explained: inertial UNDER-TUNED (generic IMU config not
the paper's per-dataset calibration; 3/11 converge; config-only fix = G9);
V2_03_difficult diverges all modes (paper excludes it too); TUM-VI = fisheye not
undistorted (invalid input, G10); mono needs Sim3. This is thesis finding F13.

OPEN: (a) LICENSE (user decision). (b) Accel-Sim NDP + AccelWattch → Phase 4
substrate design (the architecture paper). (c) optional Layer-3 kernel-arg
correlation to name the static-memory residuals. (d) accuracy follow-ups (G9/G10,
low priority, don't affect characterization): tune EuRoC IMU from sensor.yaml +
re-run 22 inertial configs; undistort TUM-VI + Sim3 mono eval.

Workstation notes: sda2 (datasets, 3.6T NTFS) is fstab-ro with a dirty bit —
rw needs `sudo mount -t ntfs3 -o rw,force /dev/sda2 /mnt/data` (the campaign
preamble re-asserts this after power cuts; ntfsfix unavailable).
cuvslam_src_cu12 has patch 0002 applied (`.orig` backups); baseline wheel kept
at `~/Projects/cuvslam_src_cu12/dist/baseline/`. KITTI images were deleted from
the laptop's ~/Projects/cuvslam_datasets (poses only; laptop disk nearly full) —
campaign datasets live on the ws at /mnt/data. See [[workstation-access]],
[[cuda-corruption-ld-library-path]].
