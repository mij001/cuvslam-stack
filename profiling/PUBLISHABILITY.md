# Publishability & Reproducibility Register

A reviewer-grade audit of this characterization against MICRO/ASPLOS/ISCA/HPCA
standards. Each issue: what a reviewer would say, severity for the target
venue, and the exact status/unblock path. Kept current — this is the working
checklist between here and submission.

Severity: 🔴 blocks submission · 🟠 guaranteed reviewer pushback · 🟡 weakens
the paper · ✅ resolved (kept for the record).

## Resolved (2026-07-03 hardening pass)

| # | Issue | Resolution |
|---|---|---|
| ✅1 | **Rooflines drawn against marketing ceilings** (80 GB/s spec vs reality) | `env/measure_ceilings.py`: measured 45.7 GB/s DRAM / 1228 GFLOP/s FP32 on the MX450 (median-of-7, clock-sampled, sync-per-op); descriptors carry `*_measured` fields; figures label ceiling provenance. Desktop-safe by design (queue depth 1, VRAM-budgeted, time-boxed). |
| ✅2 | **Classification thresholds asserted, not stress-tested** | ±25% threshold perturbation on every kernel; `stability` column; borderline kernels can't carry `high` confidence. Result on the TUM pass: 36/47 stable, and **all headline kernels are stable**. |
| ✅3 | **Cold-start hit rates bias LFMR** (ncu flushes caches between replay passes) | `--cache-control {all,none}` bracket capture, MEASURED on TUM: headline kernels' brackets are tight (`gaussian_scaling` 0.55→0.46, `reduced_system_stage_2` 0.05→0.03 — classes survive); wide-bracket kernels (`build_full_system_2` 0.38→0.00) were already flagged borderline. |
| ✅4 | **Single-run numbers, no variance** | ×5 nsys / ×3 ncu repeats + `analysis/variance.py`, MEASURED: instance counts deterministic to 0.13%; ncu Mem-SoL CoV 1–5% (headline kernels), 9.7% median; raw time swings 3.4× from DVFS → `--gpu-warmup 8` protocol collapses CoV 49.6→9.3% (time) / 16.3→5.8% (share). Statistic hierarchy documented in METHODOLOGY §4.2. |
| ✅5 | **Small-sample kernels classified confidently** | n<5 launches → confidence capped `low`, rationale says so. |
| ✅6 | **sync-mode SLAM share (69%) is inflated vs deployment** | paired async capture, MEASURED: st_* kernels = 69.4% sync / **50.6% async** of GPU time — the ISP claim survives deployment mode. |
| ✅7 | **Dataset integrity unverified on new machines** | `fetch_datasets.sh` verifies file count, total bytes, and index-file sha256 against the values used in committed reports. |
| ✅8 | **Analysis not reproducible without dataset/GPU** | all analysis (incl. classification) reruns from committed report CSVs alone. |

## Open — ordered by severity

| # | Issue | What the reviewer says | Status / unblock |
|---|---|---|---|
| ✅1 | **Prototype GPU, unlocked clocks** | "All numbers come from a 25 W laptop part that can't lock clocks — nothing here is a stable quantity." | **CLOSED 2026-07-03.** Locked-clock RTX 2000 Ada pass complete (persistence + `-lgc 1620,1620` / `-lmc 7001,7001`): 5-repeat CoV median **0.14%** (was 49.6% unlocked laptop / 9.3% warmed); ceilings measured at lock (205.0 GB/s ±0.1, 5445 GFLOP/s ±3). Reports: `2026-07-03_*_rtx2000ada`. Locks reset on reboot — re-apply after power events. |
| ✅2 | **One workload mode, one dataset** (TUM RGBD) | "You characterize *a* configuration of cuVSLAM, not cuVSLAM. Stereo — the flagship mode — is never measured in the report." | **FULL-SCALE CAMPAIGN 2026-07-04** (`reports/2026-07-04_campaign/`): **27 sequences × 4 datasets** (KITTI 00-10, EuRoC MH/V1/V2 ×11, TUM fr3 ×4 incl. texture/structure ablations, TUM-VI), each odometry + SLAM, locked-clock Ada, **0 failures**. Cross-sequence modal consistency **91%** (24/49 kernels unanimous, 42/49 ≥80%); remaining flips are the physically-meaningful L2 crossover, not noise. Stereo (KITTI+EuRoC) is now the bulk of the matrix. |
| ✅3 | **Kernel-level claims, data-structure conclusions** | "You claim the *keyframe database* belongs in ISP but you never observed which allocation the kernel reads." | **CLOSED 2026-07-04** (`reports/2026-07-04_attribution/`). The §11.2 three-layer pipeline built and run: TaggedAllocator journal (`patches/0002-tagged-allocator-nvtx.patch`, RelWithDebInfo wheel), NVBit `cuMemAlloc`/`cuMemHostAlloc` sidecar (`blocked/mem_trace_alloc_events.patch`), `analysis.attribution` resolve+join. All wrapper allocations resolve to owning data structures via backtrace+addr2line. Headline: the GPU memory budget is **static** (keyframe state a fixed 6.7 MB) — the session-scale database grows host-side, confirming the Slice-3 ISP re-grounding from the allocator itself. |
| ✅4 | **No reuse-distance / locality evidence** | "DAMOV's core is locality analysis; your LFMR is a one-point proxy." | **DONE 2026-07-04** (`reports/2026-07-04_slice3_locality/`). Slice 3 unblocked (driver 575.64.05/CUDA 12.9/linux-lts; cu12 wheel rebuilt; NVBit mem_trace + KERNEL_FILTER patch). Measured from real per-warp addresses: front-end reuse CDF **flat across 64 KiB→48 MiB** (cache-immune streaming, proven not inferred); st_track footprint 0.47→1.10 MB with 99.9% reuse <64 KiB (L2-resident) and inter-launch migration (Jaccard 0.67→0.90). Divergence axis added (all kernels 32.0 active lanes = converged). **Trace overturned the counter proxy** — see the correction row below. |
| ✅NEW | **Counter-vs-trace disagreement on the loop-closure kernel — RESOLVED: the counter was right** | "Your §7 classifier called st_track a scattered gather; the trace says coalesced." | Full arc, a reviewer-facing strength: (1) 2026-07-04 the unfiltered trace read st_track as coalesced streaming and the report claimed "proxy overturned"; (2) the attribution join then showed 94% of the kernel's accesses are register-spill; (3) the 2026-07-05 space-filtered re-derivation (Slice-3 FINDINGS §5, `data_v2/`) shows the *global* accesses are **23–30 sectors/warp, 2–6% coalesced — a scattered gather, matching ncu's 18–30 sectors/request**. Two independent methods now agree; the **G2-scatter label stands**. The ISP case keeps session-scale DB growth/migration (footprint+Jaccard rows unaffected by the correction) and regains within-scan scatter; the spill stream adds a register-file/spill-SRAM ask. |
| ✅NEW2 | **Slice-3 locality mixed memory spaces** (found by the attribution join, 2026-07-04) | "Your reuse/footprint numbers include shared-memory tiles and local-spill windows — those aren't data locality." | **CLOSED 2026-07-05.** `analysis/locality.py --spaces {global,shared,local,all}` (default global); Slice-3 re-derived from the kept traces into `reports/2026-07-04_slice3_locality/data_v2/` with a dated correction section (§5). Outcomes: st_track coalescing verdict reversed (see NEW row); front-end streaming claim *strengthened* (conv_grad's apparent scatter/reuse was shared-memory tiles; global-only it is pure coalesced streaming, so the "conv_grad_y data-layout target" sub-finding is withdrawn); footprints/Jaccard essentially unchanged (spill window ≈3 KB). The campaign traces (`/mnt/data/attribution_out/`) support the same re-derivation at 27-sequence scale when needed. |
| 🟠5 | **No PiM-side model** — candidacy without a substrate evaluation | "G1/G2 kernels *might* benefit — show me a speedup/energy estimate." | Phase-4 scope: Accel-Sim NDP config (reduced L2, bank-level BW) + AccelWattch energy; report **deltas**. The characterization paper (ISPASS/IISWC) can stand without it; the MICRO/ASPLOS paper cannot. |
| 🟠6 | **No energy numbers** | "PiM's main win is energy; you never measure a joule." | NVML power sampling is feasible today for whole-run energy (add to harness on the workstation); per-kernel needs AccelWattch (Slice 3+). |
| ✅7 | **G-taxonomy validated by decision tree, not clustering** | "DAMOV derived classes from clustering; you asserted a tree." | **Pooled k-means over the 27-sequence campaign** (`reports/2026-07-04_campaign/`, `analysis/campaign.py`): best silhouette at **k=7–8** (matching the 7 G-classes), purity 0.68, ARI 0.30 vs the decision-tree labels, monotone in k. The classes fall out of the combined feature cloud at scale — the tree is the labeling, clustering the independent validation. |
| ✅8 | **Sub-frame stages not attributed** (NVTX absent) | "Which kernels belong to feature-detect vs tracking is regex over names." | **CLOSED 2026-07-04.** cuVSLAM's own profiler domains enabled (`-DUSE_NVTX=ON` + `profiler_enable.h` flip in `patches/0002`); nsys `nvtx_kern_sum` gives the measured kernel→stage table (`reports/2026-07-04_attribution/nvtx_kern_sum.csv`). Notably `st_track_with_cache` sits under **SLAM:LC & optimization** and `st_build_cache` under SLAM keyframe ingest — loop-closure attribution is measured, not name-inferred. |
| 🟡9 | **cuVSLAM is closed-source at this phase** | "Can anyone reproduce your workload?" | The runner pins the public wheel (v15) + configs + datasets are public; the from-source phase upgrades this. Artifact evaluation can run everything headless. |
| 🟡10 | **No repo LICENSE** | Artifact evaluation requires an explicit license. | **User decision needed** — cannot be chosen unilaterally (cuVSLAM wheel EULA interacts with repo licensing). |
| 🟡11 | **Inter-kernel data movement unmeasured** | "GPU-DAMOV §9 says kernel-to-kernel movement matters more on GPUs — you ignore it." | Host↔device side RESOLVED: `analysis/transfers.py` (measured: explicit copies = 41% of kernel time on TUM; H2D 1.68 MB/frame = the sensor upload, i.e. direct near-sensor evidence, in report §5). Inter-kernel reuse still needs Slice-3 traces. |

## Venue framing (honest)

- **Now (workstation pass + 3-dataset matrix both DONE)** → ISPASS/IISWC
  characterization paper is **submittable**: "GPU-DAMOV applied to a production
  V-SLAM stack" with the G-taxonomy, the loop-closure/ISP finding + measured
  L2 crossover, and the emergent G7 class as contributions.
- **+ Slice-3 (traces, sim, clustering) + TaggedAllocator** → the data-structure-
  level characterization that motivates a design.
- **+ PiM/ISP substrate design + delta evaluation + energy** → the
  MICRO/ASPLOS/ISCA/HPCA submission. The characterization above becomes §3–4
  of that paper.

## Standing rules (enforced by the tooling)

1. No number without provenance (`metadata.json` or it didn't happen).
2. No roofline against an unmeasured ceiling.
3. No classification without stability + sample-size flags.
4. No headline claim from a kernel with time-CoV > 10%.
5. Simulated numbers are deltas, never absolutes.
6. Laptop numbers argue methodology; only locked-clock numbers argue results.
