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
| ✅3 | **Cold-start hit rates bias LFMR** (ncu flushes caches between replay passes) | `--cache-control {all,none}` bracket capture: cold (LFMR high) + warm (LFMR low) bound the steady-state truth. Bracket runs are part of the standard chain. |
| ✅4 | **Single-run numbers, no variance** | ×5 nsys / ×3 ncu repeats + `analysis/variance.py` (per-kernel CoV, instance-count determinism check); CoV>10% kernels excluded from headline claims. |
| ✅5 | **Small-sample kernels classified confidently** | n<5 launches → confidence capped `low`, rationale says so. |
| ✅6 | **sync-mode SLAM share (69%) is inflated vs deployment** | paired async capture (`tum_office_slam_async_profile.toml`); report states both numbers with their meanings. |
| ✅7 | **Dataset integrity unverified on new machines** | `fetch_datasets.sh` verifies file count, total bytes, and index-file sha256 against the values used in committed reports. |
| ✅8 | **Analysis not reproducible without dataset/GPU** | all analysis (incl. classification) reruns from committed report CSVs alone. |

## Open — ordered by severity

| # | Issue | What the reviewer says | Status / unblock |
|---|---|---|---|
| 🔴1 | **Prototype GPU, unlocked clocks** | "All numbers come from a 25 W laptop part that can't lock clocks — nothing here is a stable quantity." | Workstation pass (RTX 2000 Ada, locked clocks, ERT-verified ceilings, full repeat protocol). Everything is scripted: `run_characterization.sh --hw hw/rtx2000ada_sm89.toml`. **This is the single highest-value next action.** |
| 🔴2 | **One workload mode, one dataset** (TUM RGBD) | "You characterize *a* configuration of cuVSLAM, not cuVSLAM. Stereo — the flagship mode — is never measured in the report." | Stereo configs are ready (EuRoC V1_01, KITTI 06 + SLAM variants). Blocked on datasets (user will provide; EuRoC server was down, KITTI needs registration). Minimum defensible matrix: {TUM RGBD, EuRoC stereo, KITTI stereo} × {odometry, SLAM}. |
| 🔴3 | **Kernel-level claims, data-structure conclusions** | "You claim the *keyframe database* belongs in ISP but you never observed which allocation the kernel reads." | TaggedAllocator + NVBit alloc-hook correlation (onboarding §11.2) — needs the from-source cuVSLAM build. Until then all claims must stay kernel-scoped (reports are already worded this way). |
| 🟠4 | **No reuse-distance / locality evidence** | "DAMOV's core is locality analysis; your LFMR is a one-point proxy." | Slice-3: NVBit mem_trace → `locality.cpp` (per-warp), gated on driver ≤575 (`blocked/check_capability.sh`); the workstation may unblock it. The cold/warm bracket narrows the gap meanwhile. |
| 🟠5 | **No PiM-side model** — candidacy without a substrate evaluation | "G1/G2 kernels *might* benefit — show me a speedup/energy estimate." | Phase-4 scope: Accel-Sim NDP config (reduced L2, bank-level BW) + AccelWattch energy; report **deltas**. The characterization paper (ISPASS/IISWC) can stand without it; the MICRO/ASPLOS paper cannot. |
| 🟠6 | **No energy numbers** | "PiM's main win is energy; you never measure a joule." | NVML power sampling is feasible today for whole-run energy (add to harness on the workstation); per-kernel needs AccelWattch (Slice 3+). |
| 🟠7 | **G-taxonomy validated by decision tree, not clustering** | "DAMOV derived classes from clustering; you asserted a tree." | The adaptation doc's step 9 (k-means over the metric vectors) once ≥3 datasets exist — the tree then becomes the *labeling*, clustering the *validation*. |
| 🟡8 | **Sub-frame stages not attributed** (NVTX absent) | "Which kernels belong to feature-detect vs tracking is regex over names." | Name-based mapping is documented + tested; NVTX ranges come with the from-source build. Risk is low (names are descriptive) but a reviewer can poke it. |
| 🟡9 | **cuVSLAM is closed-source at this phase** | "Can anyone reproduce your workload?" | The runner pins the public wheel (v15) + configs + datasets are public; the from-source phase upgrades this. Artifact evaluation can run everything headless. |
| 🟡10 | **No repo LICENSE** | Artifact evaluation requires an explicit license. | **User decision needed** — cannot be chosen unilaterally (cuVSLAM wheel EULA interacts with repo licensing). |
| 🟡11 | **Inter-kernel data movement unmeasured** | "GPU-DAMOV §9 says kernel-to-kernel movement matters more on GPUs — you ignore it." | Host↔device side RESOLVED: `analysis/transfers.py` (measured: explicit copies = 41% of kernel time on TUM; H2D 1.68 MB/frame = the sensor upload, i.e. direct near-sensor evidence, in report §5). Inter-kernel reuse still needs Slice-3 traces. |

## Venue framing (honest)

- **Now + workstation pass + 3 datasets** → ISPASS/IISWC characterization
  paper: "GPU-DAMOV applied to a production V-SLAM stack" with the G-taxonomy,
  the loop-closure/ISP finding, and the emergent G7 class as contributions.
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
