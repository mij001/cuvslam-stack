# cuVSLAM Profiling — Full Walkthrough

A step-by-step guide to the profiling phase: what it is, what's built, how to run
it, and what's left. Read top to bottom to get fully up to speed. For the deep
strategy see `PROFILING_PLAN.md`; for command reference see `README.md`.

---

## 0. Status at a glance

| Piece | State | Where |
|---|---|---|
| Researched plan + tooling reality | ✅ done | `PROFILING_PLAN.md` (commit `a27e96d`) |
| Hardware descriptors (MX450 / RTX 2000 Ada / Jetson) | ✅ done | `hw/*.toml` + **auto-generator** `env/gen_hw_descriptor.py` |
| Harness: `profile.py` (nsys + ncu wrappers) | ✅ done | `harness/profile.py` (commits `5d8b65e`, `b659f05`) |
| ncu "no report" bug fixed (targeted metrics) | ✅ done | inline `METRIC_SETS` in `profile.py` |
| **Slice 2 analysis layer** (DAG, screen, roofline, bandwidth, report) | ✅ done | `analysis/` (stdlib-only, headless, SVG figures) |
| **Steady-state capture** (`--auto-window`, `characterize` metric set) | ✅ done | `profile.py` |
| **Loop-closure workloads** (`[slam]`-enabled configs) | ✅ done | `configs/*_slam_profile.toml` |
| **Run-anywhere portability** (dataset var, preflight, fetcher, one-command pipeline) | ✅ done | `env/`, `run_characterization.sh` |
| GPU-free analysis tests | ✅ done | `tests/test_analysis.py` |
| First characterization report (TUM long_office, MX450) | ✅ committed | `reports/` |
| **GPU-DAMOV classification → PiM/ISP candidates** | ✅ done | `analysis/classify.py` (NCU-proxy first cut; report §7); reproducible from committed CSVs — no dataset/GPU needed |
| Workstation (RTX 2000 Ada) locked-clock production pass | ✅ done | 3-workload matrix, CoV 0.14%, measured ceilings; `reports/2026-07-03_*` |
| Multi-dataset generalization (`compare.py` / `cluster.py`) | ✅ done | TUM↔KITTI 97% agreement; k-means prefers k=7; `reports/2026-07-03_matrix_synthesis/` |
| **DAMOV / NVBit / Accel-Sim data-movement track** | 🟡 unblocking | analyzer done (`analysis/locality.py` + mem_trace launch-window patch); driver-downgrade to 575/CUDA-12 in progress on the workstation |
| Source-level attribution (TaggedAllocator, NVTX) | ✅ done | `patches/0002` instrumented wheel + `analysis/attribution.py`; `reports/2026-07-04_attribution/` |
| Jetson AGX Orin re-run | ⏳ later | `hw/jetson_orin_sm87.toml` exists |

---

## 1. What we are doing, and why

**Goal (goal.md):** motivate memory-centric hardware (Processing-in-Memory /
In-Storage-Processing) for Physical AI with a rigorous per-kernel *memory*
characterization of cuVSLAM — for each CUDA kernel: is it compute- or
memory-bound, what is its access pattern, which data does it touch, and how
often.

**Thesis to test** (from the onboarding doc, not assumed): cuVSLAM's data falls
into three *persistence classes* —

- **Streaming** — image data, touched once per frame then discarded → near-sensor SRAM.
- **Hot-persistent** — local map, touched every frame (~MBs) → LPDDR/HBM-PiM.
- **Cold-persistent** — keyframe database, touched only on loop closure (100s of MB) → ISP.

The deliverable is the *evidence* for/against that taxonomy: roofline placement,
access-pattern fingerprints, bandwidth breakdown, DAMOV-style classification.
The `[slam]`-enabled configs exist precisely because the cold-persistent class
has no evidence in an odometry-only run.

Publication path (be honest about it): this characterization is an
ISPASS/IISWC-grade artifact and the prerequisite for a MICRO/ASPLOS/ISCA/HPCA
architecture paper (Phase 4+: an actual PiM/ISP design + simulator evaluation
built on these numbers).

---

## 2. The two methodologies we are fusing

1. **NCU roofline / stall** (the Cao23 "gpudb" lineage). Per kernel:
   Speed-of-Light (Compute% vs Memory%), stall reasons, cache hit rates,
   FLOP-counter arithmetic intensity → roofline. Answers *"is this kernel
   memory-bound, and why."* **Unblocked — this is the spine, and Slice 2
   automates it end-to-end.**

2. **DAMOV data-movement** (CPU→GPU adaptation). NVBit address streams →
   `locality.cpp` reuse/locality → Accel-Sim steady-state cache deltas →
   DAMOV classification. Answers *"what is each data structure's locality /
   PiM-affinity."* **Gated in `blocked/` (see §3); lights up on a ≤575-driver
   host.**

---

## 3. The hardware + tooling reality

Development happens on a laptop **MX450** (sm_75, 14 SM, 512 KiB L2, 64-bit bus,
2 GB, no clock locking), driver **610.43.02**, CUDA 13.2. Real-results runs
happen headless on the **RTX 2000 Ada workstation** (sm_89, 12 MB L2, 224 GB/s,
ECC, lockable clocks) with identical commands — only `--hw` changes.

| Tool | Laptop (driver 610) | Workstation |
|---|---|---|
| cuVSLAM runner | 🟢 | 🟢 |
| Nsight Systems / Compute | 🟢 | 🟢 |
| NVBit → Accel-Sim | 🔴 driver > 575 | 🟢 *if* its driver ≤ 575 (`blocked/check_capability.sh` decides) |

Everything is **hardware-parameterized**: per-GPU facts live in one `hw/*.toml`.
On a brand-new machine, `env/gen_hw_descriptor.py` generates one with exact
structural values (SMs, L2, bus, VRAM, ECC via `cudaDeviceGetAttribute`) and
ceiling *estimates* flagged for ERT verification.

---

## 4. How it fits together (the architecture)

```
   configs/<workload>.toml ──► run.py (Phase-0 TOML runner) ──► cuVSLAM (the work)
        │   ${CUVSLAM_DATASETS} expanded         ▲
        │   by profile.py                        │ launched under a profiler
        ▼                                        │
   profiling/harness/profile.py ─────────────────┘  --profiler {nsys,ncu} --hw hw/<gpu>.toml
        │                                            --metrics characterize
        │                                            --auto-window <nsys_run>:200:300
        ▼
   profiling/results/<date>_<seq>_<profiler>_<hw>/   (raw/ + derived/ + metadata.json)
        │
        ▼  (read-only, stdlib, no GPU)
   profiling/analysis/{build_dag,screen,roofline,bandwidth}.py
        │
        ▼
   analysis/make_report.py ──► profiling/reports/<date>_<hw>/report.md + SVG + CSV  (committed)
```

`run_characterization.sh` chains the whole thing: preflight → nsys baseline →
nsys SLAM → steady-state ncu → report.

---

## 5. What's implemented (step by step)

### 5.1 Plan + descriptors + harness — commits `a27e96d`, `5d8b65e`
Strategy docs, `hw/` descriptors, `profile.py` (versioned results dirs,
mandatory `metadata.json`, targeted metric sets instead of the `--set full`
that dies on 2 GB GPUs).

### 5.2 Slice 2 — commits `b659f05`, `b90f129`
- **`analysis/`** (stdlib-only, figures are hand-emitted SVG — headless anywhere,
  no GPU needed to analyze):
  - `stages.py` — the kernel→stage→persistence-class taxonomy (regex rules over
    the 42-kernel inventory; SLAM-only kernels get their own rules).
  - `build_dag.py` — nsys kernel summary → per-stage time share + kernels/frame.
  - `screen.py` — DAMOV-GPU Step-1 screen: memory-bound / compute-leaning /
    memory-latency / underutilized verdicts + the 9-bucket stall breakdown.
  - `roofline.py` — FLOPs from SASS counters (fadd+fmul+2·ffma, Yang20),
    AI vs ceilings from the hw descriptor.
  - `bandwidth.py` — per-stage DRAM bytes, achieved GB/s vs ceiling, per-frame
    extrapolation (bytes/launch × launches/frame).
  - `classify.py` — the GPU-adapted DAMOV taxonomy (G1 bandwidth / G2
    coalescing / G3 L2-reuse / G4 latency / G5 compute / G6 on-chip / G7
    dependency — G7 emerged from the data, as the adaptation doc prescribed)
    with LFMR_gpu = 1 − L2-hit and MPKI from NCU counters; outputs per-kernel
    class + PiM/ISP affinity + confidence + rationale, and the stage→class→
    substrate synthesis. Accepts results dirs *or* committed report CSVs.
  - `make_report.py` — the committed report: provenance, DAG, screen, roofline,
    bandwidth, loop-closure delta, GPU-DAMOV classification, persistence-class
    evidence table.
- **Harness upgrades:** `characterize` metric set (29 counters: + FLOPs,
  L1/L2 bytes, sectors/request coalescing proxy, full stall set);
  `${CUVSLAM_DATASETS}` expansion; `--auto-window NSYS_DIR:WARM:N` for
  steady-state ncu windows (kernels/frame measured from the nsys run).
- **Workloads:** TUM long_office odometry + SLAM (loop-closure stress, the
  onboarding doc's designated sequence), EuRoC V1_01 pair, KITTI 06 SLAM.
- **Portability:** `env/check_env.sh` (preflight), `env/fetch_datasets.sh`
  (headless, resumable), `env/gen_hw_descriptor.py`, `run_characterization.sh`.
- **Tests:** `tests/test_analysis.py` — 7 GPU-free tests over fabricated CSVs.

### 5.3 Slice 3 gating
`blocked/check_capability.sh` + gated `run_nvbit_memtrace.sh` /
`run_accelsim.sh`. On this laptop they print the exact block reason (driver
610 > NVBit's 575 cap) and the unblock paths.

---

## 6. How to run it

```bash
# new machine, three steps:
profiling/env/check_env.sh
profiling/env/fetch_datasets.sh tum_office
python3 profiling/env/gen_hw_descriptor.py

# everything, one command:
profiling/run_characterization.sh --hw profiling/hw/<gpu>.toml

# or the individual steps — see README.md
```

Results land under `profiling/results/` (gitignored); reports under
`profiling/reports/` (committed).

---

## 7. What's NOT yet implemented

### Workstation real-results pass (next, highest value per hour)
Same commands with `--hw hw/rtx2000ada_sm89.toml`, clocks locked
(`env/lock_clocks.sh`), ≥ 5 repeats, ERT-verified ceilings in the descriptor.
Laptop numbers support methodology; workstation numbers go in the paper.

### Slice 3 — DAMOV / data-movement track (🔴 driver-gated)
Reuse-distance histograms, coalescing/divergence from real address streams,
Accel-Sim steady-state hit-rate **deltas**, the stage → DAMOV-class → PiM/ISP
synthesis table. Unblocks via `blocked/check_capability.sh` on a ≤575-driver
host or a newer NVBit.

### Source-level attribution — DONE 2026-07-04
The address→data-structure mapping (onboarding §11.2) is implemented and run:
`TaggedAllocator` journal in the from-source RelWithDebInfo wheel
(`patches/0002-tagged-allocator-nvtx.patch`), NVBit `cuMemAlloc`/`cuMemHostAlloc`
sidecar (`blocked/mem_trace_alloc_events.patch`), and the
`analysis/attribution.py` resolve+join. NVTX stage ranges enabled via the
same rebuild (`-DUSE_NVTX=ON`). Results: `reports/2026-07-04_attribution/`.
Kernel-arg correlation (Layer 3) remains optional — call-site backtraces
already disambiguate every observed allocation.

### Rigor items for the paper
Multi-dataset generalization (EuRoC + KITTI + TUM VI), the six §12.2 sweeps
(resolution, sequence length, loop rate, feature density, …), ≥ 5 repeats with
distributions, Sieve representative-invocation sampling, Jetson Orin (Phase 3.5).

---

## 8. Directory map

```
profiling/
├── PROFILING_PLAN.md            ✅ strategy
├── README.md                    ✅ command reference
├── WALKTHROUGH.md               ✅ this file
├── run_characterization.sh      ✅ one-command pipeline
├── hw/                          ✅ mx450 · rtx2000ada · jetson_orin (+ auto-generated)
├── env/                         ✅ check_env · fetch_datasets · gen_hw_descriptor · lock_clocks · setup_perms
├── harness/profile.py           ✅ the entrypoint
├── configs/                     ✅ tum_office{,_slam} · euroc_v101{,_slam} · kitti06{,_slam}
├── analysis/                    ✅ stages · build_dag · screen · roofline · bandwidth · svgfig · make_report
├── tests/                       ✅ test_analysis.py (GPU-free)
├── blocked/                     🔴 check_capability · nvbit · accelsim (driver-gated)
├── results/                     (runtime, gitignored)
└── reports/                     ✅ committed characterization reports
```

---

## 9. Decisions on record

1. **Hardware:** hardware-parameterized; prototype on MX450, real runs on RTX 2000 Ada.
2. **Scope:** NCU-first, NVBit-gated (evidence-based deviation from the onboarding order).
3. **Instrumentation:** wheel-only until 2026-07-04; the from-source phase added `patches/0002-tagged-allocator-nvtx.patch` (allocation journal + NVTX), env-gated so the default build is bit-identical in behavior.
4. **Slice-2 capture fidelity:** proper steady-state (warm 200 frames, then profile).
5. **Figures:** dependency-free SVG instead of matplotlib — the repo must run headless anywhere with zero pip installs.
6. **ncu metric curation:** inline `METRIC_SETS` in `profile.py` (kept from Cao23's taxonomy) instead of vendoring `counter_config.py` — one file, one source of truth; cite [Cao23].
7. **Loop closure:** TUM fr3 long_office is the canonical loop-closure workload (EuRoC's ETH server proved unreliable; KITTI needs manual registration).
8. All commits authored `mij001`; no AI-attribution trailers.

---

## 10. Key facts worth remembering

- **NVBit is blocked on driver > 575** — `blocked/check_capability.sh` is the gate.
- **`ncu --set full` fails on 2 GB GPUs** — always a targeted metric set
  (`--metrics characterize` for report captures).
- **ncu hit rates are cold-start** (cache flush between replay passes); SoL,
  stalls and byte counters are robust. Steady-state hit rates need Slice 3.
- **This MX450 has 512 KiB L2** — reuse cutoff here is ~512 KiB vs 12 MB on the
  Ada. Cite the hardware descriptor, not generic numbers.
- **The runner is the workload** — any dataset = one TOML.
- **Report deltas, not absolutes**, for anything simulated.
- **Laptop-local quirk (2026-07):** the system CUDA package was corrupted by a
  disk-full update; until `sudo pacman -S cuda`, workload runs need
  `LD_LIBRARY_PATH=~/.local/cuda-repair/lib`. The workstation is unaffected.
