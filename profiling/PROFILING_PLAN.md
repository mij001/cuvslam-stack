# cuVSLAM Memory-Profiling — Researched Plan (Phase 1+)

> Status: **active**. This is the engineering plan for extending the cuvslam-stack
> umbrella from "runs cuVSLAM and scores accuracy" (Phase 0) to "characterizes
> cuVSLAM's memory behaviour kernel-by-kernel." It is grounded in the guiding
> documents under `suggestions_and_summuries/` **and** in empirical tests of what
> actually runs on the available hardware. Where the evidence contradicts the
> onboarding doc, this plan follows the evidence and says so.

---

## 1. Purpose and thesis

We are doing for cuVSLAM what DAMOV [Oliveira21] did for its benchmark suite and
what Cao et al. [Cao23] did for GPU databases: a rigorous, **per-kernel memory
characterization** that says which parts of the pipeline are memory-bound, what
their access patterns are, and therefore which are candidates for memory-centric
hardware (Processing-in-Memory / In-Storage-Processing).

**Thesis (from the onboarding doc, to be tested, not assumed):** cuVSLAM's
accesses fall into three persistence classes —

| Class | Touched | Size | Hardware affinity |
|---|---|---|---|
| **Streaming** | once per frame, then discarded | scales with image | near-sensor SRAM |
| **Hot-persistent** | every frame (local map) | ~1–10 MB | LPDDR/HBM-PiM |
| **Cold-persistent** | only on loop closure (keyframe DB) | 100s of MB | ISP / computational storage |

The deliverable of this phase is the **evidence** for or against that taxonomy:
roofline placement, access-pattern fingerprints, bandwidth breakdown, and a
DAMOV-style classification table per cuVSLAM stage.

This is the characterization that any later accelerator/PiM design would cite. It
stands alone as an ISPASS/IISWC-style workload-characterization artifact.

---

## 2. Two methodologies, fused — and the assets already on disk

The phase fuses two complementary, pre-existing methodologies. Both have their
reference material and code already present in this repo's working tree.

| Lineage | Question it answers | On-disk asset | Tool |
|---|---|---|---|
| **NCU roofline / stall** (Cao23 "gpudb") | Is each kernel compute- or memory-bound, and *why*? | `suggestions_and_summuries/gpudb-*.md`, `capstone-paper-repo-and-generalization*.md`, `gpudb-perf.ncu-rep`; the forked tooling in `external_repos`/sibling dirs | Nsight Compute + `counter_config.py` |
| **DAMOV data-movement** (CPU→GPU adaptation) | What is each data structure's locality / reuse / PiM-affinity? | `suggestions_and_summuries/DAMOV*.{md,pdf}`, `Adapting_DAMOV_to_GPU.md`; `external_repos/DAMOV-main/` (incl. reusable `simulator/src/locality.cpp`); `external_repos/nvbit_release_x86_64/` | NVBit → `locality.cpp` → Accel-Sim |

The capstone doc explicitly generalises the NCU method to "a completely different
workload" — **cuVSLAM is that workload.** The DAMOV-to-GPU doc maps every DAMOV
component to a GPU analog (Pin→NVBit, ZSim+Ramulator→Accel-Sim/GPGPU-Sim,
VTune→NCU/Nsys, CACTI→AccelWattch) and re-derives the bottleneck taxonomy for
GPUs (latency class collapses; coalescing / L2-contention / occupancy classes
emerge).

---

## 3. The decisive research result: what actually runs here

The onboarding doc targets a Dell Precision 7875 workstation with an **RTX 2000
Ada (sm_89)**. The current development box is a laptop **MX450 (sm_75 Turing,
2 GB)**, driver **610.43.02**, CUDA **13.2**, with `ncu 2026.2` and `nsys 2026.1`
installed. Empirical findings (tested, not assumed):

| Track | Status | Evidence |
|---|---|---|
| cuVSLAM execution | 🟢 works | a prior `nsys` run captured a real `.nsys-rep` on EuRoC V1_01 |
| **Nsight Systems** (timeline → DAG) | 🟢 works | 180 KB `.nsys-rep` produced |
| **Nsight Compute** (roofline/stall) | 🟡 works, needs the right invocation | `ncu` smoke test profiled a kernel (`gpu__time_duration.sum = 2.75 µs`); `RmProfilingAdminOnly: 0` (perms OK). Prior runs produced **no `.ncu-rep`** because `--set full` on a 2 GB GPU is killed before it finishes. |
| **NVBit** (mem-trace, alloc-tags) | 🔴 blocked | NVBit README requires **CUDA driver ≤ 575.xx**; this box is **610**. The shipped `libnvbit.a` (Apr 2025) will not inject. |
| **Accel-Sim** (steady-state cache) | 🔴 blocked | depends on NVBit traces; also no validated sm_89 config |

**Conclusion that shapes the plan.** The v5 doc's headline artifacts
(reuse-distance, Accel-Sim hit-rate deltas) ride on the NVBit→locality→Accel-Sim
track, which is **blocked by the driver-610 > NVBit-575 incompatibility** on this
hardware. The fully-unblocked, high-signal path is the **Nsight Systems + Nsight
Compute characterization**. Therefore:

> **NCU/Nsys is the spine of this phase. The DAMOV/NVBit/Accel-Sim
> data-movement track is wired but gated behind an explicit capability check; it
> lights up the moment a compatible NVBit/driver pair exists (a newer NVBit that
> supports ≥ 610, or a ≤ 575 driver — e.g. on the RTX 2000 Ada workstation).**

This is a deliberate, evidence-based deviation from the onboarding doc's phase
ordering, not a reduction in ambition: every DAMOV step stays in the design,
ready to run, but the phase produces real results now instead of stalling on a
blocked toolchain.

---

## 4. Hardware-parameterized design

Everything that differs between GPUs lives in **one descriptor file** under
`profiling/hw/*.toml` (SM count, L2 size, DRAM bandwidth, FP32 peak, clock-lock
policy, Accel-Sim config hint). Scripts and analysis read ceilings/constants from
there. This is what lets the harness develop on the MX450 and run for real on the
RTX 2000 Ada (and later Jetson Orin) with no code change — only a `--hw` flag.

Descriptors provided: `mx450_sm75.toml` (this box), `rtx2000ada_sm89.toml`
(workstation target, the doc's hardware), `jetson_orin_sm87.toml` (Phase 3.5).

---

## 5. Repo integration — the `profiling/` subsystem

The elegant link to Phase 0: **the existing TOML runner is the workload-under-test.**
A profiler wrapper takes a runner config + a profiler + a hardware descriptor and
emits a versioned results directory. Nothing in Phase 0 changes.

```
profiling/
├── PROFILING_PLAN.md     # this file (strategy)
├── README.md             # operational how-to
├── hw/                   # mx450_sm75.toml · rtx2000ada_sm89.toml · jetson_orin_sm87.toml
├── env/                  # lock_clocks.sh · setup_perms.sh · requirements snapshot · system_info
├── harness/              # profile.py + run_nsys.sh + run_ncu.sh  → results/<date>_<seq>_<profiler>_<hw>/
├── ncu_tooling/          # vendored Cao23 counter_config.py + ncu_parser (targeted metric sets)
├── analysis/             # build_dag.py · roofline.py · stall_breakdown.py · bandwidth.py (read-only consumers)
├── blocked/              # run_nvbit_memtrace.sh · run_accelsim.sh · locality/  (driver-gated, fail-fast w/ reason)
└── results/              # versioned, never overwritten; metadata.json mandatory per run
```

Results-dir discipline (from the onboarding doc, retained): every run is a
timestamped directory with a mandatory `metadata.json` (GPU, driver, CUDA,
versions, sequence, frame range, exact command); `raw/` and `derived/` are
separate so analysis can be re-run without re-collecting; `analysis/` only ever
*reads* `results/`.

---

## 6. Work plan — three slices

### Slice 1 — Consolidate + make NCU actually produce a report  *(the unblock)*
- Stand up `profiling/` from the working parts of the prior prototype in
  `/home/m_inomal/Projects/cuvslam profiling/` (run scripts, metadata schema,
  `lock_clocks.sh`, `env/` snapshot, forked `ncu_tooling/`).
- `profile.py`: one entrypoint that wraps the **stack's** TOML runner
  (`run.py <config>`) under nsys or ncu, reads a `--hw` descriptor, writes the
  versioned results dir + `metadata.json`.
- **Fix ncu:** replace `--set full` with a **targeted metric set** drawn from
  `counter_config.py` (SoL + roofline + stall + memory-workload subset), trim the
  launch window, and confirm `-o` writes into `results/.../raw/`. Validate that a
  real `.ncu-rep` lands.
- **Deliverable:** the first real cuVSLAM `.ncu-rep` + a parsed per-kernel
  SoL/roofline table on the MX450.

### Slice 2 — Characterization  *(the signal)*
- `build_dag.py`: turn the nsys timeline into the kernel→stage DAG (kernel launch
  order per frame), seeded from the cuVSLAM paper's module list and the source's
  ~11 CUDA kernels; tag each stage with a persistence class hypothesis.
- `roofline.py` / `stall_breakdown.py` / `bandwidth.py`: hierarchical roofline
  (L1/L2/DRAM ceilings from `hw/`), the DAMOV-GPU Step-1 screen (keep kernels with
  Memory% ≫ SM% **or** a memory stall dominant), stall pies, per-stage DRAM
  bandwidth bars.
- **Deliverable:** a cuVSLAM characterization report — DAG + roofline + bottleneck
  screen + bandwidth breakdown. The genuinely novel artifact.

### Slice 3 — DAMOV / data-movement track  *(gated)*
- Wire NVBit `mem_trace` + the lifted `locality.cpp` + Accel-Sim **behind a
  capability check** (`driver ≤ 575 && NVBit present`) that fails fast with the
  reason and the unblock instructions.
- Until unblocked: approximate reuse/locality from NCU memory-workload counters
  (hit rates, sectors/request) where defensible, and clearly label them as
  profiler-derived (cold-start) rather than steady-state.
- **Deliverable (when unblocked):** reuse-distance histograms, coalescing /
  divergence metrics, Accel-Sim steady-state hit-rate **deltas** (report deltas,
  not absolutes, per the simulator-methodology caveat), and the
  stage → DAMOV-class → PiM/ISP-affinity synthesis table.

---

## 7. Deviation from the onboarding doc (and why)

| Onboarding v5 | This plan | Why |
|---|---|---|
| Phase 2/3 lean on NVBit + Accel-Sim early | NVBit/Accel-Sim **gated to Slice 3** | driver 610 > NVBit 575 cap — blocked on this box |
| `ncu --set full` | **targeted metric set** | `--set full` is killed on a 2 GB GPU before writing a report |
| RTX 2000 Ada assumed | **hw-parameterized**, prototype on MX450 | the doc's GPU isn't the dev box; build portable |
| Roofline/reuse-distance need the simulator | **NCU-roofline first**; sim deltas later | NCU is unblocked and high-signal today |
| Standalone profiling repo | a **`profiling/` subsystem driving the Phase-0 TOML runner** | reuse "TOML is the only input"; one umbrella |

Everything else (results-dir discipline, clock locking, reproducibility runs,
the persistence taxonomy, the DAMOV classification goal) is retained.

---

## 8. Milestones (adapted from the v5 table)

| # | Milestone | Concrete artifact | Status |
|---|---|---|---|
| M1 | Harness operational | `profile.py` + nsys/ncu wrappers + `hw/` descriptors; versioned `results/` | ✅ |
| M2 | **First NCU report** | a real cuVSLAM `.ncu-rep` + parsed per-kernel table (Slice 1 done) | ✅ |
| M3 | DAG + roofline | kernel→stage DAG, roofline w/ FLOP counters, bottleneck screen (`analysis/`) | ✅ |
| M4 | Bandwidth + first report | per-stage DRAM bandwidth; committed report under `reports/` incl. the loop-closure (cold-persistent) delta | ✅ |
| M4.5 | Run-anywhere headless | `${CUVSLAM_DATASETS}` configs, `env/gen_hw_descriptor.py`, `check_env.sh`, `fetch_datasets.sh`, `run_characterization.sh`, GPU-free tests | ✅ |
| M4.75 | **GPU-DAMOV classification** | `analysis/classify.py`: NCU-proxy bottleneck classes (Adapting_DAMOV_to_GPU §6, + emergent G7) → per-kernel PiM/ISP affinity + stage synthesis; reproducible from committed CSVs with no dataset/GPU | ✅ |
| M6 | Workstation re-run | RTX 2000 Ada, **locked clocks**, ×5 repeats (CoV 0.14%), measured ceilings; 3-workload matrix + `compare.py`/`cluster.py` generalization | ✅ |
| M5 | Data-movement track | NVBit/locality/Accel-Sim on a ≤575-driver host — refines M4.75 with reuse distance, divergence, sim deltas | 🟡 unblocking — `analysis/locality.py` + mem_trace launch-window patch committed; workstation downgrade to 575/CUDA-12 in progress |
| M7 | Source-level attribution | TaggedAllocator + NVTX from-source build → data-structure-level claims (onboarding §11.2) | ✅ done (`reports/2026-07-04_attribution/`) |

---

## 9. Reproducibility rules (retained)

Clocks locked (or recorded as un-lockable on laptops), CPU governor noted, ECC
state recorded, no other GPU jobs, identical sequence + frame range, discard
warm-up frames, ≥ 5 repeats with mean + 95th percentile for any timing claim.
Architecture metrics (bandwidth, hit rate, instruction counts) are stable across
RANSAC nondeterminism; only trajectory error swings — report distributions.

---

## 10. Provenance — what was consolidated from where

- **Base:** `/home/m_inomal/Projects/cuvslam profiling/` (run scripts, `config.toml`
  schema, `metadata.json` provenance, `lock_clocks.sh`, `env/`, forked
  `ncu_tooling/`, working nsys results).
- **Python bits:** `/home/m_inomal/Projects/prifile-cuvslam/cuvslam_profiler/`
  (`profile_runner.py`, `ncu_parser.py`, `extract_stats.py`).
- **NCU tooling origin:** Cao23 `gpudb-char-and-opt` (`counter_config.py`,
  `report_parser/ncu_parser.py`, `stats/flush_ncu_csv.py`) — cite [Cao23].
- **DAMOV reuse:** `external_repos/DAMOV-main/simulator/src/locality.{h,cpp}`
  (architecture-independent; lifted in Slice 3) — cite [Oliveira21].
- **NVBit:** `external_repos/nvbit_release_x86_64/` — cite [Villa19].
- **Workload launcher:** the stack's own `cuvslam_runner/` TOML package (Phase 0).

The prior scratch directories are left intact; this subsystem is the consolidated,
maintained home.
