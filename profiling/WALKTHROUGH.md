# cuVSLAM Profiling — Full Walkthrough

A step-by-step guide to the profiling phase: what it is, what's built, how to run
it, and what's left. Read top to bottom to get fully up to speed. For the deep
strategy see `PROFILING_PLAN.md`; for command reference see `README.md`.

---

## 0. Status at a glance

| Piece | State | Where |
|---|---|---|
| Researched plan + tooling reality | ✅ done | `PROFILING_PLAN.md` (commit `a27e96d`) |
| Hardware descriptors (MX450 / RTX 2000 Ada / Jetson) | ✅ done | `hw/*.toml` |
| Harness: `profile.py` (nsys + ncu wrappers) | ✅ done | `harness/profile.py` (commit `5d8b65e`) |
| ncu "no report" bug fixed (targeted metrics) | ✅ done | inline `METRIC_SETS` in `profile.py` |
| First real cuVSLAM `.ncu-rep` + `.nsys-rep` | ✅ verified | `results/` (local, gitignored) |
| Env scripts (clock lock, profiler perms) | ✅ done | `env/*.sh` |
| **Analysis layer** (DAG, roofline, screen, bandwidth) | ⏳ Slice 2 | `analysis/` (not created yet) |
| **Steady-state capture** (track 0→250, profile 200→250) | ⏳ Slice 2 | — |
| Vendored Cao23 NCU tooling (`counter_config.py`) | ⏳ Slice 2 | `ncu_tooling/` (not created yet) |
| **DAMOV / NVBit / Accel-Sim** data-movement track | 🔴 gated | `blocked/` (driver-blocked) |
| Jetson AGX Orin re-run | ⏳ later | `hw/jetson_orin_sm87.toml` exists |

We are **paused after Slice 1** for you to review. Next is Slice 2.

---

## 1. What we are doing, and why

**Goal:** a per-kernel *memory* characterization of cuVSLAM — for each CUDA kernel,
is it compute-bound or memory-bound, what is its access pattern, and is it a
candidate for memory-centric hardware (Processing-in-Memory / In-Storage-Processing)?

**Thesis to test** (from the onboarding doc, not assumed): cuVSLAM's data falls
into three *persistence classes* —

- **Streaming** — image data, touched once per frame then discarded → near-sensor SRAM.
- **Hot-persistent** — local map, touched every frame (~MBs) → LPDDR/HBM-PiM.
- **Cold-persistent** — keyframe database, touched only on loop closure (100s of MB) → ISP.

The deliverable is the *evidence* for/against that taxonomy: roofline placement,
access-pattern fingerprints, bandwidth breakdown, DAMOV-style classification.

This is "Phase 1+" — Phase 0 was the TOML runner that runs cuVSLAM and scores
trajectory accuracy. Profiling builds *on top of* that runner.

---

## 2. The two methodologies we are fusing

Both already have reference material and code on disk (`suggestions_and_summuries/`,
`external_repos/`).

1. **NCU roofline / stall** (the "gpudb" / Cao23 lineage). Use Nsight Compute to
   measure, per kernel: Speed-of-Light (Compute% vs Memory%), stall reasons,
   cache hit rates, arithmetic intensity → roofline. Answers *"is this kernel
   memory-bound, and why."* **This is unblocked and is our spine.**

2. **DAMOV data-movement** (CPU→GPU adaptation). Capture each kernel's memory
   address stream with NVBit, run it through DAMOV's architecture-independent
   `locality.cpp` for reuse/locality, simulate steady-state cache behaviour with
   Accel-Sim, and classify. Answers *"what is each data structure's locality /
   PiM-affinity."* **This is currently blocked (see §3) and deferred to Slice 3.**

---

## 3. The hardware + tooling reality (read this — it shaped everything)

The onboarding doc targets an **RTX 2000 Ada (sm_89)** workstation. The current
dev box is a laptop **MX450 (sm_75 Turing, 14 SM, 512 KiB L2, 64-bit, 2 GB,
no ECC)**, driver **610.43.02**, CUDA **13.2**.

What we tested, empirically:

| Tool | Works here? | Why it matters |
|---|---|---|
| cuVSLAM run | 🟢 yes | the runner is the workload-under-test |
| **Nsight Systems** | 🟢 yes | timeline → the DAG |
| **Nsight Compute** | 🟢 yes (with targeted metrics) | per-kernel roofline/stalls |
| **NVBit** | 🔴 **no** | its release caps **CUDA driver ≤ 575**; this box is **610** |
| **Accel-Sim** | 🔴 no | needs NVBit traces |

**Consequence:** the DAMOV headline (reuse-distance, Accel-Sim deltas) rides on a
toolchain that won't load on driver 610. So we made **NCU/Nsys the spine now** and
**gated the NVBit/Accel-Sim track** behind "a CUDA driver ≤ 575 (e.g. on the
workstation) or a newer NVBit." This is an evidence-based deviation from the doc's
phase order, not a reduction in ambition — every DAMOV step stays wired.

Everything is **hardware-parameterized**: the only per-GPU differences live in one
`hw/*.toml`, so this develops on the MX450 and runs for real on the RTX 2000 Ada
with just a `--hw` flag.

---

## 4. How it fits together (the architecture)

```
   configs/<workload>.toml ──► run.py (Phase-0 TOML runner) ──► cuVSLAM (the work)
                                      ▲
                                      │ launched under a profiler by
                                      │
   profiling/harness/profile.py ──────┘   --profiler {nsys,ncu}   --hw hw/<gpu>.toml
                                      │
                                      ▼
   profiling/results/<date>_<seq>_<profiler>_<hw>/
        ├── metadata.json   (GPU, driver, CUDA, versions, exact command — provenance)
        ├── raw/            (kernels.ncu-rep  OR  profile.nsys-rep)
        └── derived/        (ncu_metrics.csv  OR  nsys_stats.txt + kern_sum CSV)
```

Key idea: **the existing TOML runner is the workload.** `profile.py` just wraps
`run.py <config>` under a profiler. "TOML is the only input" carries over from
Phase 0; any dataset the runner supports is profilable with no new code.

---

## 5. What's implemented (step by step)

### 5.1 Plan + hardware descriptors — commit `a27e96d`
- `PROFILING_PLAN.md` — strategy, the GREEN/AMBER/RED tooling reality, the 3-slice
  plan, the deviation rationale, milestones, provenance.
- `README.md` — operational how-to.
- `hw/mx450_sm75.toml` — this laptop (values measured from the CUDA runtime).
- `hw/rtx2000ada_sm89.toml` — workstation; the **real-results** target.
- `hw/jetson_orin_sm87.toml` — Phase 3.5 unified-memory target.
- `.gitignore` — keeps large reports/clones out of git; vendors the guiding `.md`s.

### 5.2 The harness — commit `5d8b65e`
- `harness/profile.py` (stdlib only, ~250 lines). One entrypoint:
  1. reads the `--hw` descriptor (provenance + clock policy),
  2. optionally overrides the frame window (`--frames START:COUNT`) into a recorded
     copy of the config (`derived/used_config.toml`),
  3. launches `run.py <config>` under nsys or ncu,
  4. writes the versioned results dir + `metadata.json`,
  5. post-processes: ncu → `ncu_metrics.csv`; nsys → `nsys_stats.txt` + kernel CSV.
- `configs/kitti06_profile.toml` — a bounded KITTI seq06 workload (its own correct
  1226×370 intrinsics; `max_frames` window; no `[eval]` — profiling doesn't need
  accuracy scoring).
- `env/lock_clocks.sh` — pins GPU/CPU clocks from a `--hw` descriptor (no-op on
  laptops, which can't lock).
- `env/setup_perms.sh` — grants the ncu profiler permission (`NVreg`); one-time, sudo.

### 5.3 The ncu fix (the key unblock)
The prior prototype called `ncu --set full` → on a 2 GB GPU it is killed before
writing, leaving a results dir with **only `metadata.json`** (no report). The fix:
a **targeted metric set** — a ~15-counter roofline/SoL/stall/memory list curated
from the Cao23 taxonomy, validated against ncu 2026.2. It collects in **~8 replay
passes** instead of dozens, and actually finishes. (The list lives inline in
`profile.py` as `METRIC_SETS["roofline"]`; vendoring Cao's full `counter_config.py`
into `ncu_tooling/` is a Slice-2 tidy-up.)

### 5.4 Verified results (MX450, KITTI seq06)
**Nsight Compute** — first real cuVSLAM per-kernel numbers, with the DAMOV-GPU
"Step-1 screen" (Memory% ≫ Compute% ⇒ memory-bound) already separating stages:

| kernel (stage) | Mem SoL% | Compute SoL% | L1 hit% | verdict |
|---|---:|---:|---:|---|
| `cast_image_kernel` (preprocess) | 82 | 8 | 3 | memory-bound, streaming |
| `conv_grad_x_kernel` (feature detect) | 85 | 13 | 54 | memory-bound |
| `gaussian_scaling_kernel` (pyramid) | 75 | 46 | 91 | mixed, high reuse |
| `DeviceMergeSortMerge` (keypoint sort) | 37 | 8 | 67 | not memory-bound |

**Nsight Systems** — the time-weighted DAG (top GPU kernels by total time):
`sba::build_full_system_1` 16% (bundle adjustment), `lk_track` 12% (optical-flow
tracking), `conv_grad_x` 9% (feature detection), `gaussian_scaling` 8% (pyramid).

These map cleanly to the canonical V-SLAM pipeline (preprocess → detect → track →
bundle-adjust), which is exactly what the Slice-2 DAG formalizes.

---

## 6. How to run it (step by step)

```bash
cd ~/Projects/cuvslam-stack

# (one-time, optional) profiler permissions for ncu — needs sudo + reboot.
#   sudo profiling/env/setup_perms.sh
# Already in effect on this box: grep RmProfilingAdminOnly /proc/driver/nvidia/params  # -> 0

# 1) Per-kernel roofline / SoL / stalls with Nsight Compute (targeted metrics)
./cuvslam_venv/bin/python profiling/harness/profile.py \
    --config profiling/configs/kitti06_profile.toml \
    --profiler ncu --hw profiling/hw/mx450_sm75.toml --metrics roofline

# 2) Timeline / DAG with Nsight Systems
./cuvslam_venv/bin/python profiling/harness/profile.py \
    --config profiling/configs/kitti06_profile.toml \
    --profiler nsys --hw profiling/hw/mx450_sm75.toml

# Output lands in profiling/results/<timestamp>_kitti06_<profiler>_mx450_sm75/
#   raw/kernels.ncu-rep      -> open in the Nsight Compute GUI
#   raw/profile.nsys-rep     -> open in the Nsight Systems GUI
#   derived/ncu_metrics.csv  -> parsed per-kernel metric table (308 columns)
#   derived/kern_sum_*.csv   -> time-weighted kernel list (the DAG seed)
```

Useful flags: `--metrics quick` (3-metric fast smoke), `--frames 0:250`
(override the window), `--launch-skip/--launch-count` (which kernel launches ncu
profiles), `--tag <name>` (label the run dir).

---

## 7. What's NOT yet implemented (step by step)

### Slice 2 — Characterization (the next thing to build)
The analysis layer that turns the raw captures into the report. None of this exists
yet:
1. **Steady-state capture.** Track frames 0→250 to warm the local map, then scope
   ncu to the 200→250 window. ncu bounds by *kernel-launch index*, not frame, so
   we derive `--launch-skip/--launch-count` from kernels-per-frame (≈ total
   instances ÷ frames, which the nsys capture already gives). Replaces the current
   cold-start bounded window. *(You chose this fidelity.)*
2. `analysis/build_dag.py` — parse the nsys kernel summary into a kernel→stage DAG,
   tag each stage with a persistence-class hypothesis.
3. `analysis/roofline.py` — hierarchical roofline (L1/L2/DRAM ceilings from `hw/`),
   place each kernel, compute arithmetic intensity.
4. `analysis/stall_breakdown.py` — per-kernel stall pies; the memory-bound screen.
5. `analysis/bandwidth.py` — per-stage DRAM bandwidth bars (the slide-2 chart).
6. `ncu_tooling/` — vendor Cao23 `counter_config.py` + parser (replace the inline
   metric list); cite [Cao23].
7. A committed **characterization report** (tables + PNG figures): DAG, roofline,
   DAMOV Step-1 screen, bandwidth breakdown.

### Slice 3 — DAMOV / data-movement track (🔴 gated on the driver)
Wired behind a capability check that fails fast with the reason. Lights up on a
driver ≤ 575 host (the workstation) or a newer NVBit:
1. `blocked/run_nvbit_memtrace.sh` — NVBit `mem_trace` for per-warp address streams.
2. `locality/` — DAMOV `locality.cpp` lifted out (GPU-adapted: per-warp granularity,
   coalescing efficiency, divergence).
3. `blocked/run_accelsim.sh` — Accel-Sim steady-state cache hit-rate **deltas**
   (report deltas, not absolutes), calibrated to NCU.
4. Reuse-distance histograms + the synthesis table: *stage → DAMOV class → PiM/ISP
   affinity*.

### Later — Jetson AGX Orin (Phase 3.5)
Re-run the whole pipeline with `--hw jetson_orin_sm87.toml`; add unified-memory
analysis. The descriptor exists; the rest ports without rework.

---

## 8. Directory map (exists vs planned)

```
profiling/
├── PROFILING_PLAN.md          ✅ strategy
├── README.md                  ✅ command reference
├── WALKTHROUGH.md             ✅ this file
├── hw/                        ✅ mx450_sm75 · rtx2000ada_sm89 · jetson_orin_sm87
├── env/                       ✅ lock_clocks.sh · setup_perms.sh
├── harness/profile.py         ✅ the entrypoint
├── configs/kitti06_profile.toml ✅ the workload
├── results/                   ✅ created at runtime (gitignored)
├── ncu_tooling/               ⏳ Slice 2 (vendor Cao23 counter_config.py)
├── analysis/                  ⏳ Slice 2 (build_dag · roofline · stall · bandwidth)
└── blocked/                   🔴 Slice 3 (nvbit · accelsim · locality)
```

---

## 9. Decisions on record (the forks and what we chose)

1. **Hardware:** hardware-parameterized — prototype on MX450, real runs on RTX 2000 Ada.
2. **Scope:** the MD docs are *suggestions*; research decided **NCU-first, NVBit-gated**.
3. **Instrumentation:** start **without** NVTX/source edits (decide later, once the
   DAG shows where labels are needed).
4. **Process:** write the plan doc first, then build.
5. **Consolidation source:** the prior prototype in `~/Projects/cuvslam profiling/`.
6. **Slice 2 capture:** **proper steady-state** (track 0→250, profile 200→250).

---

## 10. Where we paused + the exact next step

**Paused after Slice 1** (you asked to review first). Everything in §5 is committed;
the example reports in §5.4 are on disk under `profiling/results/` (gitignored).

**Next step when you say go:** build the Slice-2 `analysis/` layer on a
**steady-state** capture — DAG, hierarchical roofline, the memory-bound screen, and
the per-stage bandwidth breakdown — and commit the characterization report.

---

## 11. Key facts worth remembering

- **NVBit is blocked here** by driver 610 > its 575 cap. NCU/Nsys are not.
- **`ncu --set full` fails on 2 GB GPUs** — always use a targeted metric set.
- **This MX450 has only 512 KiB L2** (measured) — tiny, which makes memory
  bottlenecks *more* visible. Reuse-distance cutoff for "spills to DRAM" is ~512 KiB
  here vs 12 MB on the Ada. Cite the hardware, not generic numbers.
- **The runner is the workload** — profile any dataset by pointing `profile.py` at
  its TOML config.
- **Report deltas, not absolutes**, for any simulated (Accel-Sim) number.
- All commits are authored `mij001`; no AI-attribution trailers.
