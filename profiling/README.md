# cuVSLAM Profiling Subsystem

Memory-profiling and characterization of cuVSLAM, layered on the Phase-0 TOML
runner. **Read `PROFILING_PLAN.md` for the strategy and the research/tooling
reality; `WALKTHROUGH.md` for the guided tour; `METHODOLOGY.md` for how every
number is produced (the paper's methodology source); `PUBLISHABILITY.md` for
the reviewer-grade open-issues register.** This file is the operational how-to.

The workload-under-test is always launched through the stack's own TOML runner
(`run.py <config>`), so any dataset the runner supports is profilable with no
new code — "TOML is the only input" carries over from Phase 0. Everything here
is **headless**: captures come from the CLI profilers, analysis is
stdlib-only Python, and figures are hand-emitted SVG (no GUI, no pip installs,
no matplotlib — clone and run on any Linux box with the NVIDIA tools).

---

## One command

```bash
# preflight → nsys baseline → nsys SLAM (loop closure) → steady-state ncu → report
profiling/run_characterization.sh                      # TUM office workloads, auto hw descriptor
profiling/run_characterization.sh --hw profiling/hw/rtx2000ada_sm89.toml \
    --warm 200 --launches 300                          # workstation
```

The report lands in `profiling/reports/<date>_<hw>/report.md` with SVG figures
and CSVs — committable as-is.

## New machine? Three steps

```bash
profiling/env/check_env.sh                       # preflight: driver, nsys/ncu, perms, venv, disk
profiling/env/fetch_datasets.sh tum_office       # headless dataset fetch (resumable)
python3 profiling/env/gen_hw_descriptor.py       # auto hw/*.toml for THIS GPU (exact structural values)
```

Datasets resolve through `${CUVSLAM_DATASETS}` (default
`~/Projects/cuvslam_datasets`) — configs never hard-code a machine path.
The generated descriptor's two *ceilings* (DRAM GB/s, FP32 TFLOP/s) are
estimates flagged `verify`: confirm with the Empirical Roofline Toolkit before
publishing numbers from that machine.

---

## Layout

| Path | What |
|---|---|
| `hw/*.toml` | **the one file that changes per GPU** — SM count, L2, DRAM BW, FP32 peak, clock policy, Accel-Sim hint. Auto-generate with `env/gen_hw_descriptor.py` |
| `env/` | `check_env.sh` (preflight) · `fetch_datasets.sh` · `gen_hw_descriptor.py` · `lock_clocks.sh` · `setup_perms.sh` |
| `configs/` | profiling workloads: `tum_office_profile` + `_slam_` (loop closure), `euroc_v101_*`, `kitti06_*` — all `${CUVSLAM_DATASETS}`-relative |
| `harness/profile.py` | one entrypoint: wrap a runner config under nsys/ncu into a versioned results dir |
| `analysis/` | stdlib-only, read-only consumers of `results/`: `build_dag` · `screen` · `roofline` · `bandwidth` · `classify` (GPU-DAMOV classes → PiM/ISP candidates) · `make_report` |
| `blocked/` | Slice-3 NVBit/locality/Accel-Sim — **driver-gated**, fails fast with the unblock instructions |
| `results/<date>_<seq>_<profiler>_<hw>/` | `metadata.json` (mandatory) + `raw/` + `derived/`; never overwritten; gitignored |
| `reports/` | committed characterization reports (markdown + SVG + CSV) |

---

## Individual steps

```bash
# 1) Nsight Systems timeline → the DAG + kernels/frame
python3 profiling/harness/profile.py --config profiling/configs/tum_office_profile.toml \
    --profiler nsys --hw profiling/hw/mx450_sm75.toml

# 2) Steady-state Nsight Compute: warm 200 frames, then profile 300 launches,
#    with the full characterization metric set (FLOPs, bytes, stalls, coalescing)
python3 profiling/harness/profile.py --config profiling/configs/tum_office_profile.toml \
    --profiler ncu --hw profiling/hw/mx450_sm75.toml \
    --metrics characterize --auto-window profiling/results/<nsys_run>:200:300

# 3) Loop-closure capture (the cold-persistent evidence)
python3 profiling/harness/profile.py --config profiling/configs/tum_office_slam_profile.toml \
    --profiler nsys --hw profiling/hw/mx450_sm75.toml

# 4) Analysis (any machine, no GPU needed — reads derived CSVs only)
python3 -m analysis.make_report --hw profiling/hw/mx450_sm75.toml \
    --nsys profiling/results/<nsys_run> --ncu profiling/results/<ncu_run> \
    --nsys-slam profiling/results/<nsys_slam_run>        # run from profiling/
```

Analysis modules also run standalone (each writes CSV + SVG into the run's
`derived/`): `python3 -m analysis.build_dag <nsys_run>`,
`python3 -m analysis.screen <ncu_run>`,
`python3 -m analysis.roofline <ncu_run> --hw …`,
`python3 -m analysis.bandwidth <ncu_run> --hw … [--nsys <nsys_run>]`,
`python3 -m analysis.classify <ncu_run|report_data_dir>… --hw …`.

**No dataset? No GPU? Still reproducible.** `classify` (and any module reading
CSVs) accepts a committed report's `data/` dir, so the DAMOV classification and
PiM/ISP candidate tables regenerate from the repo alone:

```bash
cd profiling && python3 -m analysis.classify \
    reports/2026-07-02_tum_office_mx450/data \
    reports/2026-07-02_tum_office_mx450/figures/slam \
    --hw hw/mx450_sm75.toml --out /tmp/cls
```

---

## Metric sets (`--metrics`)

Targeted sets, never `--set full` (which is killed on small GPUs before writing
a report — the original Slice-1 bug):

| Set | Metrics | Use |
|---|---|---|
| `quick` | 3 | smoke test (~1–2 replay passes) |
| `roofline` | 15 | SoL + hit rates + key stalls + DRAM traffic (~8 passes) |
| `characterize` | 29 | + FP32 FLOP counters (arithmetic intensity), L1/L2 bytes (hierarchical roofline), sectors/request (coalescing fingerprint), full stall taxonomy (~12 passes). **Use for report captures.** |

Curated from the Cao23 `gpudb-char-and-opt` taxonomy [Cao23]; validated against
ncu 2026.2.

---

## What works today vs. what's gated

| Tool | State | Notes |
|---|---|---|
| `nsys` | 🟢 | timeline, kernel order, launch counts (the DAG + auto-window source) |
| `ncu` | 🟢 | per-kernel SoL/roofline/stall/traffic with targeted sets |
| NVBit (`blocked/`) | 🔴 driver-gated | needs CUDA driver ≤ 575; `blocked/check_capability.sh` prints the unblock path |
| Accel-Sim (`blocked/`) | 🔴 gated | needs NVBit traces + a calibrated config; **report deltas, not absolutes** |

Caveats that carry into any writeup: ncu flushes caches between replay passes
(hit rates are cold-start, not steady-state); laptop GPUs can't lock clocks
(the hw descriptor records `enforce_clock_locks=false` and metadata captures
the actual clocks per run).

---

## Citations

Cao23 (NCU methodology + tooling), Oliveira21 (DAMOV), Villa19 (NVBit),
Khairy20 (Accel-Sim), Yang20 (hierarchical roofline), Naderan23 (Sieve),
Korovko25 (cuVSLAM). Full entries in the onboarding doc's bibliography under
`suggestions_and_summuries/`.
