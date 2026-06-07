# cuVSLAM Profiling Subsystem

Memory-profiling and characterization of cuVSLAM, layered on the Phase-0 TOML
runner. **Read `PROFILING_PLAN.md` for the strategy and the
research/tooling reality.** This file is the operational how-to.

The workload-under-test is always launched through the stack's own TOML runner
(`run.py <config>`), so any dataset the runner supports is profilable with no new
code — "TOML is the only input" carries over from Phase 0.

---

## Layout

| Path | What |
|---|---|
| `hw/*.toml` | **the one file that changes per GPU** — SM count, L2, DRAM BW, FP32 peak, clock policy, Accel-Sim hint |
| `env/` | `lock_clocks.sh`, `setup_perms.sh`, frozen requirements, captured `system_info` |
| `harness/profile.py` | one entrypoint: wrap a runner config under a profiler, into a versioned results dir |
| `harness/run_nsys.sh`, `harness/run_ncu.sh` | thin profiler wrappers (also usable directly) |
| `ncu_tooling/` | vendored Cao23 `counter_config.py` + parser; **targeted** metric sets, not `--set full` |
| `analysis/` | `build_dag.py`, `roofline.py`, `stall_breakdown.py`, `bandwidth.py` — read-only consumers of `results/` |
| `blocked/` | NVBit / Accel-Sim / locality — driver-gated, fail fast with the reason (see Plan §3) |
| `results/<date>_<seq>_<profiler>_<hw>/` | `metadata.json` (mandatory) + `raw/` + `derived/`; never overwritten |

---

## Quick start

```bash
# Nsight Systems (timeline → DAG).  Works on the current driver.
python profiling/harness/profile.py \
    --config configs/euroc_mh01.toml \
    --profiler nsys \
    --hw profiling/hw/mx450_sm75.toml \
    --frames 200:250

# Nsight Compute (per-kernel roofline/stall).  TARGETED metric set, not --set full.
python profiling/harness/profile.py \
    --config configs/euroc_mh01.toml \
    --profiler ncu \
    --hw profiling/hw/mx450_sm75.toml \
    --metrics roofline \
    --launch-skip 100 --launch-count 20
```

Each invocation writes `profiling/results/<timestamp>_<seq>_<profiler>_<hw>/` with
`metadata.json`, the raw report under `raw/`, and parsed CSVs under `derived/`.

---

## The Nsight Compute fix (why prior runs produced nothing)

The earlier prototype called `ncu --set full --launch-skip 100 --launch-count 50`.
On a 2 GB MX450, `--set full` collects the maximum metric set with many replay
passes per kernel; over cuVSLAM's kernel stream it is killed (time/memory) before
`ncu` writes the `.ncu-rep`, so the result dir held only `metadata.json`.

The fix, used here:

- **Targeted metric sets** from `ncu_tooling/utility/counter_config.py` (e.g.
  `roofline`, `sol`, `stall`, `memory`) instead of `--set full`. Far fewer passes.
- A **smaller launch window** (`--launch-count 10–20`) — cuVSLAM repeats the same
  kernels, so a handful of representatives suffice.
- Confirm `-o` points inside `results/.../raw/` and the report is non-empty before
  declaring success.

`ncu` itself is healthy on this box: a smoke kernel profiles fine and
`RmProfilingAdminOnly` is `0` (profiler permissions already granted). If you ever
see `ERR_NVGPUCTRPERM`, run `env/setup_perms.sh` (needs sudo + reboot).

---

## Hardware descriptors (`hw/`)

A descriptor records the constants the analysis needs (roofline ceilings, cache
sizes) and the run policy (clock locking). Pick one with `--hw`. Values flagged
`verify` should be confirmed with `deviceQuery` / ERT before they appear in any
published number.

- `mx450_sm75.toml` — this dev laptop. Small caches/BW make memory bottlenecks
  *more* visible (good for finding PiM-favourable behaviour); 2 GB caps input size.
- `rtx2000ada_sm89.toml` — the onboarding doc's workstation GPU. The real-results
  target. (Its driver is also where the NVBit/Accel-Sim track can unblock.)
- `jetson_orin_sm87.toml` — Phase 3.5 deployment target (unified memory).

---

## What works today vs. what's gated

| Profiler | State | Notes |
|---|---|---|
| `nsys` | 🟢 use freely | timeline, kernel order, NVTX (when present), H2D/D2H |
| `ncu` | 🟢 use with targeted metrics | per-kernel SoL, roofline, stalls, memory workload |
| `nvbit` (`blocked/`) | 🔴 driver-gated | needs CUDA driver ≤ 575; this box is 610. Fails fast with the reason. |
| `accelsim` (`blocked/`) | 🔴 gated | needs NVBit traces + a validated sim config |

See `PROFILING_PLAN.md` §3 for the evidence and the unblock conditions.

---

## Citations

Cao23 (NCU methodology + tooling), Oliveira21 (DAMOV), Villa19 (NVBit),
Khairy20 (Accel-Sim), Korovko25 (cuVSLAM). Full entries in the onboarding doc's
bibliography under `suggestions_and_summuries/`.
