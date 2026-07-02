# blocked/ — the gated Slice-3 DAMOV data-movement track

The NVBit → `locality.cpp` → Accel-Sim pipeline answers *"what is each data
structure's reuse/locality and PiM-affinity"* — the part of the characterization
that NCU counters cannot give (NCU aggregates; reuse distance needs the raw
address stream, and NCU's replay flushes caches so its hit rates are cold-start).

It is **gated**, not abandoned: NVBit's release caps the CUDA driver at ≤ 575.xx,
and driver-610 hosts (like the current dev laptop) cannot load it. Every script
here calls `check_capability.sh` first and fails fast with the reason and the
unblock instructions.

| Script | What it does when unblocked |
|---|---|
| `check_capability.sh` | driver ≤ 575? NVBit present? DAMOV sources present? |
| `run_nvbit_memtrace.sh <config>` | per-warp address stream of a runner workload → zstd |
| `run_accelsim.sh <traces>` | steady-state cache simulation (report **deltas**, never absolutes) |

Unblock paths (either):
1. run on a host whose driver is ≤ 575 (check with `nvidia-smi`), or
2. a newer NVBit release that supports the installed driver — update
   `NVBIT_MAX_DRIVER` in `check_capability.sh`.

Slice-3 work plan (PROFILING_PLAN.md §6): trace the steady-state and
loop-closure windows, lift DAMOV's architecture-independent `locality.cpp`
(per-warp granularity, coalescing, divergence), Sieve-sample representative
kernel invocations, simulate with a calibrated config, and produce
reuse-distance histograms + the stage → DAMOV-class → PiM/ISP-affinity table.
