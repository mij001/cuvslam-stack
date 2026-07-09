# profiling/sim/ — Phase 3 groundwork (NDP simulation)

Phase 3 evaluates the substrate design: the speedup/energy **delta** an NDP-GPU
would give per taxonomy class. This directory holds the part that is buildable
and testable **now** — turning the measured verdicts into concrete simulator
configs — so the remaining work is running a simulator, not writing plumbing.

## The pipeline (what feeds what)

```
substrate.py verdicts + hw descriptor + placement (k,c)
        │  gen_ndp_config.py   (here, runs now — no Accel-Sim needed)
        ▼
configs/<scenario>.ndp.config     Accel-Sim gpgpusim overlay (L2 /k, DRAM ×k, PiM c)
configs/<scenario>.manifest.csv   which kernels are offloaded + why
        │  profiling/blocked/run_accelsim.sh <traces>   (GATED — Phase 3)
        │      applies the calibrated sm_89 BASE + this overlay over the NVBit
        │      SASS traces; runs baseline and NDP; AccelWattch for per-component energy
        ▼
base-vs-NDP DELTAS: cycles + joules per G-class (never absolute simulated numbers)
```

## Run the groundwork (now)

```bash
python3 profiling/sim/gen_ndp_config.py \
    --hw profiling/hw/dellworkstation_sm89.toml \
    --verdicts reports/2026-07-07_substrate/substrate_verdicts.csv \
    --out profiling/sim/configs
```

Two scenarios, matching `pim_placement_model.py`'s analytical model so the sim
and the model are directly comparable:

| scenario | k (internal BW ×) | c (PiM compute) | offloads | affinity set |
|---|---|---|---|---|
| conservative | 4 | 0.50 | strong-affinity kernels | 16/49 |
| moderate | 8 | 0.75 | strong + conditional | 26/49 |

## What's gated (Phase 3, not now)

`run_accelsim.sh` and AccelWattch need an **Accel-Sim checkout** and a **base
sm_89 config calibrated to ~5%** against ncu microbenchmarks (no stock Ada
config ships; the hw descriptor's `[accelsim].base_config` currently points at
`SM75_RTX2060` as the adapt-from starting point). `check_capability.sh` gates
the trace side (driver ≤575 + NVBit). Until then, the overlays + manifests here
are the reviewable, version-controlled statement of exactly what Phase 3 will
simulate and why — derived from the measured evidence, not hand-picked.

The NDP config knobs (`-gpgpu_cache:dl2_ndp_bytes`, `-gpgpu_dram_ndp_bw_gbps`,
`-gpgpu_ndp_compute_ratio`) are the parameters Phase 3 sweeps; they are named so
the delta report can attribute a speedup to bandwidth vs cache-bypass vs
compute-ratio.
