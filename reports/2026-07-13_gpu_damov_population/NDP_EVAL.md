# NDP evaluator — measurement-fitted, holdout-validated

**Model:** per-kernel t = a/f_core + b/f_mem + c, fitted EXACTLY from the three primary clock points; validated on the held-out 4th point (1005/5001 MHz, off both fitted axes).
**Fits:** 296 kernels; median holdout error **1.0%**; trusted (err<=10%, physical): **258**; rejected: 2 holdout-fail, 36 non-physical, 11 missing points.

**NDP sweep (k∈{2,4,8,16}× internal BW, c_r∈{0.25..1} PiM compute, +2% offload overhead): gains AND losses, as DAMOV's Fig 18b requires:** at k=8/c_r=0.5, 49 kernels gain >1.1x and **203 kernels LOSE >10%** — the compute-/dependency-bound classes pay a/c_r, exactly the predicted failure mode.

| class | n | k=2 | k=4 | k=8 | k=16 (all c_r=0.5, median) |
|---|---|---|---|---|---|
| G0-nosignal | 3 | 0.53 | 0.53 | 0.53 | 0.53 |
| G1-bandwidth | 89 | 1.0 | 1.12 | 1.2 | 1.24 |
| G2-coalescing | 10 | 0.59 | 0.6 | 0.6 | 0.6 |
| G3-l2-reuse | 70 | 0.53 | 0.53 | 0.53 | 0.53 |
| G4-latency | 17 | 0.59 | 0.59 | 0.59 | 0.59 |
| G5-compute | 39 | 0.5 | 0.5 | 0.5 | 0.5 |
| G6-onchip | 8 | 0.54 | 0.55 | 0.55 | 0.55 |
| G7-dependency | 22 | 0.59 | 0.59 | 0.59 | 0.6 |

Cycle-accurate Accel-Sim (the literal ZSim+Ramulator analog) stays the gated Phase-4 instrument; its NDP overlays are already generated (profiling/sim/). This evaluator is silicon-grounded by construction.

-> ndp_model_fits.csv, ndp_speedups.csv, ndp_class_curves.csv
