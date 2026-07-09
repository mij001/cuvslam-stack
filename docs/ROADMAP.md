# Roadmap

The phased plan from "cuVSLAM is characterized" to "the heterogeneous substrate
is designed and evaluated". Complements `docs/BACKLOG.md` (the item-by-item
triage) — this is the *sequence* and the *groundwork now in place*.

Three phases, gated by evidence, not by wishlist:

```
Phase 1  CHARACTERIZE (done)     what the workload does to memory, per kernel,
                                 with a substrate verdict + the on-GPU fault
Phase 2  QUANTIFY + GROUND       close the measurement gaps that sharpen the
   (now) claim; lay the sim groundwork so Phase 3 is config-not-plumbing
Phase 3  DESIGN + EVALUATE       Accel-Sim NDP configs + AccelWattch energy —
 (later) speedup/energy DELTAS per taxonomy class (the architecture paper)
```

---

## Built — Phase 1 & the Phase-2 measurement gaps (done)

Everything the characterization paper (ISPASS/IISWC) needs is measured and in
the app. Highlights, newest first:

| capability | evidence | where |
|---|---|---|
| **Host-side I/O + memory** (closes THESIS G3) | icl: 66 MB storage read (sensor ingestion → the H2D upload, F4) + 708 MB peak host RSS (keyframe DB, GPU alloc static per F8); mmap page-ins via `majflt` that `/proc-io` alone misses | `profiling/env/host_io.py` → summary + finding |
| **Whole-run energy** (closes PUB open-6 / G2) | icl 34.67 J (12.54 W mean, 25.08 W peak), RTX 2000 Ada | `profiling/env/energy.py` |
| **Op-type AI** (generalization: any GPU codebase) | `auto` picks the dominant op-type (fp32/fp16/fp64/int); cuVSLAM stays fp32, a DNN gets fp16 with zero config | `roofline.py OPTYPE_FLOPS` |
| **Deep methodology, stamped from source** | decision tree + every formula/counter + NVTX/NVBit mechanics + placement & fault mapping | `viz/gen_methodology.py`, Methodology tab |
| **Substrate candidacy + dynamics** | 72% of GPU time offload-eligible; 25/49 verdicts flip across workloads | `analysis/substrate.py` |
| **Attribution (data structures)** | 48/49 kernels unanimous tag; st_track 92.6% spill | `analysis/attribution.py` |
| **Taxonomy discovered + validated** | tree G0–G7, k-means best k=7–8, purity 0.68 | `analysis/classify.py`, `campaign.py` |
| **Locality / reuse** | front-end reuse-CDF flat 64 KiB→48 MiB (cache-immune) | `analysis/locality.py` |
| **Accuracy-neutral instrumentation** | 141-run matrix; nsys/ncu bit-identical, NVBit ≤2 mm | `validation_regime.sh` |
| **Locked-clock rigor** | CoV 0.14%; measured ceilings 205 GB/s, 5445 GF/s | `env/lock_clocks`, PUB ✅1–4 |

## Phase 2 — grounding now in place (this cycle)

Two groundwork pieces so Phase 3 is configuration, not new plumbing:

| piece | what it does now | de-risks |
|---|---|---|
| **Layer-3 target quantifier** — `analysis/residuals.py` | ranks kernels by their **unmapped + untagged-driver** DRAM-traffic share from the committed attribution data → the exact list Layer-3 kernel-arg correlation must resolve, with the traffic at stake | THESIS G5 / #21 — turns "~40% unmapped somewhere" into a named, prioritized target list |
| **NDP config generator** — `sim/gen_ndp_config.py` | reads the hw descriptor + `substrate_verdicts.csv` + the placement model's (k, c) scenarios → emits Accel-Sim **NDP config skeletons** (reduced L2, near-bank BW ×k) + a manifest of *which kernels* the NDP models and *why* | THESIS G1 / Path B / #24 — the config-generation step that needs no Accel-Sim checkout is done and tested |

## Deferred — real, scheduled (Phase 2 finish → Phase 3)

Ordered by value-per-effort; each names its dependency and the groundwork it
builds on.

| # | item | plan | needs | effort |
|---|---|---|---|---|
| 1 | **Layer-3 kernel-arg correlation** (G5) | build the NVBit kernel-argument capture tool (pointer args per launch) → join to the LiveSet (reuse `attribution.py`) → resolve the residuals `residuals.py` already named | an NVBit tool build (podman); the analysis join interface is pinned in `residuals.py`'s docstring | 3–5 d |
| 2 | **Jetson Orin re-run** (G4) | run the existing regime on Orin (the app already targets it over ssh) → re-derive the spill/shared/global split off sm_89 → confirm codegen-independence | the physical device; `hw/jetson_orin_sm87.toml` exists | 1–2 d on-device |
| 3 | **Accel-Sim NDP + AccelWattch** (G1, Path B) | feed the NVBit SASS traces to Accel-Sim; run the `gen_ndp_config.py` NDP configs vs baseline → speedup deltas; AccelWattch per-component energy → energy deltas per G-class | Accel-Sim + AccelWattch checkout (gated); sm_89 base config calibrated to ~5% vs ncu microbenchmarks; the config generator + trace tooling (`mem_trace.cu`) are ready | weeks |
| 4 | **Characterization paper** (Path A) | write it — the evidence is complete | — (writing) | — |

**Standing rule for Phase 3:** simulated numbers are reported as *deltas* vs the
measured GPU baseline, never as absolutes (PUB standing rule 5). The measured
energy (this cycle) is that baseline.

## Dropped (see BACKLOG for rationale)

Accel-Sim-as-primary-instrumentation, the DAMOV CPU pipeline (ZSim/Ramulator/
VTune), MIG/MPS concurrency, a second SLAM system, and the config-level
occupancy sweep (replaced — the question is already answered by single-point
occupancy + the G4/G7 classification).
