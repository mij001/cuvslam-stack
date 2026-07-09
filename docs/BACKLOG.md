# Backlog & suggestions register — one-by-one triage

Every actionable suggestion from `suggestions_and_summuries/` (the 6 method
docs), `profiling/PUBLISHABILITY.md`, and `docs/THESIS_FINDINGS.md` §4–5 (gaps
G1–G10, paths A–C), triaged against what the code actually does. Each row was
**verified in the source** — where the original inventory over-claimed, the
correction is noted.

**Legend** — `DONE` implemented (file cited) · `BUILD` doing it this cycle ·
`READY` supported, needs a run/device (no code) · `DEFER` real, scheduled later ·
`DROP` superseded / not-applicable, with why.

This file is the source for the dashboard's Methodology → Roadmap strip.

---

## Built or building now

| # | Suggestion (source) | Decision | Where |
|---|---|---|---|
| 1 | Detailed, expandable methodology in the UI (this request) | **DONE** | `viz/gen_methodology.py` → `reports/methodology.json`; `dashboard/app.js` accordion; decision tree shared with `explorer.js` |
| 2 | Whole-run energy / joules — "PiM's main win is energy; you never measure a joule" (PUB open-6, THESIS G2) | **DONE** — 34.67 J measured on the RTX 2000 Ada | `profiling/env/energy.py` power sampling → `metadata.json`/`summary.json` → ENERGY finding + explorer field |
| 2b | Multi-op-type AI numerator (fp16 / int / fp64), for the "profile any GPU codebase" adapter story (Doc2-5/6, Doc6-4) | **DONE** — `auto` picks the dominant op-type per workload (unit-tested); cuVSLAM stays fp32 | `roofline.py OPTYPE_FLOPS` + `profile.py` characterize set + `--ai-optype` |

## Done (surfaced in the Methodology tab)

| # | Suggestion (source) | Reality — verified | Where |
|---|---|---|---|
| 3 | LFMR_gpu = L2miss/L1miss for the 2-level hierarchy (Doc1-3, Doc4-4) | DONE | `screen.py:88` (`1 − l2_hit%`) |
| 4 | MPKI, DRAM/Mem/Comp-SoL, sectors/request, occupancy, stall taxonomy from ncu (Doc1-2, Doc5-2/3) | DONE | `screen.py` `M`/`STALLS` |
| 5 | Roofline; AI = fadd+fmul+2·ffma / DRAM bytes (Doc2-4, Doc5-4) | DONE | `roofline.py` |
| 6 | Two rooflines (DRAM **and L2**); AI differs per level (Doc2-4, Doc5-4) | DONE — `ai_l2` present (inventory was right) | `roofline.py:58` |
| 7 | Sum extensive quantities, divide once — never average ratios (Doc2-3, Doc5-5); instruction-weight stalls (Doc5-6) | DONE | `screen.py` aggregation, `summarize_run.py` |
| 8 | Curated ncu metric set to avoid replay explosion; collect broad, compute narrow (Doc2-2, Doc5-2/3) | DONE — as `METRIC_SETS` (roofline/quick/characterize), *not* a separate `counter_config.py`; equivalent, no action | `profile.py METRIC_SETS` |
| 9 | Discover the taxonomy by clustering, not assertion; k-means validates the tree (Doc1-7, Doc4-6, PUB-7) | DONE — pooled k-means, best k=7–8, purity 0.68 | `campaign.py`, `reports/2026-07-04_campaign/` |
| 10 | Threshold sensitivity — thresholds are calibrated, stress-tested (Doc4-1, PUB-2) | DONE — ±25% `sensitivity()`, borderline caps confidence | `classify.py:159` |
| 11 | Cold/warm cache bracket; run-to-run variance / CoV (Doc2-1, Doc6-1, PUB-3/4) | DONE — `--cache-control`, ×5/×3 repeats, CoV 0.14% | `profile.py`, `variance.py`, PUB ✅1–4 |
| 12 | Locality from an architecture-independent address stream; drop non-global spaces (Doc3-2, Doc4-2, PUB-NEW2) | DONE — Fenwick reuse-distance + hit-CDF; `--spaces global` default | `locality.py` |
| 13 | 3-layer address→data-structure attribution + space-aware join (Doc3-6/7, PUB-3) | DONE — layers 1–2 built; join is streaming/O(1) | `attribution.py`, patch 0002, `blocked/*alloc*.patch` |
| 14 | NVTX kernel→stage table (measured, not name-guessed) (Doc3-5/8, PUB-8) | DONE | patch 0002 `USE_NVTX`, `nvtx_kern_sum.csv` |
| 15 | `-lineinfo`/`-g` for source correlation; lock clocks + CPU governor (Doc3-1/3, Doc5-1) | DONE | build flags; `env/lock_clocks`/`gen_hw_descriptor` |
| 16 | Full campaign with coverage audit / gap-fill, 0 missing kernels (Doc3-9) | DONE — 27 seq, 0 gaps | `campaign/plan_gapfill.py`, `attribution_campaign.py` |
| 17 | Accuracy/QoR validation — the profiled build tracks correctly (Doc3-10, F13) | DONE — 141 configs, profiling accuracy-neutral | `validation_regime.sh`, `accuracy_report.py` |
| 18 | Host↔device transfer accounting (inter-kernel movement, GPU-DAMOV §9) (PUB-11) | DONE (host↔device side) — copies = 41% of kernel time | `transfers.py` |
| 19 | LICENSE for artifact evaluation (PUB open-10) | DONE — MIT added (register entry stale) | `LICENSE` |

## Deferred — real, scheduled

| # | Suggestion (source) | Decision + why | Effort |
|---|---|---|---|
| 20 | Host-side **I/O + memory** characterization (THESIS G3 / Path C1) | **DONE — done RIGHT**: `profiling/env/host_io.py` samples the whole process tree's `/proc` — `read_bytes` (storage reads), **`majflt`×page (the mmap page-in traffic `/proc-io` alone misses)**, `rchar/wchar`, and `VmHWM` (peak host RSS). Measured on icl: **66 MB storage read** (the dataset/sensor ingestion feeding the H2D upload, F4) + **708 MB peak host RSS** (where the keyframe DB lives, while the GPU allocation is static per F8). The characterization now has a host dimension | done |
| 21 | Layer-3 **kernel-arg correlation** — name the ~40% static-memory residuals (`__device__` globals, texture reads) (THESIS G5 / Path C3) | **DEFER** — needs NVBit kernel-argument capture (new instrumentation), not just analysis; incremental value (last 40% of a few kernels) | 3–5 d |
| 22 | **Occupancy sweep** — vary occupancy/CTA count to see latency-hiding reliance (Doc1-5) | **DROP → replaced** — the *question* ("does this kernel lean on occupancy to hide latency") is already answered per-kernel by the single-point occupancy% + the G4/G7 classification and stall taxonomy. A real sweep can't be done via config (cuVSLAM's launch bounds are compiled in); it needs source recompiles — Phase-4 sim territory. No separate deliverable |
| 23 | Multi-op-type **AI numerator** (INT / FP16 / fp64) for non-FP32 workloads (Doc2-5/6, Doc6-4) | **DONE this cycle** — see row 2b. (Tensor-core FLOPs deliberately excluded — need arch-specific ops-path counters) | — |
| 24 | **Accel-Sim NDP configs + AccelWattch** energy — substrate-side speedup/energy *deltas* (Doc1-1/6/8, PUB open-5, THESIS G1/Path B) | **DEFER (Phase 4)** — the architecture-paper (MICRO/ASPLOS) scope; the characterization paper stands without it. Large | weeks |

## Ready — supported, needs a run or device (no code)

| # | Suggestion (source) | Decision |
|---|---|---|
| 25 | **Jetson Orin re-run** to validate the codegen-dependent spill/shared/global split off sm_89 (THESIS G4 / Path C2) | **READY** — hw descriptor exists (`hw/jetson_orin_sm87.toml`) and the app already targets Orin over ssh; run it when the device is on the bench |
| 26 | Write the **characterization paper** (ISPASS/IISWC) — the evidence is complete (THESIS Path A) | **READY** — a writing task, not a code task |

## Dropped — superseded or not applicable

| # | Suggestion (source) | Decision + why |
|---|---|---|
| 27 | Accel-Sim tracer as the **primary** instrumentation, replacing ncu/nsys (Doc1-1/8) | **DROP** — we deliberately use native ncu+nsys+NVBit (driver-matched, on real hardware); Accel-Sim is Phase-4 *simulation*, not the measurement front-end |
| 28 | DAMOV CPU pipeline: **VTune top-down, ZSim + Ramulator, core-count sweeps** (Doc4-1/3/7) | **DROP** — this is the CPU origin GPU-DAMOV was adapted *from*; the GPU analogs (ncu SoL, occupancy sweep, the taxonomy) are the port |
| 29 | **MIG/MPS** concurrency trade-off measurement (Doc5-9, Doc2-7) | **DROP for cuVSLAM** — verified NOT present (only `--target-processes all` for multi-process attach); cuVSLAM is single-tenant, so MIG/MPS adds nothing. Revisit only if multi-tenant serving is ever in scope |
| 30 | Add a **second SLAM system** (ORB-SLAM3) for representativeness (THESIS G8) | **DROP** — THESIS explicitly recommends against (scope creep, weak marginal evidence); handle "is cuVSLAM representative of Physical AI" in related-work prose |

---

**Summary (updated — acted on the backlog):** 17 previously done + **3 done this
cycle** (methodology tab, energy, op-type AI) · **3 deferred with sharpened,
honest rationale** (LMDB/ISP needs cold-cache+majflt not naive /proc-io; Layer-3
needs new NVBit capture; Accel-Sim is Phase-4) · **1 ready** (Orin — needs the
device) + paper (writing) · **5 dropped** (Accel-Sim-as-primary, DAMOV-CPU
pipeline, MIG/MPS, second-SLAM, **occupancy-sweep** — replaced by the existing
single-point occupancy + G4/G7 classification). Nothing open is a silent gap;
each deferral now states *why the fast version would be wrong*.
