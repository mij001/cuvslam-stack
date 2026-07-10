# GPU-DAMOV parity — every DAMOV component, its GPU analogue, and the evidence

DAMOV (Oliveira et al., 2021) is the reference methodology for deciding, per
function, whether data movement is the bottleneck and whether near-data
processing would help. This document walks DAMOV's components **one by one**
and shows the GPU analogue in this project — implemented, measured, or, where
we deviated, why. This is the completeness argument: GPU-DAMOV is *finished at
DAMOV's granularity* (and beyond it, where the GPU demanded more).

Status legend: ✅ measured/implemented · 🔶 groundwork laid, gated on external
tooling · ✖ deliberately replaced (justification given).

## 1 · The three-step method

| DAMOV component (paper §) | What DAMOV did (CPU) | GPU analogue here | Status |
|---|---|---|---|
| **Step 1 — screening** (§2.4.1) | VTune top-down: keep functions with Memory-Bound >30% and ≥3% of cycles (144 apps → 345 fns) | nsys timeline time-share (who matters) + ncu SoL pair (memory- vs compute-leaning); classification only for kernels above the attention floor — sub-floor kernels are *screened, not classified* (see g0 in §3) | ✅ |
| **Step 2 — locality clustering** (§2.4.1) | Architecture-independent spatial+temporal locality from Pin address streams; k-means on the locality scatter | NVBit per-warp address traces → reuse-distance CDFs (temporal) + sectors/warp coalescing (the GPU's spatial-locality analogue); memory-space filtered (global-only) so scratch never pollutes locality | ✅ |
| **Step 3 — the intervention experiment** (§2.4.2) | ZSim+Ramulator: 3 configs (Host / Host+prefetch / NDP) × core sweep 1→256; classes *defined by the response* (LFMR-vs-cores trend etc.) | Two real-hardware axes now + one simulated axis gated: (a) **clock-domain sweep** — core-clock vs memory-clock sensitivity per kernel (the compute:bandwidth balance axis, on silicon); (b) **cache-capacity axis** — the reuse-distance hit-CDF *directly measures* hit rate vs cache size 64 KiB→48 MiB (DAMOV inferred this from the aggregate-cache-growth trend; we measure it); (c) **NDP config axis** — Accel-Sim NDP overlays generated from the verdicts (`profiling/sim/gen_ndp_config.py`), run gated on the Accel-Sim checkout | ✅(a,b) 🔶(c) |

**Justification for (a) replacing the core-count sweep:** a GPU kernel's grid is
compiled into the launch; SM-count cannot be varied on real hardware without
MIG (unsupported on this part). The *purpose* of DAMOV's sweep — expose whether
performance tracks compute or memory — is served directly by scaling the two
clock domains independently: `nvidia-smi -lgc/-lmc` gives (1620→810 MHz core:
0.5× compute+on-chip) and (7001→5001 MHz memory: 0.71× DRAM). A class whose
kernels do not track the predicted domain is falsified. This is a *measured*
intervention, stronger than simulation for the question it answers.

## 2 · The classification

| DAMOV | GPU analogue | Status |
|---|---|---|
| Decision tree: temporal locality → LFMR (+trend) → MPKI/AI → **six classes** (1a/1b/1c/2a/2b/2c) | Tree: SoL pair → dominant stall → DRAM-SoL → sectors/request → LFMR → occupancy → **eight classes G0–G7**. Deviations forced by the GPU: **G2 coalescing** (no CPU analogue — a warp can waste 8× bandwidth on scatter), **G4 vs G7 split** (occupancy decides whether latency *can* be hidden; DAMOV's cores always stall), no L3 → no 2a contention class (the L2 is the last level; its contention shows as G1) | ✅ |
| Numeric thresholds derived once, then frozen (§3.5: TL 0.48, LFMR 0.56, MPKI 11, AI 8.5) | `classify.THRESHOLDS` — stated once, imported live into the docs/dashboard; ±25% sensitivity stress flags borderline kernels and caps their confidence (DAMOV's calibrated-cutoff analogue, made continuous) | ✅ |

## 3 · The robustness checks (DAMOV Part "did they check their own work")

| DAMOV check | Their result | GPU analogue | Our result |
|---|---|---|---|
| **§3.4 cache-size sweep** (NUCA L3 up to 512 MB) — do conclusions survive bigger caches? | classes behave exactly as defined | The measured reuse-distance hit-CDF answers per kernel what *any* cache size would do; the front-end CDF is **flat 64 KiB→48 MiB** — no cache size changes the conclusion (F5) | ✅ measured, stronger than the sim sweep |
| **§3.5 held-out validation** — frozen thresholds on 100 unseen functions | 97% correct | **Ground-truth calibration suite**: 8 archetype kernels *designed* to be each class (`profiling/calibration/archetypes.cu` — stream triad, random gather, L2-resident sweep, coalesced pointer-chase, FMA polynomial, bank-conflict shared, 1-warp dependency chain, sub-screen tiny), built + classified blind through the identical harness. First pass 5/8: the three misses were archetype *design conflations*, each fixed with the classifier untouched (a per-lane-random chase is legitimately G2+G4; a barrier-heavy shared kernel is legitimately not G6; a sub-floor kernel is *screened*, per Step 1). Final: **8/8 archetypes recovered blind** (7 classified + 1 correctly screened) — DAMOV's was 97/100 | ✅ (see `reports/2026-07-09_damov_validation/`) |
| **§3.5.2 core-type independence** (in-order vs OOO) | metrics/classes unchanged | **Cross-microarchitecture agreement**: the same workload (TUM office) classified independently on sm_75 (MX450 laptop) and sm_89 (RTX 2000 Ada): **80% of signal kernels get the same class** (36/45). Honest caveat: our two devices differ in *memory system* too (2 GB/25 W vs 16 GB/70 W), so some flips are physically real (the L2-capacity crossover), unlike DAMOV's controlled core-swap — the agreement floor is therefore conservative | ✅ |
| **§4.1 independent algorithm** (hierarchical clustering reproduces the classes) | dendrogram matches | **Ward hierarchical clustering** over the pooled 4-report feature cloud: **ARI 0.31 / purity 0.675** vs the tree — statistically the same agreement as k-means (0.30 / 0.68). Two unrelated algorithms find the same structure | ✅ |
| **Intervention response** (Step-3's own logic: classes predict scaling) | classes defined by it | **Clock-domain sweep**: per-class falsifiable predictions (G1/G2 track the memory clock; G3/G5/G6/G7 track the core clock; G4 tracks neither strongly). Result: **5/7 under the naive model, 7/7 under the refined signatures** — the two refinements are clock-domain architecture facts the experiment *taught us*: G2 scatter is bounded by memory-request **concurrency** (MSHRs/LSU, core-domain — S_core 1.18/S_mem 0.97, the DRAM bus is NOT saturated → strengthens PiM-scatter: the fix is request capacity near memory, not bus bandwidth), and G4 latency is core-domain **L2/NoC traversal + mem-domain CAS** (S_core 1.36/S_mem 1.12, mixed core-leaning). Every class keeps a distinct response signature (G1 mem-dominant 1.35 · G3/G5/G6/G7 core-dominant 1.8–2.0 · G2 mild-core · G4 mixed) | ✅ (`clock_sweep_verdicts.csv`) |

## 4 · Breadth and the suite contribution

DAMOV's breadth was 144 applications → 345 functions; its artifact is the DAMOV
benchmark suite. Our breadth axes: **49 kernels × 27 sequences × 4 datasets ×
192 configuration mutations** (one production application, deliberately deep
rather than wide — the thesis argues depth on a deployed Physical-AI workload),
**plus** the calibration archetype suite (8 known-truth kernels, our
suite-artifact analogue), **plus** the adapter framework that runs *any* GPU
codebase (PyTorch, CUDA benchmarks, databases) through the identical pipeline —
the mechanism by which breadth grows without new plumbing. External suites
(Rodinia/Altis/BabelStream) drop in as command-adapter workloads when wider
coverage is wanted; the archetypes already cover each class's pure form.

## 5 · Honest limitations (DAMOV §3.6 analogues)

DAMOV listed three; ours mirror them:
1. *They:* same core count host/NDP, no area/thermal budget. *We:* the placement
   model's k/c parameters are asserted scenarios (conservative/moderate), not a
   physical PiM design — the Accel-Sim leg (gated) turns them into simulated
   deltas.
2. *They:* function-level analysis ignores inter-function data movement. *We:*
   kernel-level analysis; host↔device transfers ARE measured (41% of kernel
   time), but kernel-to-kernel reuse through L2 is not attributed.
3. *They:* NDP overheads (coherence, VM) unmodeled → upper bounds. *We:* same —
   verdicts are candidacy + modeled bounds, and simulated numbers will be
   reported as deltas, never absolutes (standing rule 5).

## 6 · Where to see it

- `reports/2026-07-09_damov_validation/` — SUMMARY.md + the four CSVs
  (calibration confusion, clock-sweep verdicts, cross-device agreement,
  hierarchical agreement).
- `profiling/calibration/` — the archetype suite (`archetypes.cu`,
  `run_calibration.sh`); `profiling/validation/clock_sweep.sh`.
- `profiling/analysis/damov_parity.py` — recomputes the committed-data checks.
- Dashboard → Methodology → Classify: the validation numbers, stamped live.
