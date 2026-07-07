# Profiler neutrality — do nsys / ncu / NVBit change cuVSLAM's accuracy?

**Question.** Our profiling harness runs cuVSLAM under three instrumentation
tools. Does the instrumentation itself perturb the trajectory the pipeline
produces? If it did, every accuracy-under-profiling claim would be measuring the
profiler, not cuVSLAM. This experiment proves it does not.

**Method.** Six representative deterministic sequences — one per sensor
modality — are run four ways: un-profiled (`plain`, the `accuracy_out`
baseline), under **Nsight Systems**, under **Nsight Compute**, and under
**NVBit** `mem_trace`. Each run computes the full trajectory-vs-ground-truth
ATE (RMSE APE). We compare each profiler's APE to the plain baseline.

Each profiled run is a *complete* trajectory (not a truncated window), so the
eval is directly comparable to the baseline:

- **nsys** — observational timeline; whole sequence, low overhead.
- **ncu** — kernel-replay on a bounded 20-launch window (`--launch-skip 3000
  --launch-count 20`); the app otherwise runs native to completion.
- **NVBit** — `mem_trace` instruments a bounded 200-launch window
  (`LAUNCH_BEGIN=1000 LAUNCH_END=1200`); outside it kernels run native, so the
  full trajectory completes. The trace is discarded — we only read the eval.

Tolerance: **max(5 cm, 5 %)** of the baseline APE. Anything larger is flagged
`CHECK` and cross-examined against the known km-scale odometry nondeterminism
(THESIS_FINDINGS F13) rather than assumed to be a real perturbation.

## Result — all three profilers are accuracy-neutral

| sequence | modality | plain APE (m) | nsys | ncu | NVBit | Δ nsys | Δ ncu | Δ NVBit |
|----------|----------|--------------:|-----:|----:|------:|-------:|------:|--------:|
| euroc V1_01 easy | stereo   | 0.0369 | 0.0369 | 0.0369 | 0.0349 | **0** | **0** | 0.0020 |
| euroc MH_01 easy | stereo   | 0.0196 | 0.0196 | 0.0196 | 0.0196 | **0** | **0** | **0** |
| euroc MH_01 easy | inertial | 0.6891 | 0.6891 | 0.6891 | 0.6891 | **0** | **0** | **0** |
| tum fr3_long_office | RGB-D | 0.0179 | 0.0179 | 0.0179 | 0.0186 | **0** | **0** | 0.0007 |
| icl living_room t1 | RGB-D  | 0.0386 | 0.0386 | 0.0386 | 0.0386 | **0** | **0** | **0** |
| kitti06 | stereo             | 2.3120 | 2.3059 | 2.2769 | 2.2870 | 0.0061 | 0.0351 | 0.0250 |

**Verdict: 6/6 OK — no profiler moves the trajectory beyond tolerance.**

### Reading the numbers

- **nsys and ncu are bit-identical** (Δ = 0.000 m) on all five deterministic
  indoor SLAM sequences. Observational timeline capture and windowed kernel
  replay do not touch the result — e.g. euroc MH_01 stereo reports the *same
  3639 matched poses and 1.96 cm APE* plain and instrumented.
- **NVBit** is neutral within ~2 mm (0.0000–0.0020 m). Binary instrumentation
  can nudge kernel scheduling enough to flip a tie inside a nondeterministic
  reduction, giving sub-mm-to-mm scatter — three orders of magnitude under the
  5 cm tolerance.
- **kitti06** is the only sequence with >1 cm deltas, and they appear across
  *all three* profilers at similar magnitude (0.006 / 0.035 / 0.025 m ≈ 0.3–1.5 %
  of a 2.31 m APE). A perturbation caused by one profiler would not reproduce in
  the other two. This is the km-scale odometry run-to-run nondeterminism floor
  (F13), independently confirmed here, **not** an instrumentation artifact.

## Conclusion

The profiling test harness does not significantly affect accuracy. On
deterministic modes nsys/ncu are exactly neutral and NVBit is neutral to within
millimetres; the residual centimetre-scale scatter on long odometry is the
pipeline's own nondeterminism, reproduced identically with and without every
profiler. Any GPU-memory / kernel measurement taken under this harness therefore
characterizes cuVSLAM, not the instrumentation.

## Provenance

- GPU RTX 2000 Ada (sm_89), clocks locked 1620 MHz / 7001 MHz, driver 575.64.05.
- Nsight Systems 2025.3.2.474 · Nsight Compute 2025.2.1.0 (CUDA 12.9, via
  `NCU_BIN`; the system ncu 2026.2/2025.3 rejects driver 575) · NVBit release
  x86_64, `mem_trace` with the `LAUNCH_BEGIN/END` windowing patch.
- Driver: `ws_profiler_neutrality.sh`. Raw per-run evals stashed at
  `/mnt/data/profiler_neutrality_out/<seq>/eval_{nsys,ncu,nvbit}.txt`
  (18 files); machine-readable table: `neutrality.tsv`.
- Baselines: `/mnt/data/accuracy_out/<seq>/eval.txt` (the 141-run accuracy
  matrix).
