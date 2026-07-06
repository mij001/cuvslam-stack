# Fulfillment of the Original Proposal Objectives — Panel Defense

*Maps the objectives of the approved FYP proposal ("profiling a
state-of-the-art autonomous-driving stack on NVIDIA Jetson Orin to justify
and inform the design of a memory-centric accelerator") onto the completed
project (the cuVSLAM memory characterization). **Every claim below carries a
citation to a committed, reproducible artifact in this repository** — file
paths are given in-line, and §D is a one-page claim→evidence index. This is
a fulfillment argument, not a new proposal.*

---

## 0. The one-paragraph position

The proposal's intellectual core — stated in its own Introduction — is that
autonomous systems are constrained by the memory wall, that *"a tailored,
workload-aware PIM design is essential,"* and that the project would
therefore deliver *"a detailed, quantitative analysis of [a real AD
workload's] memory access patterns"* to drive that design. **That core is
fulfilled in full, and at a finer granularity than proposed** — down to
which named data structure every GPU kernel's memory traffic lands in,
across 27 sequences from 4 public datasets. Two substitutions were forced by
procurement reality (the Jetson Orin could not be procured; only the
workstation could) and are shown below to be scope-preserving: the workload
(Autoware → cuVSLAM, NVIDIA's production localization stack) and the
platform (Jetson Orin → RTX 2000 Ada workstation, with the methodology
deliberately built hardware-parameterized so the Orin run is a
configuration swap — the descriptor is already committed,
`profiling/hw/jetson_orin_sm87.toml`). The deliberately deferred item is
the proposal's Phase-3 co-design back half (Objectives 4–6: the
architectural specification, scheduler, and analytical model), deferred for
a scientific reason the characterization itself produced (§B), with every
input artifact for that phase already generated, committed, and cited.

---

## A. Objective Mapping

### Objective 1 — *"Deploy and configure the [AD] stack on an NVIDIA Jetson Orin to create a stable testbed."*

**Core requirement:** a stable, repeatable, real-system testbed running a
state-of-the-art autonomous-system workload, suitable for trustworthy
profiling (proposal §5.1: *"a repeatable and relevant experimental
environment is the foundation of this research"*).

**How it is fulfilled — with proof:**

- **Deployed production stack.** cuVSLAM 15 — NVIDIA's production visual-SLAM
  library, shipped in NVIDIA Isaac — deployed as a fully scripted testbed,
  including a **from-source instrumented build** (which the proposal never
  contemplated for Autoware).
  *Proof:* the runner and configs (`run.py`, `cuvslam_runner/`,
  `configs/`), the build recipe (`patches/0001-podman-wheel-build-cuda13.patch`),
  and the instrumentation patch (`patches/0002-tagged-allocator-nvtx.patch`).
- **Stability quantified, not asserted.** Locked GPU clocks give a 5-repeat
  run-to-run coefficient of variation of **0.14 %**, versus **49.6 %**
  measured unlocked — a 350× improvement in measurement stability, with the
  protocol documented.
  *Proof:* `profiling/PUBLISHABILITY.md` (issue 1, CLOSED),
  `profiling/METHODOLOGY.md`, variance analysis in
  `profiling/analysis/variance.py`.
- **Calibrated, not datasheet-driven.** Roofline ceilings measured on the
  testbed at lock: **205.0 GB/s ± 0.1 DRAM, 5 445 GFLOP/s ± 3 FP32**.
  *Proof:* `profiling/env/measure_ceilings.py`;
  `profiling/hw/dellworkstation_sm89.toml`.
- **Deterministic replay.** The workload's allocation structure is
  bit-identical across runs and run lengths — the same 240 device
  allocations for a 30-frame and a 2 500-frame run.
  *Proof:* `profiling/reports/2026-07-04_attribution/alloc_table_full_run.csv`
  and ATTRIBUTION.md ("identical structure across all runs").
- **Correctness validated against the vendor.** The proposal had no
  correctness check at all; our testbed's trajectories are evaluated
  against ground truth with the vendor paper's own metrics, and reproduce
  or beat its published accuracy: KITTI 00–10 SLAM **RMSE APE 1.413 m vs
  the paper's 1.98 m**, segment avgRTE **0.750 % vs 0.85 %**; EuRoC V1_01
  stereo **ATE 7.78 cm**, reproduced exactly across machines and builds.
  *Proof:* `results/kitti_00-10_sweep.txt` (per-sequence tables with the
  paper deltas printed), `kitti_paper_sweep.py` (paper constants encoded),
  `PAPER_DATASETS.md` (per-mode validation status), `evaluate.py` +
  `cuvslam_runner/eval.py` (the metric implementations), and the 104-config
  matrix tooling (`gen_accuracy_configs.py`, `ws_accuracy_matrix.sh`,
  `accuracy_report.py`).

**On the two substitutions:**

- *Platform.* The Jetson Orin AGX Developer Kit **could not be procured;
  the department could provide only the Dell workstation** (RTX 2000 Ada).
  The constraint was outside the project's control, and it was absorbed
  methodologically rather than by narrowing scope: every analysis is
  hardware-parameterized through per-device descriptor files — the Orin
  descriptor is already written (*proof:*
  `profiling/hw/jetson_orin_sm87.toml`, alongside `mx450_sm75.toml` and
  `rtx2000ada_sm89.toml`, demonstrating the harness genuinely runs across
  device classes). This claim must be stated precisely rather than broadly,
  because two distinct things are involved and only one of them is
  architecture-independent: (1) the **reuse-distance-CDF technique** — given
  one fixed address trace, predicting the hit rate at *any* hypothetical
  cache capacity without needing a real cache of that size — is a property
  of the algorithm operating on a trace, and needs no re-derivation per GPU
  (*proof:* `profiling/reports/2026-07-04_slice3_locality/FINDINGS.md` §1);
  and (2) the **data-structure taxonomy** — which buffers exist, their sizes,
  their persistence class — is expected to be architecture-stable *because
  it is set by cuVSLAM's own source-level allocation calls*, not by
  target-architecture codegen. What is **not** architecture-independent, and
  is disclosed as such rather than assumed: the specific *split* of a
  kernel's DRAM traffic between register spill (local memory) and true data
  movement (global memory) is produced by the compiler's register allocation
  for a *specific target compute capability* — a different SM version (Orin
  is SM 8.7 vs the Ada SM 8.9 tested here) can allocate registers
  differently and spill a different amount, or none at all. The 94 %
  register-spill finding on the loop-closure kernel (Objective 4) is
  therefore a measured property of the **Ada-compiled binary**, flagged for
  re-validation — not re-assertion — the moment an Orin is available; the
  harness and taxonomy transfer immediately, that one codegen-dependent
  number does not transfer without re-measurement. The workstation
  moreover *enabled* rigor a Jetson cannot in the meantime: NVBit binary
  instrumentation required a controlled driver downgrade to 575.64.05
  coexisting with CUDA-12.9 Nsight tools (*proof:* the unified-stack notes
  in `profiling/PROJECT_STATUS.md` and the gating script
  `profiling/blocked/check_capability.sh`), which is impossible on
  JetPack's fixed driver.
- *Workload.* Autoware is an umbrella stack; its data-intensive, GPU-heavy
  core modules are perception and **localization**. cuVSLAM is a production
  localization module, and this is not merely an analogy — **it satisfies
  Autoware's own documented plug-in contract for a pose estimator,
  interface-for-interface**:
  - Autoware's Required Localization Architecture specifies that a pose
    estimator is a swappable component whose output must be expressed in
    "ROS primitives for reusability" — a `PoseWithCovarianceStamped`-class
    message plus a `TwistWithCovarianceStamped`-class message plus a
    `map → base_link` transform — precisely so that alternative
    localization sources can be substituted behind one interface (Autoware
    Documentation, *Localization* architecture design page). This is a
    working mechanism today, not an aspiration: Autoware already runs
    interchangeable pose sources (NDT scan matching, GNSS) through the same
    `/localization/pose_estimator/pose_with_covariance` topic, and Autoware
    Universe ships a dedicated arbitration node,
    `autoware_pose_estimator_arbiter`, whose job is to manage multiple
    simultaneous pose-estimator sources and switch between them.
  - cuVSLAM's shipped ROS 2 interface (NVIDIA's `isaac_ros_visual_slam`
    package) publishes **exactly those message types**: a
    `geometry_msgs/msg/PoseWithCovarianceStamped` on
    `visual_slam/tracking/vo_pose_covariance`, and a
    `nav_msgs/msg/Odometry` (which itself bundles pose-with-covariance and
    twist-with-covariance in one message) on `visual_slam/tracking/odometry`
    — the identical message-type contract Autoware's architecture calls
    for, produced natively by the module we profiled, with no adapter
    required.
  - **Stated at the strength the evidence supports, and no further:** this
    is a verified **interface-compatibility** claim — cuVSLAM's output
    contract matches Autoware's documented input contract for a pose
    estimator, type-for-type. It is *not* a verified deployment claim: the
    pose-estimator arbiter's currently documented supported sources are NDT,
    YabLoc, Eagleye, and landmark/AR-tag localizers; no camera-based VSLAM
    source is listed there today. We disclose this distinction explicitly
    rather than imply an existing integration we cannot cite.
  cuVSLAM also carries the decisive advantages that it is GPU-native
  end-to-end, production-deployed (a stronger instance of the proposal's
  "real-world application needs" than a research assembly), and
  source-obtainable, which enabled the from-source instrumentation that
  became the project's main methodological contribution. The proposal's own
  gap statement — *"a detailed analysis that connects [PIM] architectural
  concepts to the specific, fine-grained bottlenecks of a full,
  state-of-the-art [autonomous] stack is still missing"* — is answered
  exactly, for the localization stack, on a module that is not merely
  Autoware-*like* but Autoware-*pluggable* by its own published interface.

### Objective 2 — *"Perform a comprehensive profiling of data-intensive modules under varying workload conditions (e.g., different sensor data densities) to understand how bottlenecks change dynamically."*

**Core requirement:** multi-configuration profiling across workload
intensities; understand bottleneck dynamics. The proposal's §5.2 names the
instruments: nsys for end-to-end movement, ncu for per-kernel stall/bound
analysis.

**How it is fulfilled — and exceeded, with proof:**

- **The proposed instruments, used as proposed.** nsys end-to-end timelines
  and ncu per-kernel counters (including the stall set the proposal called
  out) drive the whole pipeline; the ncu metric taxonomy follows the
  published GPU-database-characterization methodology [Cao23].
  *Proof:* `profiling/harness/profile.py` (the nsys/ncu wrapper with its
  curated `METRIC_SETS`), `profiling/analysis/roofline.py` (arithmetic-
  intensity/boundness against the measured ceilings — the proposal's
  memory-bound/compute-bound axis), `profiling/analysis/build_dag.py`,
  `profiling/WALKTHROUGH.md`.
- **Varying workload conditions, at far larger scale than proposed.** The
  proposal envisioned "multiple rosbags (sparse highway vs dense urban)".
  Delivered: a **27-sequence campaign across 4 public datasets** — KITTI
  street-scale driving 00–10, EuRoC aggressive drone flight ×11, TUM RGB-D
  indoor ×4 *including the texture/structure ablations (the direct analog
  of sensor-density variation)*, TUM-VI — each in odometry *and* full-SLAM
  mode, **0 failures**; plus a 104-configuration accuracy matrix spanning
  mono/stereo/RGB-D/inertial sensing and sync/async/CPU-SLAM feature
  toggles.
  *Proof:* `profiling/reports/2026-07-04_campaign/CAMPAIGN.md` (the
  campaign synthesis; re-derivable byte-identical from the committed
  per-sequence CSVs in `per_sequence/`), `profiling/campaign/gen_configs.py`,
  `gen_accuracy_configs.py` (the 104-config generator).
- **Bottleneck classification, validated.** Every kernel classified into
  memory/compute behavior classes (GPU-adapted DAMOV taxonomy);
  **61 % of GPU time carries PiM affinity** (21 % strong + 40 %
  conditional) — the quantitative justification the proposal's title
  promised; and the class structure is confirmed by unsupervised
  clustering (k-means silhouette optimum at k = 7–8, matching the class
  count; purity 0.68).
  *Proof:* `profiling/analysis/classify.py` (the G1–G7 decision tree),
  `profiling/reports/2026-07-04_campaign/` (`clusters.csv`,
  `cluster_sweep.csv`, PiM rollup table in CAMPAIGN.md),
  `suggestions_and_summuries/Adapting_DAMOV_to_GPU.md` (the adaptation
  study).
- **Deeper than proposed, twice.** Beyond counters: (1) ground-truth
  **address traces** (NVBit) giving measured reuse-distance-vs-capacity
  curves, footprints, and coalescing — the front-end's reuse CDF is flat
  from 64 KiB to 48 MiB, i.e. *provably* cache-immune streaming; (2)
  **per-data-structure attribution** — which named allocation every access
  lands in.
  *Proof:* `profiling/analysis/locality.py` +
  `profiling/reports/2026-07-04_slice3_locality/` (`data/`, `data_v2/`);
  `profiling/analysis/attribution.py` +
  `profiling/reports/2026-07-04_attribution/` (274/274 allocations
  resolved; join tables); `profiling/blocked/mem_trace_windowing.patch`
  and `mem_trace_alloc_events.patch` (the tool extensions that made both
  feasible).
- **Data movement at the system edge, measured.** Explicit host↔device
  copies cost **41 % of kernel time** on the TUM workload; the sensor
  upload is **1.68 MB/frame** — the proposal's "data movement overhead"
  item (§5.2, nsys), quantified.
  *Proof:* `profiling/analysis/transfers.py`; PUBLISHABILITY issue 11
  (host↔device side marked resolved, with the numbers).
- **"How bottlenecks change dynamically" — answered with evidence.**
  Cross-sequence, per-kernel class consistency is **91 %** (24/49 kernels
  unanimous, 42/49 ≥ 80 %), and the flips that do occur happen at a
  physically meaningful boundary — working set vs L2 capacity (e.g., the
  loop-closure scan's per-scan footprint grows 0.46 → 1.09 MB from
  room-scale to street-scale).
  *Proof:* `profiling/reports/2026-07-04_campaign/class_agreement.csv`;
  footprint/Jaccard rows in
  `reports/2026-07-04_slice3_locality/FINDINGS.md` (§3 table, unchanged by
  the §5 correction).

*Residual within this objective, stated before the panel finds it:* the
proposal listed tegrastats power logging; energy measurement moved with the
deferred phase (§B), where per-kernel energy comes from the simulator's
power model. This is disclosed in `profiling/PUBLISHABILITY.md` (issue 6).

### Objective 3 — *"Identify a low-cost, runtime-observable metric that correlates with a kernel's shift between memory-bound and compute-bound."*

**Core requirement:** find the observable that predicts bound-ness.

**How it is fulfilled (with an honest reframing) — with proof:**

- The proposal *hypothesized* the metric by analogy (LiDAR point-cloud
  size). The measured answer has the same form and is now grounded: the
  bound-ness-shifting mechanism is **working-set footprint vs cache
  capacity** (the L2 crossover), and the runtime-observable quantities
  driving that footprint are the workload-density analogs the proposal
  anticipated — map size / keyframe count / tracked-feature count / scene
  scale.
  *Proof:* the loop-closure scan's footprint scaling with map extent
  (0.464 MB room → 1.093 MB street,
  `reports/2026-07-04_slice3_locality/data_v2/*_sttrack_global/locality.csv`);
  the ±25 % threshold sensitivity analysis identifying exactly which
  kernels sit at the crossover (`profiling/METHODOLOGY.md`;
  PUBLISHABILITY issue 2's note that the residual class flips are "the
  physically-meaningful L2 crossover, not noise").
- The stronger scientific result: for **48 of 49 kernels, the
  memory-behavior composition does not shift at all across 27 sequences**
  — bound-ness is dominantly a *static* property of the kernel and its
  data structure, with a small, well-characterized dynamic residue.
  *Proof:* `profiling/reports/2026-07-05_attribution_campaign/attribution_consistency.csv`
  (the unanimity column) and `CAMPAIGN_ATTRIBUTION.md` (the 48/49
  headline); `class_agreement.csv` (91 % at class level).
- We present this as a finding, not a shortfall: an objective whose form is
  "find the metric that predicts X" is fulfilled when the prediction
  problem is **solved by measurement** — here, the answer is that the
  predictive observable is *which data structure the kernel touches*
  (known statically) plus footprint-vs-L2 for the crossover kernels. The
  proposal's dynamic-scheduling premise is thereby *partially refuted by
  our own data* — the strongest possible input to the design phase,
  because it converts a hard online-scheduling problem into a tractable
  static-placement problem with one dynamic corner case (§B, argument 1).

### Objective 4 — *"To use this analysis to propose a heterogeneous computing architecture combining the main GPU with a specialized PIM accelerator."*

**Status: fulfilled at the level the analysis licenses — the evidence-backed
architectural direction — with the detailed specification deferred (§B).**

What exists, committed and measured — this is the substance any such
architecture must encode, and it is *derived from measurement rather than
asserted*:

- A **three-way substrate mapping at data-structure granularity**:
  *streaming* structures → near-sensor processing (images 39.97 MB +
  pyramids 31.15 MB, flat reuse CDFs = no cache size helps, 1.68 MB/frame
  upload); *hot-persistent* solver state → DRAM-PiM (`ba_linear_system`,
  24.26 MB pre-sized, carrying **96.9 %** of the solver kernel's global
  traffic — the proposal's Fulcrum-class near-bank unit's ideal target);
  *cold-persistent* keyframe database → in-storage processing (the GPU
  working buffer is a **fixed 6.73 MB** while the session-scale store
  grows host-side — measured from the allocator, not inferred).
  *Proof:* the memory-budget table in
  `profiling/reports/2026-07-04_attribution/ATTRIBUTION.md` (Finding 1)
  backed by `alloc_table_full_run.csv`; the traffic shares in
  `join_steady_state/attribution.csv`; the taxonomy table with its
  hardware asks in `docs/THESIS_FINDINGS.md` §0/§1 (F5, F8–F11).
- **The GPU/PIM division of labor the objective asks for**, read off
  measured tables: the GPU retains the compute-side kernels (89–98 % of
  their traffic is on-chip shared-memory tiles — measured), while the
  memory-bound streaming and scan structures are the offload set.
  *Proof:* the per-kernel space split columns (med_shared/med_spill/
  med_global) in
  `reports/2026-07-05_attribution_campaign/attribution_consistency.csv`.

What is deferred is the *specification document* (unit microarchitecture,
interface, controller, sizing) — §B explains why writing it now would have
been premature.

### Objective 5 — *"To design a lightweight, PAPI-like dynamic scheduling policy…"* and Objective 6 — *"To develop an analytical model that evaluates the performance and energy benefits…"*

**Status: deferred with the architecture phase (§B) — but materially
advanced, with the advancement citable:**

- The scheduler objective's **decision inputs are done**: the threshold the
  proposal called "X" (its Phase-3 "hypothesis and thresholding" step) is
  measured — it is the L2-crossover footprint, per kernel, with published
  sensitivity bounds — and the 48/49 stability result defines exactly
  which few decisions are dynamic at all. A scheduler designed *before*
  this result would have been designed for a workload structure that does
  not exist.
  *Proof:* Objective 3's citations; `docs/THESIS_FINDINGS.md` §4-G1 and
  §5 Path B (the step-ordered design-phase plan).
- The analytical-model objective's **inputs are done and committed**:
  measured bandwidth/compute ceilings for calibration
  (`profiling/hw/*.toml`, `profiling/env/measure_ceilings.py`); per-kernel
  per-structure byte tables
  (`reports/2026-07-04_attribution/join_steady_state/attribution.csv`,
  `reports/2026-07-05_attribution_campaign/attribution_by_sequence.csv`);
  per-class GPU-time shares (`reports/2026-07-04_campaign/CAMPAIGN.md`);
  window traces in the exact format the standard academic GPU simulator
  replays (`profiling/blocked/run_accelsim.sh`, the mem_trace tooling, and
  the capture campaign scripts in `profiling/campaign/`); and the
  simulator-calibration targets (the measured reuse CDFs in
  `reports/2026-07-04_slice3_locality/data_v2/`).
- Per Objective 3's finding, the evaluation the proposal wanted ("dynamic
  vs static") collapses into the cleaner "placed vs unplaced" comparison —
  fewer free parameters, stronger conclusions — which is exactly how the
  design-phase plan specifies it (`docs/THESIS_FINDINGS.md`, Path B).

---

## B. Justifying the Missing Architecture Phase

Four arguments, in descending order of weight. We recommend presenting them
in this order.

**1. The characterization changed the design question — proceeding on the
proposal's Phase-3 plan would have been scientifically unsound.**
The proposal's architecture, scheduler, and analytical model were all
premised on one hypothesis: that kernels *shift dynamically* between memory-
and compute-bound with scene density, requiring an online decision unit at
the architecture's center. The measured result — 91 % class consistency
(`class_agreement.csv`), 48/49 kernels with workload-invariant
data-structure composition (`attribution_consistency.csv`), residual
dynamics localized to a single physically-understood L2 crossover —
**partially refutes that premise**. Discovering this *before* committing to
an architecture is the characterization doing exactly the job the proposal
assigned to it: *"justify and **inform** the design."* Designing the
proposed dynamic scheduler anyway, against our own evidence, would have
produced a specification optimized for a fiction. The correct engineering
response — re-derive the design phase from the measured taxonomy (static
placement across three substrates + a narrow dynamic residue) — is
precisely what the deferral enables.

**2. The prerequisite scope grew because the evidence demanded it, within a
fixed time budget.**
Two self-corrections during measurement were mandatory for any architecture
built on these numbers to be trustworthy, and each is documented in place:
(a) the counter-vs-trace reconciliation on the loop-closure kernel, which
ended with two independent instruments agreeing
(`reports/2026-07-04_slice3_locality/FINDINGS.md` §5, `data_v2/`;
PUBLISHABILITY rows NEW/NEW2, both closed); (b) the memory-space filtering
discovery — 88–98 % of raw trace records are on-chip tile or register-spill
traffic that silently poisons locality and attribution analyses
(`profiling/analysis/attribution.py`, the space buckets; the corrected
tables). Resolving these was not a detour; it is why the numbers can face a
panel. The proposal budgeted ~5 weeks for its Phase 2; the delivered
Phase 2 is a 27-sequence, three-instrument, self-validating characterization
with a 104-run accuracy validation. The time came from somewhere: it came
from Phase 3.

**3. Nothing about the deferred phase is blocked — every input artifact
exists, is committed, and is cited above.**
This is the difference between "not done" and "not started": simulator
replay tooling and traces (`profiling/blocked/run_accelsim.sh`; the capture
campaigns), measured ceilings for calibration (`profiling/hw/*.toml`),
per-kernel × per-structure byte tables (both attribution reports), the
substrate mapping with sizes (`docs/THESIS_FINDINGS.md` §0), a QoR-neutral
instrumented build for future co-design measurements (`patches/0002`, with
the accuracy matrix's instrumented-vs-baseline phase in
`ws_accuracy_matrix.sh`), and a written, step-ordered plan for the phase
(`docs/THESIS_FINDINGS.md`, Path B). The phase is *de-risked and
specified* — materially more than the proposal's own "Proposed Level of
Implementation" promised for it (a high-level model, explicitly excluding
RTL).

**4. The two-stage structure is the discipline's standard, not an
improvisation.**
Workload characterization and architecture design are separately publishable
stages with separate venues (ISPASS/IISWC → MICRO/ASPLOS-class), for the
same reason the proposal's own literature review criticizes PIM works
"evaluated with general-purpose benchmarks or specific ML kernels": the
field's failure mode is architecture designed on weak workload evidence.
The completed project sits deliberately on the strong side of that
boundary — the evidence stage, finished to publication grade
(`profiling/PUBLISHABILITY.md`: reviewer issues 1, 2, 3, 4, 7, 8 closed
with dated evidence) — rather than straddling both stages weakly within one
semester.

**Supporting fact for the record:** the platform substitution (§A,
Objective 1) also consumed unplanned effort the proposal's timeline never
budgeted — solving the driver/toolchain compatibility matrix
(driver 575.64.05 + CUDA 12.9 + ncu 2025.2 + NVBit 1.8 coexisting) that
made binary instrumentation possible on the procurable hardware. The Jetson
Orin remains in the plan exactly as the proposal intended, as a
configuration swap on committed tooling (`profiling/hw/jetson_orin_sm87.toml`),
the moment one can be procured.

---

## C. Closing statement (for the panel, verbatim if useful)

> "The proposal asked us to prove, with measurements from a real autonomous
> stack, that a workload-aware memory-centric design is justified — and then
> to design one. We completed the first mandate beyond its proposed scope:
> a production workload, four public datasets, three independent measurement
> instruments that cross-validate, per-data-structure attribution that no
> prior SLAM study provides, and accuracy validated against the vendor's own
> published results — every number regenerable from committed data. The
> measurement stage then did what measurement stages are for: it corrected
> the design premise — placement in this workload is dominantly static, not
> dynamic. The architecture phase is deferred, not abandoned: its inputs are
> complete, committed, and cited, and the design it now feeds will be built
> on evidence rather than on the hypothesis we started with."

---

## D. Claim → evidence index (one page for the panel pack)

| # | Claim | Number | Committed artifact |
|---|---|---|---|
| 1 | Testbed stability (locked clocks) | CoV 0.14 % vs 49.6 % unlocked | `profiling/PUBLISHABILITY.md` (issue 1), `profiling/METHODOLOGY.md` |
| 2 | Measured roofline ceilings | 205.0 GB/s; 5 445 GFLOP/s | `profiling/hw/dellworkstation_sm89.toml`, `profiling/env/measure_ceilings.py` |
| 3 | Campaign scale | 27 sequences × 4 datasets, odom+SLAM, 0 failures | `profiling/reports/2026-07-04_campaign/CAMPAIGN.md` |
| 4 | Bottleneck-class stability | 91 % modal consistency (24/49 unanimous) | `…/2026-07-04_campaign/class_agreement.csv` |
| 5 | Taxonomy validated, not asserted | k-means best k = 7–8; purity 0.68 | `…/2026-07-04_campaign/cluster_sweep.csv`, `clusters.csv` |
| 6 | PIM opportunity quantified | 61 % of GPU time PiM-affine (21 strong + 40 conditional) | `…/2026-07-04_campaign/CAMPAIGN.md` (rollup) |
| 7 | Data-movement overhead | copies = 41 % of kernel time; 1.68 MB/frame upload | `profiling/analysis/transfers.py`; PUBLISHABILITY issue 11 |
| 8 | Front-end is cache-immune streaming | reuse CDF flat 64 KiB→48 MiB | `…/2026-07-04_slice3_locality/data_v2/tum_odom_global/reuse_cdf.csv` |
| 9 | Loop-closure scan = scattered gather; counters confirmed | 23.4–30.0 sectors/warp global-only | `…/slice3_locality/FINDINGS.md` §5, `data_v2/*_sttrack_global/` |
| 10 | Working set scales with map (the crossover observable) | 0.464 → 1.093 MB; Jaccard 0.669 → 0.899 | same as #9 |
| 11 | GPU memory budget static; DB grows host-side | 240 allocs / 108.65 MB; keyframe buffer fixed 6.73 MB | `…/2026-07-04_attribution/alloc_table_full_run.csv`, ATTRIBUTION.md |
| 12 | Per-structure attribution, complete | 274/274 allocations resolved; unmapped ≤7 % (mostly <1 %) | `…/2026-07-04_attribution/` (join tables) |
| 13 | Attribution is workload-invariant | 48/49 kernels unanimous across 27 sequences | `…/2026-07-05_attribution_campaign/attribution_consistency.csv` |
| 14 | Kernel→stage mapping measured (not name-guessed) | st_track under "SLAM: LC & optimization" | `…/2026-07-04_attribution/nvtx_kern_sum.csv` |
| 15 | Accuracy ≥ vendor paper | KITTI SLAM APE 1.413 m vs 1.98 m; EuRoC V1_01 7.78 cm | `results/kitti_00-10_sweep.txt`, `PAPER_DATASETS.md`, `kitti_paper_sweep.py` |
| 16 | Config-matrix coverage (features/IMU/mono/async/CPU) | 104 generated configs | `gen_accuracy_configs.py`, `ws_accuracy_matrix.sh` |
| 17 | Orin readiness (platform substitution absorbed) | descriptor committed | `profiling/hw/jetson_orin_sm87.toml` |
| 18 | Reproducibility | analyses re-derive from committed CSVs; 18 GPU-free tests | `profiling/tests/test_analysis.py`, report `data/` dirs |
| 19 | Reviewer-issue register (methodological honesty) | issues 1,2,3,4,7,8 closed, dated | `profiling/PUBLISHABILITY.md` |
| 20 | Design-phase inputs ready (deferral ≠ blockage) | traces, ceilings, byte tables, plan | `profiling/blocked/run_accelsim.sh`, `docs/THESIS_FINDINGS.md` Path B |
| 21 | cuVSLAM's output interface matches Autoware's pluggable pose-estimator contract, type-for-type | `PoseWithCovarianceStamped` + `Odometry` on both sides | Autoware: [Localization architecture](https://tier4.github.io/autoware-documentation/latest/design/autoware-architecture/localization/), [pose_estimator_arbiter](https://autowarefoundation.github.io/autoware_universe/main/localization/autoware_pose_estimator_arbiter/); cuVSLAM: [isaac_ros_visual_slam topics](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_visual_slam/isaac_ros_visual_slam/index.html) — external, independently verifiable sources; no existing wired-up deployment is claimed |
