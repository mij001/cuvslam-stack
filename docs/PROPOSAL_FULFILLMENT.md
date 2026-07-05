# Fulfillment of the Original Proposal Objectives — Panel Defense

*Maps the objectives of the approved FYP proposal ("profiling a
state-of-the-art autonomous-driving stack on NVIDIA Jetson Orin to justify
and inform the design of a memory-centric accelerator") onto the completed
project (the cuVSLAM memory characterization). Every claim cites a committed,
reproducible artifact in this repository. This is a fulfillment argument, not
a new proposal.*

---

## 0. The one-paragraph position

The proposal's intellectual core — stated in its own Introduction — is that
autonomous systems are constrained by the memory wall, that *"a tailored,
workload-aware PIM design is essential,"* and that the project would
therefore deliver *"a detailed, quantitative analysis of [a real AD
workload's] memory access patterns"* to drive that design. **That core is
fulfilled in full, and at a finer granularity than proposed.** Two
substitutions were forced by procurement reality (the Jetson Orin could not
be procured; only the workstation could) and are shown below to be
scope-preserving: the workload (Autoware → cuVSLAM, NVIDIA's production
localization stack) and the platform (Jetson Orin → RTX 2000 Ada
workstation, with the methodology deliberately built hardware-parameterized
so the Orin run is a configuration swap, not new science). The single
deliberately deferred item is the proposal's Phase-3 back half — the
architectural specification, scheduler, and analytical model — deferred for
a scientific reason the characterization itself produced (§B), with every
input artifact for that phase already generated and committed.

---

## A. Objective Mapping

### Objective 1 — *"Deploy and configure the [AD] stack on an NVIDIA Jetson Orin to create a stable testbed."*

**Core requirement:** a stable, repeatable, real-system testbed running a
state-of-the-art autonomous-system workload, suitable for trustworthy
profiling (proposal §5.1: "a repeatable and relevant experimental
environment is the foundation of this research").

**How it is fulfilled.** We deployed **cuVSLAM 15** — NVIDIA's production
visual-SLAM stack, shipped inside NVIDIA Isaac and deployed on real robots
and AVs — as a fully scripted, versioned testbed on the RTX 2000 Ada
workstation, including a **from-source instrumented build** (which the
proposal never reached for Autoware). The testbed exceeds the proposal's
stability bar by construction:

- Locked GPU clocks (1620/7001 MHz): run-to-run coefficient of variation
  **0.14 %** vs 49.6 % unlocked — i.e., the "stable testbed" is *quantified*,
  not asserted (`profiling/METHODOLOGY.md`).
- A pinned, mutually compatible driver/toolchain stack (driver 575.64.05,
  CUDA 12.9, ncu 2025.2, NVBit 1.8) solved on one machine.
- Full-sequence deterministic replay: the allocation structure is
  bit-identical across runs and run lengths (240 identical allocations for a
  30-frame and a 2 500-frame run).
- **Correctness validated, not assumed**: a 104-run accuracy matrix evaluates
  our deployment against ground truth on every dataset and reproduces or
  beats the vendor paper's published accuracy (e.g., KITTI SLAM APE 1.41 m
  vs the paper's 1.98 m; EuRoC V1_01 ATE 7.78 cm). The proposal had no
  correctness-validation step at all — our testbed is demonstrably running
  the workload *as intended by its vendor*.

**On the two substitutions:**

- *Platform.* The Jetson Orin AGX Developer Kit **could not be procured**;
  the department could provide only the Dell workstation (RTX 2000 Ada).
  This constraint was outside the project's control. The substitution was
  absorbed methodologically: all analyses are **hardware-parameterized**
  (per-device descriptor files; `hw/jetson_orin_sm87.toml` is written and
  committed, ready for the Orin), and the headline analyses are
  **architecture-independent by construction** (reuse-distance CDFs and
  address-to-data-structure attribution are properties of the workload's
  address stream, not of the machine measured on). Orin (SM 8.7) and the
  RTX 2000 Ada (SM 8.9) are adjacent members of the same architectural
  family. Moreover, the workstation *enabled* rigor the Orin cannot deliver:
  NVBit binary instrumentation (driver-version-gated; required a controlled
  driver downgrade impossible on JetPack), interchangeable Nsight versions,
  and the trace formats required by the GPU simulator for the follow-on
  phase.
- *Workload.* Autoware is an umbrella stack; its data-intensive, GPU-heavy
  core modules are perception and **localization**. cuVSLAM is precisely a
  production localization module — the visual-SLAM counterpart of the
  proposal's target — with the decisive advantages that (a) it is
  GPU-native end-to-end (Autoware's GPU coverage is partial), (b) it is
  production code, satisfying the proposal's "real-world application needs"
  requirement more strongly than a research assembly, and (c) its full
  source was obtainable, enabling the from-source instrumentation that
  became the project's main contribution. The proposal's gap statement —
  "a detailed analysis that connects [PIM] architectural concepts to the
  specific, fine-grained bottlenecks of a full, state-of-the-art
  [autonomous] stack is still missing" — is answered exactly, for the
  localization stack.

### Objective 2 — *"Perform a comprehensive profiling of data-intensive modules under varying workload conditions (e.g., different sensor data densities) to understand how bottlenecks change dynamically."*

**Core requirement:** multi-configuration profiling across workload
intensities; understand bottleneck dynamics.

**How it is fulfilled — and exceeded.** This is the completed project's
center of mass:

- **Varying workload conditions, at far larger scale than proposed:** a
  **27-sequence campaign across 4 public datasets** (KITTI street-scale
  driving 00–10, EuRoC aggressive drone flight ×11, TUM RGB-D indoor ×4
  including texture/structure ablations, TUM-VI), each in odometry *and*
  full-SLAM mode, zero failures — spanning indoor/outdoor, room/street
  scale, sparse/dense features, loop/no-loop trajectories, and (in the
  accuracy matrix) mono/stereo/RGB-D/inertial sensor configurations. The
  proposal envisioned "multiple rosbags (sparse highway vs dense urban)";
  the delivered matrix is the same axis, wider.
- **The proposed toolchain, plus a deeper one:** the proposal named nsys and
  ncu stall/roofline analysis; both were used exactly as proposed (Nsight
  timeline + per-kernel counters, roofline against *measured* ceilings of
  205.0 GB/s and 5 445 GFLOP/s). We then went two levels deeper than
  proposed: **NVBit address traces** (ground-truth reuse-distance and
  coalescing measurement, DAMOV Step-2) and **per-data-structure
  attribution** (which allocation every access lands in — 48/49 kernels
  attributed unanimously across all sequences).
- **Bottleneck classification:** every kernel classified into memory-bound /
  compute-bound behavior classes (GPU-adapted DAMOV taxonomy), validated by
  unsupervised clustering (k-means silhouette optimum at k = 7–8 matching
  the class count), with **61 % of GPU time carrying PiM affinity** — the
  quantitative justification the proposal's title promised.
- **"How bottlenecks change dynamically" — answered with evidence:** the
  cross-sequence analysis shows per-kernel class consistency of **91 %**,
  and the class flips that do occur happen at a *physically meaningful
  boundary* (working set vs L2 capacity — e.g., the loop-closure scan's
  footprint grows 0.46 → 1.09 MB from room-scale to street-scale). This is
  the direct answer to the objective's question, and it is a *finding*, not
  a limitation: bottleneck identity in this workload class is
  predominantly **structural (per kernel and per data structure), not
  transient (per scene)** — see Objective 3 and §B for why this reshapes
  the design phase.

*Residual within this objective:* the proposal listed tegrastats power
logging. Whole-run power via NVML is pending (it accompanies the deferred
design phase, where per-kernel energy comes from AccelWattch); energy is the
one axis of Objective 2 that moved with the deferred phase.

### Objective 3 — *"Identify a low-cost, runtime-observable metric that correlates with a kernel's shift between memory-bound and compute-bound."*

**Core requirement:** find the observable that predicts bound-ness.

**How it is fulfilled (with an honest reframing).** The proposal
hypothesized the metric by analogy (LiDAR point-cloud size). The measured
answer for the localization stack has the same form and is now grounded:

- The bound-ness-shifting mechanism, measured from address traces, is
  **working-set footprint vs cache capacity** (the L2 crossover) — and the
  runtime-observable quantities that drive that footprint are exactly the
  workload-density analogs the proposal anticipated: **map size / keyframe
  count / tracked-feature count / scene scale** (the loop-closure scan's
  per-scan footprint scales with map extent; the ±25 % threshold
  sensitivity analysis identifies precisely which kernels sit near the
  crossover).
- The stronger scientific result is that for **48 of 49 kernels the
  memory-behavior composition does not shift at all across 27 sequences**
  — it is a stable property of the kernel and its data structure. The
  honest conclusion, which we present as a first-class finding rather than
  hide: *the dynamic-scheduling premise is largely refuted for this
  workload class; the observable that matters is which **data structure** a
  kernel touches (static, known at design time), with a small,
  well-characterized dynamic residue at the L2 crossover.* An objective is
  fulfilled when its research question is **answered by measurement** —
  including when the answer is "the effect you planned to exploit is
  smaller and more structured than hypothesized." This redirection is what
  de-risks the design phase (§B): it converts a hard online-scheduling
  problem into a tractable static-placement problem with a dynamic corner
  case, and that is a *better* input to architecture than the metric the
  proposal guessed at.

### Objective 4 — *"To use this analysis to propose a heterogeneous computing architecture combining the main GPU with a specialized PIM accelerator."*

**Status: fulfilled at the level the analysis licenses — the evidence-backed
architectural direction — with the detailed specification deferred (§B).**

What exists, committed and measured (this is more than a sketch; it is the
substance an architecture must encode):

- A **three-way substrate mapping at data-structure granularity** — the
  heterogeneous-architecture proposal in its evidence form:
  *streaming* structures (images 40 MB + pyramids 31 MB; flat reuse CDFs =
  provably cache-immune; 1.68 MB/frame sensor upload) → **near-sensor
  SRAM**; *hot-persistent* solver state (`ba_linear_system`, 24.3 MB,
  96–100 % of the solver's DRAM traffic on one pre-sized structure) →
  **DRAM-PiM**, the proposal's Fulcrum-class near-bank unit's ideal target;
  *cold-persistent* keyframe database (fixed 6.7 MB on-GPU working buffer;
  session-scale growth measured to live host-side) → **in-storage
  processing**, plus two device-side asks the proposal could not have
  anticipated (spill-local SRAM; near-memory gather).
- The GPU-vs-PIM division of labor the objective asks for is therefore
  *derived from measurement*: which bytes belong next to which memory, per
  named structure, with the GPU retaining the compute-bound, shared-
  memory-tiled kernels (89–98 % on-chip traffic — measured).

What is deferred is the *specification document* (interface, controller,
sizing) — §B explains why writing it now would have been premature.

### Objective 5 — *"To design a lightweight, PAPI-like dynamic scheduling policy…"* and Objective 6 — *"To develop an analytical model that evaluates the performance and energy benefits…"*

**Status: deferred with the architecture phase — but materially advanced.**

- The scheduler objective's *decision inputs* are done: the threshold the
  proposal called "X" (its Phase-3 "hypothesis and thresholding" step) is
  measured — it is the L2-crossover footprint, per kernel, with sensitivity
  bounds — and the 48/49 stability result defines exactly which (few)
  decisions are dynamic at all. A scheduler designed *before* this result
  would have been designed for a workload structure that does not exist.
- The analytical-model objective's *inputs* are done and committed: measured
  ceilings, per-kernel per-structure byte tables, per-class time shares,
  window traces in the exact format the standard GPU simulator (Accel-Sim)
  replays, and an energy path (NVML whole-run; AccelWattch per-kernel).
  The evaluation the proposal wanted ("dynamic vs static") collapses, per
  Objective 3's finding, into the cleaner "placed vs unplaced" comparison —
  fewer free parameters, stronger conclusions.

---

## B. Justifying the Missing Architecture Phase

Four arguments, in descending order of weight. We recommend presenting them
in this order.

**1. The characterization changed the design question — proceeding on the
proposal's Phase-3 plan would have been scientifically unsound.**
The proposal's architecture, scheduler, and analytical model were all
premised on one hypothesis: that kernels *shift dynamically* between memory-
and compute-bound with scene density, requiring an online decision unit.
The measured result (91 % class consistency; 48/49 kernels with workload-
invariant data-structure composition; the residual dynamics localized to a
single physically-understood crossover) **partially refutes that premise**.
Discovering this *before* committing to an architecture is the
characterization doing exactly the job the proposal assigned to it —
"justify and **inform** the design." Designing the proposed dynamic
scheduler anyway, against our own evidence, would have produced a
specification optimized for a fiction. The correct engineering response —
re-derive the design phase from the measured taxonomy (static placement +
narrow dynamic residue) — is precisely what the deferral enables.

**2. The prerequisite scope grew because the evidence demanded it, within a
fixed time budget.**
Two self-corrections during measurement (the counter-vs-trace reconciliation
on the loop-closure kernel; the memory-space filtering that revealed 94 % of
its traffic is register spill) were not detours — resolving them was
mandatory for any architecture built on these numbers to be trustworthy, and
each produced a publishable methodological finding. The proposal's Phase 2
("profiling & dynamic bottleneck analysis") was budgeted ~5 weeks of a
semester; the delivered Phase 2 is a 27-sequence, three-instrument,
self-validating characterization with a 104-run accuracy validation — the
depth that top-venue characterization work requires. The time came from
somewhere: it came from Phase 3.

**3. Nothing about the deferred phase is blocked — every input artifact
exists, is committed, and is reproducible.**
This is the difference between "not done" and "not started": window traces
in Accel-Sim's replay format; measured bandwidth/compute ceilings for
simulator calibration; per-kernel × per-data-structure byte tables; the
substrate mapping with sizes; the QoR-neutral instrumented build for any
future co-design measurements; and a written, step-ordered plan
(`docs/THESIS_FINDINGS.md`, Path B) for the NDP simulation configs and
AccelWattch energy deltas. The phase is *de-risked and specified*, which is
materially more than the proposal's own "Proposed Level of Implementation"
promised for it (a high-level model, explicitly excluding RTL).

**4. The two-stage structure is the discipline's standard, not an
improvisation.**
Workload characterization and architecture design are separately publishable
stages with separate venues (ISPASS/IISWC → MICRO/ASPLOS-class), for the
same reason the proposal's own literature review criticizes PIM works
"evaluated with general-purpose benchmarks": the field's failure mode is
architecture designed on weak workload evidence. The completed project sits
deliberately on the strong side of that boundary — the evidence stage,
finished to publication grade — rather than straddling both stages weakly
within one semester.

**Supporting fact for the record:** the platform substitution (§A,
Objective 1) also consumed unplanned effort that the proposal's timeline
had not budgeted — solving the driver/toolchain compatibility matrix that
made binary instrumentation possible on the procurable hardware. The Jetson
Orin remains in the plan exactly as the proposal intended, as a
configuration swap on committed tooling (`hw/jetson_orin_sm87.toml`), the
moment one can be procured.

---

## C. Closing statement (for the panel, verbatim if useful)

> "The proposal asked us to prove, with measurements from a real autonomous
> stack, that a workload-aware memory-centric design is justified — and then
> to design one. We completed the first mandate beyond its proposed scope:
> production workload, four public datasets, three independent measurement
> instruments that cross-validate, per-data-structure attribution that no
> prior SLAM study provides, and accuracy validated against the vendor's own
> published results. The measurement stage then did what measurement stages
> are for: it corrected the design premise. The architecture phase is
> deferred, not abandoned — its inputs are complete, committed, and
> reproducible, and the design it now feeds will be built on evidence
> rather than on the hypothesis we started with."
