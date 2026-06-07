# cuVSLAM Memory-Profiling Research — Onboarding Package v5 (Annotated Edition)

**Status.** Supersedes v4. Same scope, same hardware, same methodology. What's new is the *reasoning*. Every major choice — every tool, every methodology, every dataset, every parameter — now comes with the rationale behind it. Read this version when you want to understand the project at a level deep enough to defend it to a sceptical reviewer, not just execute it.

**Scope.** Profiling and memory-access characterization of NVIDIA cuVSLAM on a Dell Precision 7875 workstation: AMD Ryzen Threadripper PRO 7945WX (12 cores), 32 GB system RAM, NVIDIA RTX 2000 Ada Generation GPU (AD107 die, 16 GB GDDR6 with ECC, 12 MB L2, 224 GB/s, `sm_89`, 70 W). Host OS: Ubuntu 22.04 LTS native. Secondary target: NVIDIA Jetson AGX Orin (planned extension after workstation profiling stabilizes).

**Reading guidance.** Each section has two layers. The plain prose tells you *what* to do; the bold-italic *Why* paragraphs tell you why you're doing it. Skim the *Why*s on first read for the conceptual map; come back to the *what*s when you sit down to execute.

---

## Table of contents

0. Why this project exists — the conceptual setting
1. Big picture and research thesis
2. SLAM and Visual SLAM — minimum theory you need
3. What cuVSLAM actually is (and isn't)
4. Hardware: what RTX 2000 Ada means for this project
5. Ubuntu 22.04 setup — OS, driver, CUDA, profiler permissions
6. cuVSLAM installation on Ubuntu
7. Datasets — getting EuRoC, KITTI, TUM RGB-D, TUM VI
8. Sanity baseline — running cuVSLAM and computing trajectory error
9. The profiling toolchain — Nsight Systems, Nsight Compute, NVBit, Accel-Sim
10. Phase 1 — Decomposing cuVSLAM into a memory-hierarchy-aware DAG
11. Phase 2 — Building the profiling testbed + the address-to-data-structure mapping pipeline
12. Phase 3 — Quantitative memory characterization, with calibrated steady-state cache simulation
13. Phase 3.5 — Extension to Jetson AGX Orin
14. Why Isaac Sim is deferred
15. Deliverables and milestones
16. Annotated bibliography
17. Glossary
18. Common pitfalls and FAQ

---

## 0. Why this project exists — the conceptual setting

If you only read one section, read this one.

### The memory wall, in one paragraph

For roughly fifty years, the computer-architecture community has watched two trends march in opposite directions. Compute throughput — the raw arithmetic a processor can perform per second — has grown roughly along Moore's law, doubling every two-or-so years until ~2010 and continuing more slowly since. *Memory bandwidth* — the rate at which data can be moved between off-chip DRAM and the processor — has grown much more slowly. The result is that on modern processors, most workloads spend most of their time waiting on memory rather than computing. The arithmetic units are idle, drumming their fingers, while data is shuffled across the memory hierarchy. This phenomenon has a name: *the memory wall*.

**Why this matters for you.** Once you accept that memory is the bottleneck, the natural architectural response is *move the compute to where the data lives*. That's the entire idea behind Processing-in-Memory (PiM) and In-Storage Processing (ISP) — instead of pulling data across the bus to the CPU/GPU, you put a small amount of compute logic inside the DRAM die or the SSD controller. The PiM/ISP literature has been building this case rigorously since about 2015, and you'll see citations to that literature (Mutlu, Gómez-Luna, Oliveira) throughout this document.

**But there's a problem.** PiM/ISP make sense *only for the right workloads*. If your workload is compute-bound — dense matrix multiplication, say — moving compute to weaker memory-side units loses. PiM is a win precisely when (a) your data is large, (b) you touch it irregularly enough that caches don't save you, and (c) the per-access arithmetic is small. So the *first* question for any prospective PiM/ISP application is: *does this workload actually look like that?* Answering that question rigorously, with measurement rather than hand-waving, is the entire job of a workload-characterization study.

### Why visual SLAM is the right vehicle

Visual SLAM (Simultaneous Localization And Mapping using cameras) is, for memory-architecture researchers, an almost ideal test workload. Three reasons:

- **It's real.** Drones, autonomous cars, mobile robots, AR/VR headsets — they all run V-SLAM in production. This isn't a synthetic benchmark; it's the thing you'd actually want to accelerate.
- **It's heterogeneous.** A V-SLAM pipeline has stages that look *very* different from each other — dense image convolutions (compute-bound, regular memory), feature gather/scatter (memory-bound, irregular), large-map graph traversal (extremely irregular, pointer-chasing). One application gives you a whole zoo of memory access patterns to study.
- **It hits a deadline.** Robotics is hard-real-time. A 30 FPS camera means every frame must finish in 33 ms or you've missed it. This forces you to think about *tail latency*, not just average throughput — and tail latency is where memory bottlenecks bite hardest.

### Why cuVSLAM specifically, and why now

Until mid-2025, the state of the art in visual SLAM that anyone could study was ORB-SLAM3 — an excellent CPU-based academic system. The problem is that ORB-SLAM3 runs on CPU, and a CPU's memory hierarchy (private L1/L2 + shared L3 + DRAM) is very different from a GPU's (per-SM L1 + unified L2 + DRAM). If you want to characterize *deployment-realistic* visual SLAM — what actually ships on a Jetson in a drone — you need a GPU-accelerated system. None existed in open source.

In June 2025, NVIDIA open-sourced **cuVSLAM** — their production CUDA-accelerated visual-inertial SLAM library, the thing that powers Isaac ROS, the thing that runs on every Jetson-based robot they sell. The technical report [Korovko25] dropped on arXiv at the same time. *This created a window*: for the first time, an architecture researcher can characterize a real, optimized, GPU-accelerated visual SLAM system, all the way down to the CUDA kernels.

**The project pitch in one sentence.** You're going to do for cuVSLAM what Oliveira et al. did for the DAMOV benchmark suite [Oliveira21] and what Gómez-Luna et al. did for the UPMEM PIM platform [Gomez-Luna22] — a rigorous, kernel-by-kernel memory characterization that identifies which parts of the workload are PiM/ISP candidates and which are not. The thesis is that the answer divides cleanly into three classes, and those three classes map onto three different kinds of memory-centric hardware. Phases 0–3 (the scope of this document) test that thesis empirically; Phase 4+ would design the hardware.

### Why this is a good first research project for you

This kind of project is the bread-and-butter of computer-architecture research — it's how almost every PhD in this area starts. You're doing the *characterization* phase, which is unglamorous compared to "I built a new accelerator" but is the prerequisite for any serious accelerator design. Reviewers at ISCA, MICRO, ASPLOS, and HPCA reject hardware proposals that aren't backed by characterization; characterization papers themselves are published at ISPASS and IISWC. If you do this well, you have a publishable artifact on its own *and* the foundation for everything downstream.

---

## 1. Big picture and research thesis

**The thesis (one sentence).** *The cuVSLAM pipeline contains three classes of memory accesses — streaming (single-frame), hot-persistent (local map, every frame), and cold-persistent (keyframe database, accessed on loop closure) — and these three classes map onto three classes of memory-centric hardware (near-sensor SRAM, LPDDR/HBM-PiM, and ISP), giving a quantitative basis for memory-centric robotics architectures.*

***Why three classes and not five, or two?*** The number isn't arbitrary, but it isn't a deep theorem either — it comes from observation of how V-SLAM data is *touched*. Image data is touched once per frame and thrown away (streaming). Map data within a few seconds of the present is touched every frame and grows slowly (hot-persistent). Map data from minutes or hours ago is touched only when you re-visit a place, which is rare per frame but expensive when it happens (cold-persistent). Three is the smallest number that captures the qualitatively distinct behaviours; fewer collapses real differences, more splits at boundaries that don't exist in the data.

***Why these three hardware classes specifically?*** Near-sensor SRAM is the right home for streaming data because you want to consume it before it ever hits off-chip DRAM — the data lives just long enough to do the streaming computation. LPDDR/HBM-PiM is the right home for hot-persistent data because (a) it has to be in DRAM anyway (too big for on-chip SRAM) and (b) you want bank-parallel access to it (which is what PiM provides). ISP is the right home for cold-persistent data because (a) it's so big you don't want it taking up DRAM, and (b) when you do touch it, you do bulk scans (descriptor matching, keyframe filtering) — which is exactly what computational storage is good at.

**Why your hardware is well-matched to the thesis.** The RTX 2000 Ada has only 12 MB of L2 cache and 224 GB/s of DRAM bandwidth — roughly 8× less bandwidth than an RTX 5090 and 8× smaller L2. This is a *feature*, not a bug, for finding PiM-favourable behaviour: working sets that would fit in an RTX 5090's L2 spill on your hardware, making the memory bottleneck more visible.

***The general principle:*** when you're trying to find a memory bottleneck, smaller caches and lower bandwidth are your friends. Top-bin GPUs (H100, MI300, RTX 5090) are designed precisely to *hide* memory bottlenecks. If you characterize a workload on those parts, you may mis-conclude it isn't memory-bound when it absolutely would be on the deployment hardware. The RTX 2000 Ada is much closer in cache/bandwidth balance to the Jetson AGX Orin you'll eventually use — so your characterization on the workstation will translate.

---

## 2. SLAM and Visual SLAM — minimum theory you need

You don't need to be a SLAM expert. You do need the vocabulary, because every kernel you profile maps to one of these stages.

***Why you should learn this section before touching any code:*** the entire project hinges on classifying CUDA kernels by *what they semantically do*, not just by what their hardware behaviour looks like. If you don't know that "PnP" means "pose estimation" or that "BoW" is the loop-closure mechanism, you'll see kernel names like `cuvslam_pnp_ransac_kernel` and have no idea what you're looking at. The vocabulary is what lets you map raw CUDA back to the pipeline.

**Definitions.**

- **SLAM** (Simultaneous Localization And Mapping). Given a stream of sensor readings, jointly estimate the agent's trajectory *and* a map of the environment. The "simultaneous" matters: you can't build a good map without knowing where you are, and you can't localize without a map. Survey: [Cadena16].
- **Visual SLAM (V-SLAM).** SLAM whose primary sensor is one or more cameras.
- **Visual-Inertial Odometry / SLAM (VIO, VI-SLAM).** V-SLAM augmented with an IMU (accelerometer + gyroscope). cuVSLAM is fundamentally a VIO system. ***Why add an IMU when cameras seem enough?*** Cameras are slow (10–60 Hz) and can be fooled by motion blur, low texture, or rapid rotation. IMUs are fast (200–1000 Hz) and immune to those failure modes, but drift over time. Fusing them gets you the best of both: short-term IMU drift is corrected by the cameras; the IMU bridges the gaps when vision fails.
- **Stereo VIO.** Two synchronized cameras with known geometry. Gives metric scale "for free" — a single camera (monocular) can't tell you whether a chair is small and close or large and far, but two cameras can triangulate.

**The canonical pipeline** (every kernel you profile maps to one of these):

1. **Image acquisition** — frames arrive at 20–60 Hz from one or more cameras.
2. **Image preprocessing** — undistortion, rectification (for stereo), pyramid construction (multi-scale, so you can detect features of different sizes).
3. **Feature detection** — find salient keypoints (corners). cuVSLAM uses a FAST/Harris-family detector; ORB-SLAM3 uses FAST + an orientation step. ***Why salient keypoints and not every pixel?*** Tracking every pixel is computationally infeasible and unnecessary — most pixels carry redundant information. Corners and edges are *distinctive* enough that you can find the same one in a later frame, which is what tracking requires.
4. **Descriptor extraction** — compute a compact signature at each keypoint, typically a 256-bit binary string (ORB). The descriptor's job is to be similar for the *same* keypoint seen from different viewpoints and different for *different* keypoints. Hamming distance between descriptors is the similarity metric.
5. **Stereo matching** — match keypoints between left and right images to triangulate 3D positions.
6. **Temporal feature tracking** — associate keypoints in frame *t* with those in frame *t-1*.
7. **Pose estimation (front-end)** — given 2D-to-3D correspondences, solve for the current camera pose. This is the **PnP** (Perspective-n-Point) problem, almost always solved inside a RANSAC loop to reject outlier matches. ***Why RANSAC?*** Feature matching is noisy; some matches are simply wrong. Plain least-squares is wrecked by a few outliers. RANSAC samples small subsets, fits a model, counts inliers, and picks the model with the most. It's slow and embarrassingly parallel, which is why GPUs do it well.
8. **IMU pre-integration** — between camera frames, integrate IMU samples into a single relative-motion constraint. Pre-integration is a clever piece of math that lets you avoid re-integrating from scratch every iteration.
9. **Keyframe selection** — most frames don't add information beyond their predecessors; pick the ones that do. ***Why selectively?*** If you stored every frame, the map would grow at 30 frames/second × however-many-hours and crush your memory. Keyframes are the spatial samples that matter.
10. **Local mapping / local bundle adjustment (BA)** — for the last *N* keyframes, jointly refine all camera poses and all 3D landmark positions by minimizing reprojection error. This is *sparse* nonlinear least squares — sparse because each landmark is only seen by a few keyframes, so the system matrix has a particular block structure that fast solvers exploit.
11. **Loop closure detection** — when you re-visit a place, recognize it. Almost universally done via a Bag-of-Words (BoW) vocabulary over historical keyframes, or a learned descriptor over the same. ***Why does this matter so much?*** Without loop closure, your pose estimate drifts unboundedly. Closing a loop lets you globally correct the trajectory.
12. **Global pose-graph optimization** — when a loop closes, redistribute the accumulated drift across the whole trajectory.
13. **Map management** — keyframes, landmarks, covisibility graph, descriptor database. Can hit hundreds of MB over a long mission.

**Front-end** = steps 2–8 (must run at frame rate, ~30 Hz hard deadline). **Back-end** = steps 10–12 (slower, runs in a separate thread). cuVSLAM has this split.

***Why this matters for memory characterization.*** Stages 3–5 are dense streaming convolutions over the current frame; stages 7, 10, 11 touch persistent map state with very different reuse patterns. This is the foundation of your three-way taxonomy — you'll see it directly in the measurements.

**Evaluation metrics.**

- **ATE** (Absolute Trajectory Error) — RMS of position error after rigid alignment to ground truth. In metres.
- **RPE** (Relative Pose Error) — drift per unit distance. In % (translation) and deg/m (rotation).

***Why two metrics and not one?*** ATE captures global drift — does your trajectory look right when you stand back. RPE captures local accuracy — are you smooth and accurate in the short term. A system can have great ATE and terrible RPE if it's accurate on average but jittery; or great RPE and terrible ATE if it's smooth locally but drifts unboundedly. You report both.

Definitions and tooling: [Sturm12] and the evo Python package [Grupp17].

---

## 3. What cuVSLAM actually is (and isn't)

**Citation.** Korovko, A., Slepichev, D., Efitorov, A., Dzhumamuratova, A., Kuznetsov, V., Rabeti, H., Biswas, J., Pouya, S. *cuVSLAM: CUDA accelerated visual odometry and mapping*. arXiv:2506.04359, 2025. https://arxiv.org/abs/2506.04359 — **read this before anything else.**

***Why this paper before any code:*** the source repository is large and not internally well-documented; reading it cold is overwhelming. The technical report describes the *architecture* — what the modules are, what they do, how they're connected. Once you have that mental model, the source becomes navigable.

**Properties:**

- Stereo + optional IMU + optional depth (RGB-D) VIO/VSLAM, supports up to 32 cameras in arbitrary geometric configuration.
- CUDA-accelerated end-to-end: feature detection, tracking, local BA, loop closure all run on GPU.
- Sub-1% translational error on KITTI Odometry; <5 cm mean position error on EuRoC.
- Real-time on Jetson Orin; >200 FPS on x86 + discrete GPU.
- Modular front-end/back-end split.
- Open-source since June 2025.

**Not a deep-learning SLAM system.** Classical feature-based VIO with GPU-accelerated classical operators.

***Why this is actually good for your project.*** A deep-learning SLAM (DROID-SLAM, NICE-SLAM, etc.) would be a monolithic neural-net forward/backward pass. Almost all the time would be in cuDNN/cuBLAS kernels, which are heavily optimized black boxes — hard to characterize meaningfully because the *interesting* code isn't yours and the access patterns are dominated by dense matmul. cuVSLAM's classical operators have *crisp logical boundaries*: feature detection is its own kernel, descriptor matching is its own kernel, bundle adjustment is its own kernel. You can isolate and characterize each.

**Repositories you'll actually use:**

- `nvidia-isaac/cuVSLAM` — the C++ library. Pre-built binaries for Ubuntu 22.04 / 24.04, x86_64 / aarch64, CUDA 12 / 13.
- `nvidia-isaac/PyCuVSLAM` — Python wrapper, pre-built wheels for Ubuntu.
- `NVIDIA-ISAAC-ROS/isaac_ros_visual_slam` — ROS 2 wrapper. Not needed for profiling.

---

## 4. Hardware: what RTX 2000 Ada means for this project

### 4.1 Confirmed specs

| Property | Value |
|---|---|
| Die | AD107 |
| Architecture | Ada Lovelace |
| Compute capability | `sm_89` |
| CUDA cores | 2,816 |
| Streaming multiprocessors | 22 |
| L1 cache | 128 KB per SM (≈ 2.75 MB total) |
| L2 cache | **12 MB** |
| Memory | 16 GB GDDR6 with ECC |
| Memory bandwidth | **224 GB/s** (128-bit bus × 14 Gbps) |
| Bus interface | PCIe 4.0 x8 |
| TDP | 70 W |

### 4.2 What this means for the project, and why

Read this section with the table above next to you — every number has a downstream consequence.

- **12 MB L2 reuse-distance threshold.** ***Why this matters:*** in Phase 3 you'll compute reuse distances; the L2/DRAM crossover is at 12 MB on this card. Any working set above ~10 MB (leaving headroom) will miss in L2 every time it's touched. This is your PiM-favourable signal. The threshold is a *property of your hardware*, not a generic number — cite it explicitly in any writeup.
- **224 GB/s bandwidth ceiling.** ***Why this matters:*** this is the height of the DRAM-bound roofline. Any kernel that approaches 100+ GB/s is bandwidth-pegged. When you report kernel bandwidth, express it as a fraction of 224 GB/s — that fraction is what defends the "this kernel is bandwidth-bound" claim.
- **16 GB VRAM.** Plenty for cuVSLAM (a typical session uses 1–4 GB), so VRAM isn't the constraint. The constraint is your *system* RAM at 32 GB.
- **PCIe 4.0 x8.** Halved compared to x16. ***Why not a problem:*** cuVSLAM is mostly device-resident; host↔device transfers are bounded to image upload and pose download, both small.
- **ECC supported.** ***Why this matters and why you must leave it on:*** consumer GPUs (RTX 5090, etc.) don't have ECC, so bit-flips in their VRAM are silent. Your card has ECC. Leave it enabled (`nvidia-smi --query-gpu=ecc.mode.current --format=csv` to check). If you disable it for the small (~1–3%) bandwidth gain, you risk silent data corruption that will burn a week of your time chasing a phantom bug in cuVSLAM that's actually an undetected bit-flip in your trace data.
- **sm_89 is well-validated** in every tool you'll use: Nsight Compute, NVBit, Accel-Sim. ***Why this matters:*** if you'd had a brand-new architecture (Blackwell `sm_120`), some tooling would have edge cases, missing metrics, or no validated simulator config. `sm_89` has been out since 2022 and is the path most architecture researchers have walked. You hit fewer surprises.

### 4.3 The 32 GB system RAM constraint

NVBit memory traces are large — a single profiled second of cuVSLAM can produce 10+ GB of address tuples. With 32 GB system RAM you cannot hold an entire sequence's trace in memory or run aggressive in-memory post-processing.

You must:

- **Stream traces to compressed disk.** Write zstd-compressed output as the trace is generated. Plan 50–200 GB of trace storage per dataset. ***Why zstd and not gzip:*** zstd is significantly faster (3–10× at comparable ratios) and your NVBit tool is already CPU-bound from the instrumentation overhead. Don't make compression worse.
- **Profile in chunks of 20–50 frames** (Section 11.3). ***Why not the whole sequence:*** a 60-second EuRoC sequence has 1200 frames × roughly 30 kernel launches per frame = 36,000 kernel invocations. Tracing all of them is both unnecessary (many invocations are redundant) and infeasible (you'd run out of disk).
- **Post-process on disk with streaming algorithms** for reuse-distance histograms. ***Why streaming:*** the naive reuse-distance algorithm needs an array as large as the unique-address count, which can be many GB for long traces. Streaming algorithms (`parda`-style, treap-based) trade some accuracy for bounded memory.

### 4.4 BIOS / firmware settings

Before installing Ubuntu, set in BIOS:

- **Secure Boot: Disabled.** ***Why:*** NVIDIA's third-party kernel modules need MOK (Machine Owner Key) signing to work with Secure Boot. The MOK procedure is brittle and you'll spend hours on it for zero research benefit. Disable.
- **AMD Precision Boost: Disabled** (for measurement reproducibility). ***Why:*** Boost gives variable CPU frequencies depending on workload, thermals, and history. Variable frequency = variable measurements. Disable for clean profiling runs; you can leave it on for development.
- **SMT: Optional.** With 12 physical cores you can spare SMT; disabling gives cleaner profiles. ***Why disable:*** SMT shares execution resources between two logical threads on each core, which adds noise to per-thread measurements. With 12 physical cores you don't need the extra throughput from SMT — cuVSLAM is GPU-bound anyway.
- **ECC: Enabled** for any installed ECC RAM. ***Why:*** same logic as GPU ECC, applied to system RAM. Catch bit-flips before they corrupt your post-processing.
- **IOMMU: Enabled** (default on most Threadripper PRO BIOSes). Required cleanly for some Nsight features.
- **Above 4G Decoding: Enabled.** Necessary for the GPU to expose its 16 GB BAR (Base Address Register) to the host. Without it the driver will fall back to a smaller window and you'll see weird performance.

---

## 5. Ubuntu 22.04 setup — OS, driver, CUDA, profiler permissions

### 5.1 Why Ubuntu 22.04 and not 24.04 or something else

22.04 LTS is the safe pick.

***The reasoning:*** every NVIDIA tool (CUDA 12.x, Nsight Compute, Nsight Systems, NVBit, Accel-Sim) is regression-tested on 22.04 first and 24.04 second. Some tools (Accel-Sim, in particular) have compile-path issues with GCC 13, which is the default on 24.04. Choosing 22.04 means you spend zero time fighting your toolchain. The LTS commitment also means you get five years of security updates without forced API churn. ***When would you pick 24.04 instead:*** only if you specifically need a newer kernel for hardware enablement reasons. The HWE kernel option in 22.04 covers most such cases without a full distro upgrade.

```bash
sudo apt install -y linux-generic-hwe-22.04   # newer kernel on a 22.04 base
```

### 5.2 Install Ubuntu 22.04 LTS

Suggested partition layout:

- 1 GB EFI System Partition.
- 80–120 GB ext4 root `/`.
- 16 GB swap.
- The remainder mounted at `/research` on a separate NVMe (or as ext4 on the same NVMe).

***Why swap when you have 32 GB:*** post-processing scripts that briefly exceed RAM (sorting a long trace, computing a histogram) will get OOM-killed without swap. With swap they slow down but finish. Swap is insurance.

***Why a separate `/research` mount:*** so a runaway script can't fill your root filesystem and brick the OS. Separation of concerns; classical sysadmin hygiene.

Pick the "minimal install" option to keep the base lean. ***Why:*** fewer background services means fewer interruptions to your profiling runs.

### 5.3 Install the NVIDIA driver

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y ubuntu-drivers-common
ubuntu-drivers devices
sudo ubuntu-drivers install
sudo reboot
nvidia-smi
```

***Why `ubuntu-drivers` and not download-from-NVIDIA-website:*** the Ubuntu-packaged driver integrates with dkms (so it rebuilds automatically against new kernels), with the package manager (so apt knows about it), and with Secure Boot signing (if you ever re-enable it). The website driver works but doesn't get these niceties.

***Why `nvidia-smi` is the first thing you run after reboot:*** if it shows the GPU, the kernel module loaded, the driver ABI matches, and the device is electrically present. If it doesn't, you've narrowed the failure to one of those four things. It's the cheapest "everything's plugged in" check.

### 5.4 Install the CUDA Toolkit

Use NVIDIA's official apt repository. The full instructions are at https://docs.nvidia.com/cuda/cuda-installation-guide-linux/.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6

echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

nvcc --version
nvidia-smi
```

***Why the NVIDIA repo and not Ubuntu's:*** Ubuntu's CUDA packages lag the official repo by months and sometimes carry distro-specific patches that interact badly with the latest profilers. NVIDIA's repo is the source of truth. Pin a specific minor version (12.6 not "12") so your builds remain reproducible.

***Why pin the version at all:*** CUDA's ABI is more stable than the driver ABI, but new CUDA minors do add metrics and occasionally rename them. If you upgrade mid-project, your scripts may stop finding metrics by their old names. Pin and only upgrade deliberately.

```bash
sudo apt install -y git build-essential
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples/Samples/1_Utilities/deviceQuery
make
./deviceQuery
```

***Why deviceQuery:*** this is the canonical end-to-end test. It enumerates the device, queries every property the driver exposes, and exercises a tiny CUDA kernel. If `deviceQuery` works, your CUDA install is sound. If it fails, the failure mode tells you exactly which layer (kernel module, user-space CUDA library, compiler) is broken.

### 5.5 Install Nsight Systems and Nsight Compute

```bash
sudo apt install -y nsight-systems nsight-compute
nsys --version
ncu --version
```

***Why these two are separate tools:*** they answer different questions. Nsight Systems gives you the *timeline* — the macro-scale view of what runs when. Nsight Compute gives you the *per-kernel deep dive* — for one kernel, every micro-architectural counter. You use Nsight Systems first to identify which kernels are worth zooming in on, then Nsight Compute on those kernels. Running Nsight Compute on every kernel is prohibitively slow because of its replay-based metric collection.

### 5.6 Profiler permissions (the one that always bites)

Out of the box, Nsight Compute will refuse to read GPU performance counters with `ERR_NVGPUCTRPERM`.

```bash
sudo tee /etc/modprobe.d/nvidia-profiler.conf <<'EOF'
options nvidia NVreg_RestrictProfilingToAdminUsers=0
EOF
sudo update-initramfs -u
sudo reboot
```

***Why this restriction exists:*** GPU performance counters can leak side-channel information between concurrent users (timing of cache hits, etc.). NVIDIA restricts counter access to root by default. On a shared cluster this matters. On your dedicated research workstation it doesn't — you're the only user — so disabling the restriction is safe and necessary. The kernel module option above flips the default for the entire system.

### 5.7 System packages and Python environment

```bash
sudo apt install -y \
    build-essential cmake git wget curl pkg-config \
    git-lfs python3-pip python3-venv python3-dev \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libeigen3-dev libopencv-dev \
    libzstd-dev zstd libtbb-dev \
    libunwind-dev binutils \
    htop iotop linux-tools-common linux-tools-generic \
    cpufrequtils
git lfs install
```

***Why each non-obvious package:***

- `git-lfs`: PyCuVSLAM ships its compiled `.so` files via Git LFS. Without it, you clone the repo and get text pointer-files instead of binaries. Then nothing works and the error is mysterious. Install upfront.
- `libzstd-dev` + `zstd`: for streaming compression of NVBit traces.
- `libunwind-dev`: for the NVBit `cuMemAlloc` interceptor in Section 11.2 — it uses `libunwind` to walk the host call stack and figure out *which line of cuVSLAM source* asked for a given allocation.
- `binutils`: provides `addr2line`, which converts program-counter values to source-file:line. Together with `libunwind`, these are the workhorses of attributing GPU allocations back to cuVSLAM source.
- `cpufrequtils`: for setting the CPU governor (next section).

Python environment:

```bash
python3 -m venv ~/venvs/vslam
source ~/venvs/vslam/bin/activate
pip install --upgrade pip wheel setuptools
pip install numpy scipy matplotlib pandas pyyaml opencv-python tqdm
pip install evo
pip install jupyter intervaltree zstandard pyarrow
```

***Why a venv and not system Python:*** isolation. A system Python install gets polluted with whatever every random apt package depends on. A venv is yours, you control it, you can `rm -rf` it and recreate without breaking anything.

***Why `intervaltree`:*** the post-trace join in Section 11.2.4 needs efficient lookup of "which allocation does this address fall into?" Interval trees do this in O(log n). A naive list scan is O(n), which over billions of accesses costs you days.

***Why `zstandard`:*** the Python bindings to zstd. Lets you stream-decompress your traces line by line without ever loading them whole into memory.

***Why `pyarrow`:*** Parquet is your output format for labeled traces. Columnar storage; fast querying; small on disk. The post-trace join writes Parquet so downstream notebooks can do `df.groupby(['kernel', 'tag']).agg(...)` without re-parsing the trace.

### 5.8 Clock locking and CPU governor

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1620
sudo nvidia-smi -lmc 7000
sudo cpupower frequency-set -g performance
```

***Why locking clocks:*** modern GPUs (and CPUs) run at variable clock rates depending on temperature, power, and workload. If clocks vary across runs, your measurements vary too — and you'll never know whether a 5% difference between runs is the workload changing or just the GPU running 5% faster the second time. Locking eliminates this confound.

***Why this value (1620 MHz)? The reasoning:*** RTX 2000 Ada has a 70 W TDP. Push the clocks too high and it thermal-throttles back, which is exactly the variable behaviour you're trying to avoid. The base clock (~1620 MHz) is the highest sustained frequency that won't throttle under any workload. You're trading peak performance for measurement stability — the right trade for research.

***Why `performance` governor on CPU side:*** same logic — variable CPU clocks add noise.

### 5.9 Sanity check

After all that setup, run a quick check to confirm everything's in place:

```bash
nvidia-smi
nvcc --version
nsys --version
ncu --version
nvidia-smi --query-gpu=clocks.graphics,clocks.memory,clocks.sm --format=csv
cpupower frequency-info | grep "current policy"
nvidia-smi --query-gpu=ecc.mode.current --format=csv
```

***Why this checklist:*** each line tests one thing. If anything's wrong, you find out *now* — not after you've spent three days collecting traces against a misconfigured system.

---

## 6. cuVSLAM installation on Ubuntu

You will do two installs over the project lifetime:

1. PyCuVSLAM wheel (Phase 1, immediate use).
2. cuVSLAM from source with the allocator wrapper (Phase 2 onwards).

***Why two phases:*** the wheel is fast (15 minutes) and lets you start Phase 1 today. But Phase 2 needs you to modify cuVSLAM source (the `TaggedAllocator` in Section 11.2), which requires a source build. Building from source on day one is wasted effort because you don't know yet what to modify. Build the wheel first, learn the codebase, then rebuild from source when you actually need to.

### 6.1 PyCuVSLAM wheel install (Phase 1)

```bash
source ~/venvs/vslam/bin/activate

git clone https://github.com/nvidia-isaac/PyCuVSLAM.git
cd PyCuVSLAM
git lfs install && git lfs pull

pip install <path-to-wheel>.whl
pip install -r examples/requirements.txt

python -c "import cuvslam; print(cuvslam.__version__)"
```

### 6.2 cuVSLAM from-source build (Phase 2 onwards)

```bash
cd ~/research
git clone https://github.com/nvidia-isaac/cuVSLAM.git
cd cuVSLAM
git lfs pull

# Edit CMakeLists.txt: ensure both -lineinfo AND -g are in CUDA / C++ flags.

./build_release.sh
CUVSLAM_BUILD_DIR=$(pwd)/build pip install python/
```

***Why `-lineinfo`:*** adds source-line information to PTX/SASS. ***Cost:*** none — it's just metadata. ***Benefit:*** Nsight Compute can show you, for every instruction that's stalled or missing in cache, exactly which line of CUDA source it came from. Without `-lineinfo` you see opaque SASS addresses; with it you see your code. The single most valuable compile flag for profiling work.

***Why `-g`:*** adds DWARF debug info on the host side. Used by `addr2line` (Section 11.2.2) to convert program-counter values to source file/line. Same intuition as `-lineinfo` but for host-side code.

***Why these don't degrade performance:*** they add metadata for tools, not runtime instrumentation. The binary runs at full speed; the profiler simply has more to look at.

---

## 7. Datasets — getting EuRoC, KITTI, TUM RGB-D, TUM VI

***Why four datasets and not just one:*** each one stresses a different part of the pipeline. EuRoC is the community-standard VIO benchmark — every paper compares on it. KITTI has long outdoor trajectories that exercise the map-size growth. TUM RGB-D's `long_office` sequences re-visit the same desk repeatedly, which is the loop-closure stress test. TUM VI has photometric calibration and AR-glass-like motions. If you characterize on only one, your results are vulnerable to "but does it generalize?" — which is the standard reviewer objection.

| Dataset | Sensor | Purpose |
|---|---|---|
| EuRoC MAV | Stereo + IMU | Community VIO standard; cuVSLAM benchmarked here in [Korovko25] |
| KITTI Odometry | Stereo (no IMU) | Long trajectories, large maps |
| TUM RGB-D | RGB-D Kinect | Loop closure stress |
| TUM VI | Stereo + IMU, photometric | AR/VR motions |

### 7.1 EuRoC MAV

Citation [Burri16]. Stereo MT9V034 at 20 fps + ADIS16448 IMU at 200 Hz, hardware-synchronized. Indoor MAV flight. Sequences: `MH_01_easy` ... `MH_05_difficult`, `V1_xx`, `V2_xx`.

***Why EuRoC is the canonical starter:*** it's small (a few GB per sequence), well-calibrated, has ground truth from a motion-capture system, and every VIO paper for the last decade reports numbers on it. If your cuVSLAM-on-EuRoC numbers don't match the published cuVSLAM numbers, *something is wrong* and you should debug before going further.

```bash
mkdir -p /research/datasets/euroc && cd /research/datasets/euroc
wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip
unzip V1_01_easy.zip
```

### 7.2 KITTI Odometry

Citation [Geiger12]. Stereo grayscale, automotive at 10 fps. 22 sequences (00–21); 00–10 have ground truth. ~22 GB. Registration required.

***Why KITTI even though it has no IMU:*** cuVSLAM works fine in stereo-only mode (`enable_imu_fusion=False`). KITTI's long automotive trajectories (kilometres) push the map to sizes that EuRoC never reaches — exactly what you need to see cold-persistent data behaviour.

### 7.3 TUM RGB-D

Citation [Sturm12]. Kinect 640×480 RGB+depth at 30 Hz; mocap ground truth at 100 Hz.

***Why TUM RGB-D:*** the `freiburg3_long_office_household` sequence has the camera looping around a desk multiple times. This is the most efficient way to force the loop-closure code path repeatedly in a short trace — exactly where you want to see the back-end's memory behaviour.

### 7.4 TUM VI

Citation [Schubert18]. Stereo IDS 1024×1024 at 20 Hz with photometric calibration, Bosch BMI160 IMU at 200 Hz.

***Why TUM VI:*** the *photometric* calibration matters. Photometric calibration models the non-linear response of the camera sensor; without it, the same scene in different lighting looks like different scene to feature matchers. TUM VI is the cleanest controlled dataset for IMU+stereo. AR/VR-typical motions (fast head rotation) are well-represented.

```bash
mkdir -p /research/datasets/tumvi && cd /research/datasets/tumvi
wget -R "index.*" -m -np -nH --no-check-certificate -e robots=off \
    https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16/
cd tumvi/exported/euroc/512_16
md5sum -c *.md5
```

### 7.5 Storage plan with 32 GB system RAM

EuRoC (~30 GB), KITTI (~22 GB), TUM RGB-D (~10 GB), TUM VI 512×512 (~50 GB) ≈ 110 GB. NVBit traces will add 50–200 GB. Provision at least 500 GB of fast NVMe SSD for `/research`; 1 TB is comfortable.

***Why NVMe and not SATA:*** the trace post-processing is I/O-bound. SATA SSD (~500 MB/s sequential) vs NVMe (~3–7 GB/s) makes the difference between hours and minutes for a full-trace pass. You're going to do that many times.

---

## 8. Sanity baseline — running cuVSLAM and computing trajectory error

***Why this section before any profiling:*** you cannot profile a broken system. If cuVSLAM is producing garbage trajectories — bad calibration, time sync issues, frame drops — then *every* memory measurement is unrepresentative of the workload as actually deployed. The discipline is: get to known-good outputs first, then profile.

### 8.1 Run cuVSLAM on EuRoC V1_01_easy

```python
import cuvslam as vslam
from pathlib import Path

rig = vslam.Rig.from_euroc_yaml('/research/datasets/euroc/V1_01_easy/calib.yaml')
cfg = vslam.TrackerConfig(
    odometry_mode=vslam.OdometryMode.MULTI_CAMERA,
    enable_imu_fusion=True,
    enable_observations_export=True,
)
tracker = vslam.Tracker(rig, cfg)

trajectory = []
for stereo_pair, imu_batch, timestamp in load_euroc_stream('/research/datasets/euroc/V1_01_easy'):
    pose = tracker.track(stereo_pair, imu_batch, timestamp)
    trajectory.append((timestamp, pose))

with open('cuvslam_estimate.txt', 'w') as f:
    for t, p in trajectory:
        tx, ty, tz = p.translation
        qx, qy, qz, qw = p.quaternion
        f.write(f"{t} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
```

(The exact names match the official PyCuVSLAM examples; verify against the repo.)

### 8.2 Compute ATE with evo

```bash
evo_ape tum groundtruth_tum.txt cuvslam_estimate.txt -va --align --plot \
    --plot_mode xy --save_results results/V1_01_cuvslam.zip
```

### 8.3 Sanity targets

From [Korovko25]: EuRoC mean position error < 5 cm typical; KITTI Odometry < 1% average translational error.

***Why these specific numbers:*** they're what NVIDIA reports in the technical report. If your numbers are dramatically worse, you have a setup bug. If they're dramatically better, you've probably mis-aligned trajectories or mis-loaded ground truth. Either way, dig in before profiling.

***Debug order (from most to least likely):*** (1) calibration — wrong intrinsics or extrinsics from the YAML; (2) IMU-camera time sync — off by more than a frame and the IMU fusion goes haywire; (3) frame-drop rate — if cuVSLAM is too slow on your hardware it drops frames and accumulates error; (4) GPU clock state — un-locked clocks can produce wildly variable accuracy across runs; (5) ECC errors (`nvidia-smi -q -d ECC`) — last resort but rules out silicon issues.

---

## 9. The profiling toolchain — Nsight Systems, Nsight Compute, NVBit, Accel-Sim

***Why four tools and not one:*** they're complementary. Each answers a question the others can't.

- **Nsight Systems** answers "what's happening, when?" — the *macroscale* view. CPU and GPU timelines, kernel launches, memory transfers, NVTX ranges, OS scheduling. Use it first to understand the shape of a frame.
- **Nsight Compute** answers "for this one kernel, what are the per-counter numbers?" — the *microscale* per-kernel view. DRAM throughput, cache hit rates, occupancy, source-correlated stalls.
- **NVBit** answers "what addresses did each thread touch?" — the *raw trace* view. Nothing summarized; just every load and store with its address.
- **Accel-Sim** answers "given those raw addresses, what would the cache hit rate be in steady state?" — the *simulation* view. Used to get the steady-state numbers that Nsight Compute can't (because of cache flushing between replay passes).

You progress through them in roughly that order: use the macro view to know where to zoom in, the per-kernel view to see what's wrong, the raw trace view to feed the simulator, and the simulator to get defensible steady-state hit rates.

### 9.1 Nsight Systems — the timeline view

```bash
nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=cpu \
    --output=cuvslam_euroc_V1_01 \
    python minimal_euroc.py
nsys stats cuvslam_euroc_V1_01.nsys-rep
```

***Why `--trace=cuda,nvtx,osrt`:*** cuda for the GPU side, nvtx for your inserted range annotations, osrt for OS-level events (page faults, scheduler decisions). Adding more (`cublas`, `cudnn`) inflates the trace without giving useful information for cuVSLAM.

***Why NVTX ranges are worth the invasion of cuVSLAM source:*** without them, the Nsight Systems timeline is a sea of unlabeled kernel launches. With them, each logical stage of cuVSLAM lights up as a labelled block. The difference between "interpretable visualization" and "wall of color" is just a few `nvtxRangePush/Pop` calls.

### 9.2 Nsight Compute — per-kernel deep dive

```bash
ncu --set full \
    --target-processes all \
    --launch-skip 100 --launch-count 50 \
    -o cuvslam_kernels \
    python minimal_euroc.py
```

***Why `--launch-skip 100`:*** the first 100 kernel launches are warm-up. The first ten launches of any GPU workload have variable timing because driver, library, and allocator state is still being warmed. Skip them.

***Why `--launch-count 50`:*** each kernel is *replayed* multiple times by Nsight Compute to collect all the metrics. Profiling 50 kernels is already slow; profiling 5,000 takes hours and adds no information because cuVSLAM repeats the same kernels.

**Steady-state caveat:** Nsight Compute flushes L1/L2 between replay passes for determinism. Reported cache hit rates are therefore *lower* than steady-state. ***Why the flush:*** to make every measurement of "kernel X with input Y" identical regardless of what ran before. ***Why this hurts you:*** in real execution, kernel X often runs against caches that were warmed by kernel X-1. Nsight Compute's flush destroys that warming, so the hit rate you see is closer to "cold-start hit rate" than to "steady-state hit rate". For absolute hit rates, use Accel-Sim. For *relative* comparisons among kernels in the same replay pass, hit rates are fine.

Use **Range Replay** instead of Kernel Replay when a stage has intra-stage cache dependency (e.g. local BA). ***Why Range Replay:*** Range Replay replays a whole logical block (set by NVTX ranges) without flushing internally — only between range repetitions. This preserves intra-stage cache state, which is what you want when a stage's first kernel warms the cache for its second kernel.

**Reusable parser and metric taxonomy.** Don't write your `.ncu-rep` parser from scratch.

***Why not from scratch:*** parsing `ncu` output is tedious — there are dozens of unit conversions (`Kbyte/second` to `byte/second`, `cycle/nsecond` to `cycle/second`, etc.) and the metric names are long and easy to typo. Someone has done this work cleanly for you.

Fork the one from Jiashen Cao's GPU-database characterization study [Cao23]: https://github.com/jiashenC/gpudb-char-and-opt. The files you want:

- `utility/counter_config.py` — a hand-curated taxonomy of Nsight Compute metric IDs grouped into eight functional buckets (`metric_sol()`, `metric_roofline()`, `metric_memory()`, `metric_compute()`, `metric_occupancy()`, `metric_launch()`, `metric_warp()`, `metric_inst()`). Adopt verbatim.
- `report_parser/ncu_parser.py` — parses `.ncu-rep` reports into structured Python, handles unit conversions.
- `stats/ncu_export.py` and `stats/flush_ncu_csv.py` — aggregate metrics across kernels into CSV.

***Why this works across projects:*** the metric names in `counter_config.py` are NVIDIA-defined, not application-defined. They mean the same thing whether you're profiling a GPU database query or a SLAM kernel. Cao's grouping is just a sensible curation of which subset to actually look at.

Licence-check before forking; cite [Cao23] when you use it.

### 9.3 NVBit — binary instrumentation for raw memory traces

NVBit gives you full per-warp memory address traces, kernel-launch arguments, and arbitrary instrumentation injection — without recompiling cuVSLAM. https://github.com/NVlabs/NVBit. Citation [Villa19].

***Why NVBit exists conceptually:*** Nsight Compute aggregates. NVBit doesn't — it gives you the raw event stream. The aggregate hides patterns; for instance, you can't compute reuse distances from "L2 hit rate = 78%". You need the actual sequence of addresses. NVBit is the cheapest way to get that sequence on an unmodified application.

***Why NVBit's "no recompilation" matters:*** cuVSLAM is a complex library with many translation units. Recompiling it to instrument each load and store would be invasive, fragile, and slow. NVBit instruments at the SASS level (the GPU's native assembly), at load time, which means the user-space binary doesn't change. You attach the instrumentation library at runtime via an environment variable.

Tools you'll use:

- `mem_trace` (shipped with NVBit) — streams `<kernel_id, warp_id, instr_pc, addr, size, R/W>` per global memory op.
- A custom tool you'll build (Section 11) that *also* captures `cuMemAlloc` calls and kernel arguments.

Install and run:

```bash
cd ~/research
wget https://github.com/NVlabs/NVBit/releases/download/<latest>/nvbit-Linux-x86_64-<ver>.tar.bz2
tar xjf nvbit-Linux-x86_64-<ver>.tar.bz2
cd nvbit_release
make -C tools/mem_trace

CUDA_INJECTION64_PATH=$(pwd)/tools/mem_trace/mem_trace.so \
    python minimal_euroc.py | zstd -1 > memtrace.zst
```

***Why `CUDA_INJECTION64_PATH` rather than `LD_PRELOAD`:*** `LD_PRELOAD` is a generic dynamic-linker mechanism that works at process load time. `CUDA_INJECTION64_PATH` is a CUDA-driver mechanism that hooks into the driver's initialization specifically. The latter is cleaner because (a) it doesn't pollute non-CUDA processes and (b) it integrates with the CUDA driver's view of the world (it knows about CUDA contexts, streams, etc., which `LD_PRELOAD` would not).

***Why piping straight to `zstd`:*** if you wrote uncompressed to disk you'd consume 5–10× more storage. Compressing in the same shell pipeline means the uncompressed bytes never touch disk at all.

Practical notes:

- 5–50× wall-clock overhead — ***why:*** every memory instruction now has a small NVBit-injected snippet before it that copies the address into a CPU-visible buffer. That's a lot of extra work.
- Never measure performance with NVBit attached.
- For `sm_89` every recent NVBit release works.

### 9.4 Accel-Sim — cache simulation for steady-state hit rates

Accel-Sim consumes NVBit-produced SASS traces and feeds them into GPGPU-Sim 4.0 with a configurable cache hierarchy. https://accel-sim.github.io. Citation [Khairy20].

***Why a simulator at all, when we have real hardware:*** because Nsight Compute's flush distorts cache state (Section 9.2). The simulator doesn't flush. It models a configurable cache hierarchy and runs your traces through it deterministically. Whatever hit rate it reports is what would happen if the real cache behaved exactly as configured.

***Why Accel-Sim and not gem5:*** gem5 has had CUDA support added and removed and re-added over the years; it's not the natural place to do GPU work. Accel-Sim is purpose-built, actively maintained, and the standard tool in the GPU-architecture research community. Reviewers know it.

```bash
cd ~/research
git clone https://github.com/accel-sim/accel-sim-framework.git
cd accel-sim-framework
git clone https://github.com/accel-sim/gpgpu-sim_distribution.git ./gpu-simulator
```

The pain is that Accel-Sim ships validated configs for `sm_70` (Volta), `sm_75` (Turing), `sm_86` (Ampere), but not for `sm_89` (Ada). You'll build a custom AD107 config in Section 12.1(c).

### 9.5 Reproducibility checklist

1. GPU clocks locked.
2. CPU governor `performance`.
3. ECC enabled.
4. No other GPU jobs.
5. Identical input sequence and frame range.
6. Discard the first N frames as warm-up.
7. Repeat ≥ 5 times; report mean and 95th percentile.

***Why 95th percentile and not max:*** real systems have rare outliers (an OS preemption, a thermal event) that have nothing to do with the workload. Max is dominated by those outliers. 95th percentile gives you the worst typical case without being hostage to noise.

***Why ≥ 5 runs and not 3:*** with 3 runs you can't really see the distribution. With 5+ you can compute a meaningful variance and detect outliers. The marginal cost of more runs is small; do them.

---

## 10. Phase 1 — Decomposing cuVSLAM into a memory-hierarchy-aware DAG

***Why this is Phase 1 and not Phase 2:*** you cannot characterize what you cannot name. Before you measure, you decompose. Phase 1 is a *paper* exercise (well, mostly): read the cuVSLAM technical report, read the source, draw the data-flow graph, label each node with what it does and what data it touches. This artifact — the annotated DAG — is what every later phase indexes against.

**Goal.** Annotated dataflow graph: each node is a logical stage, labelled with CUDA kernel name(s), working-set size (as a function of inputs), R/W ratio (qualitative), and persistence class.

**Steps.**

1. Read the cuVSLAM technical report end-to-end [Korovko25].
2. Browse cuVSLAM source. Identify CUDA kernels (`__global__`, `cuvslam::launch`). Build a spreadsheet: kernel name → logical stage.
3. Run cuVSLAM under `nsys` and read the timeline. The kernel-launch order per frame *is* the dataflow graph in execution order. ***Why the timeline matches the graph:*** in a single CUDA stream, kernels run strictly in launch order. Multi-stream changes this — and you'll need to disambiguate using NVTX ranges or stream IDs.
4. Insert NVTX ranges around each logical stage. Rebuild from source.

**The three persistence classes** — recap with the *why* for each label:

- **Streaming.** Working set scales with image size only. Read once, processed, discarded. ***Why this matters for hardware:*** if the data is touched once and thrown away, there's no reuse for any cache to capture; pre-fetch and consume on-the-fly. Near-sensor SRAM is the natural fit.
- **Hot-persistent.** Small (~1–10 MB), accessed every frame. ***Why this matters:*** small enough to fit in a banked PiM unit, accessed often enough that PiM amortizes its activation cost. The classic PiM target.
- **Cold-persistent.** Large (~100s of MB), accessed only on loop closure or session resume. ***Why this matters:*** too big to fit in DRAM efficiently; too cold to justify the DRAM seat. Computational storage (ISP) lets you keep this state on the SSD and only "wake it up" on the rare loop event.

Classify qualitatively here; Phase 3 measurements will validate or revise.

**Deliverable.** A DAG (any diagram tool) plus a 2-page written description. Time budget: 3–4 weeks.

***Why 3–4 weeks for what sounds like a paper exercise:*** the cuVSLAM source is substantial (tens of thousands of lines). You'll spend more time than you expect chasing function calls through templates and verifying that what you wrote in your spreadsheet matches what actually launches. Don't compress this.

---

## 11. Phase 2 — Building the profiling testbed + address-to-data-structure mapping

***Why two distinct things in one phase:*** the testbed (the boring infrastructure) and the mapping pipeline (the novel methodology) interlock. The testbed runs your profilers with versioned outputs; the mapping pipeline post-processes those outputs into labelled traces. Build them together because each one's design constrains the other.

### 11.1 The reproducible profiling harness

```
/research/cuvslam-profiling/
├── cuvslam/                  # cuVSLAM source, NVTX-annotated, allocator-wrapped
├── datasets/                 # symlinks to /research/datasets/...
├── env/                      # frozen pip requirements, CUDA version, driver
├── ncu_tooling/              # forked from jiashenC/gpudb-char-and-opt
├── sieve/                    # Sieve clustering + representative-invocation selection
├── scripts/
│   ├── lock_clocks.sh
│   ├── run_nsys.sh
│   ├── run_ncu.sh
│   ├── run_sieve_pass.sh
│   ├── run_nvbit_memtrace.sh
│   ├── run_accelsim_trace.sh
│   └── postprocess/
├── results/
│   └── 2026-MM-DD_<dataset>_<sequence>_<profiler>/
│       ├── metadata.json
│       ├── raw/
│       └── derived/
└── notebooks/
```

***Why this layout:*** four design rules. (1) Every run produces a versioned directory under `results/` — no run ever overwrites another. (2) Each run's `metadata.json` is non-negotiable; six months in, you will not remember what produced which graph. (3) Raw outputs and derived analyses are separate folders — you can re-derive without re-collecting. (4) Notebooks are read-only consumers of `results/`, never producers. This separation is what lets you trust a graph six months later.

***Why this isn't over-engineered:*** research code that doesn't have these properties drifts into a state where no result is reproducible and you spend Friday afternoons doing "let me re-run this to figure out where the number came from." Two weeks of harness work pays for itself by month four.

**Frame-range conventions** (pick once, stick with them):

- *Steady-state window:* frames 200–250 of EuRoC `MH_01_easy`.
- *Loop-closure window:* frames around a known loop event in `MH_05_difficult` or TUM RGB-D `long_office_household`.
- *Cold-start window:* frames 0–50.

***Why three windows and not random sampling:*** the three windows correspond to qualitatively distinct behaviour. Frames 0–50 have cold caches and a tiny map; frames 200–250 are steady-state; loop-closure frames trigger the rare back-end path. Random sampling would dilute these signals and make your "what's different about loop closure?" question harder to answer.

### 11.2 Address-to-data-structure mapping — the methodology

The problem: NVBit gives raw addresses (`0x7f1234567000`). The thesis needs labels (`covisibility_graph`, `keyframe_descriptors`). You map between them with a three-layer pipeline.

***Why three layers and not one:*** redundancy across independent signals. Each layer can have bugs or gaps; if all three agree, you trust the label; if they disagree, you've caught a bug. Layer 1 (source-side) is your primary signal but depends on you correctly auditing every allocation site. Layer 2 (driver-side) catches allocations you missed in the audit. Layer 3 (kernel-arg correlation) verifies that the labels actually correspond to what the kernels touch. Defence-in-depth.

#### 11.2.1 Layer 1 — Allocator wrapping (source-side, primary signal)

cuVSLAM is open source. Modify it to route every internal `cudaMalloc` through a wrapper:

```cpp
// cuvslam/src/memory/tagged_allocator.hpp
#pragma once
#include <cuda_runtime.h>
#include <string>
#include <mutex>
#include <vector>

namespace cuvslam {

struct AllocRecord {
    void*       base;
    size_t      size;
    std::string tag;
    int64_t     timestamp_us;
};

class TaggedAllocator {
public:
    static TaggedAllocator& instance();
    cudaError_t alloc(void** ptr, size_t size, const std::string& tag);
    cudaError_t free(void* ptr);
    void dump_table(const std::string& path) const;

private:
    mutable std::mutex            mu_;
    std::vector<AllocRecord>      live_;
    std::vector<AllocRecord>      history_;
};

#define CUVSLAM_ALLOC(ptr, size, tag) \
    cuvslam::TaggedAllocator::instance().alloc((void**)(ptr), (size), (tag))

}
```

***Why a singleton:*** allocations happen from anywhere in the codebase, but they all need to end up in the same table. Singleton with a mutex is the simplest pattern that works.

***Why a fixed vocabulary of tags, defined upfront:*** if every developer (or every audit pass) invents their own tag names, downstream analysis devolves into string-matching hell. Fix the vocabulary, document it, treat new tags as a change request.

Suggested vocabulary:

```
images_raw, images_undistorted, pyramid_levels,
features_current, features_previous, descriptors_current, descriptors_previous,
stereo_matches, ransac_scratch, pose_state,
local_map_landmarks, local_map_observations,
covisibility_edges, covisibility_adjacency,
keyframe_poses, keyframe_descriptors, keyframe_landmarks_idx,
bow_vocabulary, bow_inverted_index, loop_candidates,
ba_jacobian, ba_hessian, ba_residuals, ba_solver_scratch,
imu_buffer, imu_preintegrated,
graph_optimizer_state, misc_scratch
```

***Why this exact vocabulary:*** it covers cuVSLAM's modules at the granularity where the memory behaviour is interesting. Coarser ("front_end_state", "back_end_state") collapses meaningful distinctions; finer ("frame_42_landmark_observations") makes the downstream tables unreadable.

#### 11.2.2 Layer 2 — NVBit `cuMemAlloc` interception (independent verification)

In your NVBit tool, also subscribe to CUDA driver allocation events:

```cpp
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    if (!is_exit) return;

    if (cbid == API_CUDA_cuMemAlloc_v2 || cbid == API_CUDA_cuMemAllocManaged) {
        auto* p = (cuMemAlloc_v2_params*)params;
        emit_alloc_record(*p->dptr, p->bytesize, host_callstack());
    }
    if (cbid == API_CUDA_cuMemFree_v2) {
        auto* p = (cuMemFree_v2_params*)params;
        emit_free_record(p->dptr);
    }
}
```

***Why instrument both Layer 1 and Layer 2:*** Layer 1 only sees allocations you've routed through `CUVSLAM_ALLOC`. If a third-party library that cuVSLAM links against (cuBLAS, cuSPARSE, or some helper) does its own `cudaMalloc`, Layer 1 won't see it, but Layer 2 will. The diff between the two tells you exactly which allocations happen outside your audit.

Resolve PCs to source locations with `addr2line` on the cuVSLAM library (built with `-g`):

```bash
addr2line -e libcuvslam.so -f -C 0x12ab34
```

#### 11.2.3 Layer 3 — Kernel-argument correlation (refinement, cheap)

```cpp
if (cbid == API_CUDA_cuLaunchKernel) {
    auto* p = (cuLaunchKernel_params*)params;
    record_launch(p->f, p->kernelParams);
}
```

***Why this is a separate signal worth collecting:*** even if Layers 1 and 2 say "this allocation is `covisibility_edges`", they don't prove that a given access actually *uses* the allocation as `covisibility_edges` data — only that it falls within that allocation. By checking whether the access address matches a pointer that was *passed as an argument to the currently-running kernel*, you confirm the data was being used by code that semantically expects it. This is what MSched [Tang26] and cuThermo [Yu25] do.

#### 11.2.4 Post-trace join

```python
import pandas as pd
from intervaltree import IntervalTree
import zstandard as zstd
import pyarrow as pa
import pyarrow.parquet as pq

allocs = pd.read_csv('alloc_table.csv')
tree = IntervalTree()
for _, r in allocs.iterrows():
    tree[r['base']:r['base'] + r['size']] = r['tag']

def parse_line(line):
    fields = line.decode().strip().split(',')
    return {
        'kernel_id': int(fields[0]),
        'warp_id':   int(fields[1]),
        'pc':        int(fields[2], 16),
        'addr':      int(fields[3], 16),
        'size':      int(fields[4]),
        'rw':        fields[5],
    }

batch = []
out_path = 'labeled_trace.parquet'
writer = None

with open('memtrace.zst', 'rb') as fz:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(fz) as reader:
        buf = b''
        while True:
            chunk = reader.read(1 << 20)
            if not chunk: break
            buf += chunk
            *lines, buf = buf.split(b'\n')
            for line in lines:
                rec = parse_line(line)
                hits = tree[rec['addr']]
                rec['tag'] = next(iter(hits)).data if hits else 'unmapped'
                batch.append(rec)
                if len(batch) >= 1_000_000:
                    table = pa.Table.from_pandas(pd.DataFrame(batch))
                    if writer is None:
                        writer = pq.ParquetWriter(out_path, table.schema)
                    writer.write_table(table)
                    batch.clear()

if batch:
    table = pa.Table.from_pandas(pd.DataFrame(batch))
    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema)
    writer.write_table(table)
if writer is not None:
    writer.close()
```

***Why batching at 1M records:*** writing one row at a time to Parquet is slow; writing the whole trace in one shot blows out RAM. 1M is a sweet spot — large enough to amortize Parquet's per-batch overhead, small enough to fit comfortably in RAM with all the other process state.

***Why an interval tree and not a sorted list with binary search:*** allocations can overlap (rare but possible — e.g. one allocation freed and the same address re-allocated). Interval trees handle this correctly. Binary search assumes disjoint intervals and will silently give wrong answers when they overlap.

***Why the `'unmapped'` bucket should be small:*** it's your sanity check on Layer 1's coverage. If it's > a few percent, fix the audit. If it's < 1%, you're seeing only driver-side scratch.

#### 11.2.5 Time budget

- Layer 1: 2–3 weeks.
- Layer 2: 1 week.
- Layer 3: 3–5 days.
- Pipeline + validation: 1 week.

Total: 4–5 weeks.

***Why this is worth 4–5 weeks:*** without this pipeline, every Phase 3 result is "L2 hit rate is 78%". With it, every result is "L2 hit rate for `covisibility_edges` data is 78%, but for `keyframe_descriptors` data is 22%." The latter is what makes the thesis defensible. Without semantic labels, you have summary statistics; with them, you have a story.

### 11.3 Trace-handling for 32 GB RAM

```bash
CUDA_INJECTION64_PATH=./cuvslam_mem_trace.so \
CUVSLAM_TRACE_FRAME_START=200 \
CUVSLAM_TRACE_FRAME_END=230 \
python minimal_euroc.py 2> /dev/null | zstd -1 > trace_V1_01_200_230.zst
```

***Why env-var gating rather than recompilation:*** you'll iterate on which frame ranges to profile. Env vars let you change the window without rebuilding the NVBit tool. Recompilation would cost minutes per iteration; env vars cost seconds.

### 11.4 Representative kernel-invocation selection — the Sieve methodology

You have a problem v3 of this document did not address: cuVSLAM launches *thousands* of kernels per second. Across a 30-frame profiling window, that's tens of thousands of kernel invocations. Even with `zstd` and 1 TB of disk, you cannot afford to NVBit-trace every invocation, and even if you could, Accel-Sim wouldn't finish simulating them this decade.

***The reasoning behind sampling at all:*** most kernel invocations are *redundant*. The same kernel runs 50 times in a frame with similar inputs and produces similar memory behaviour each time. Tracing all 50 gives you ~50× the data for ~zero new information. You want to sample wisely — picking a small set whose behaviour is statistically representative of the full population.

You need a principled way to pick representative kernel invocations to trace and simulate. **Sieve** [Naderan23] does exactly this. It's stratified sampling of GPU kernel invocations, with statistical bounds.

***Why "stratified" and not "random":*** random sampling from a heterogeneous population under-represents rare strata. If 99% of invocations are `feature_detect_kernel` and 1% are `loop_closure_kernel`, random sampling gives you mostly the former and almost none of the latter — but the rare one is exactly what you care about characterizing. Stratified sampling first partitions invocations into strata of similar behaviour, then samples each stratum proportionally (or oversamples the rare strata). This is the same statistical principle that makes opinion polls work.

**The Sieve recipe** (from https://github.com/gpubench/sieve, ISPASS 2023 paper):

1. Run the application end-to-end and collect lightweight per-invocation summaries (kernel name, occupancy proxy, IPC proxy, memory traffic proxy — cheap counters that don't require NVBit).
2. *Stratify* invocations: same kernel name + similar summary statistics → same stratum.
3. Within each stratum, sample one (or a small number of) invocations — these are the "representatives".
4. NVBit-trace and Accel-Sim-simulate only the representatives.
5. Reconstruct full-population behaviour by weighting each representative by its stratum size.

For cuVSLAM concretely:

- `cuvslam_feature_detect_kernel` may launch 50 times per frame × 30 frames = 1500 invocations. Sieve might find that 5 strata cover this population (e.g. one per pyramid level), so you trace only 5 representatives instead of 1500.
- `cuvslam_local_ba_solver_kernel` has few invocations but each is heavyweight. Sieve keeps them all because the stratum size is already small.

**Where it sits in your pipeline:**

```
Phase 2 frame window selected (Section 11.1)
        ↓
Lightweight per-invocation profiling (cheap Nsight Compute pass)
        ↓
Sieve clustering → representative invocation IDs
        ↓
NVBit instrumented pass (Section 9.3) — emits ONLY at representative invocations
        ↓
Accel-Sim simulation (Section 12.1(c))
        ↓
Re-weight results by stratum sizes
```

The "emits ONLY at representative invocations" gate is implemented with the same env-var pattern you use for frame ranges.

***Reading priority:*** add the Cactus paper [Naderan21] and the Sieve paper [Naderan23] to your literature pile *before* you start the NVBit trace pass.

***Time-budget impact:*** Sieve adds ~1 week to Phase 2 but saves ~3–4 weeks in Phase 3. Net savings: 2–3 weeks plus much more defensible numbers.

---

## 12. Phase 3 — Quantitative memory characterization

***Why this is the "real" research phase:*** Phases 0–2 are infrastructure. Phase 3 is where you produce numbers that go in a paper. Your goal here is artifacts that survive peer review — every number must be reproducible, every claim must be backed by measurement, every methodology must be defensible.

### 12.1 The four artifacts

#### (a) Hierarchical roofline placement

For each major cuVSLAM stage, plot it on a hierarchical roofline (L1 / L2 / DRAM bandwidth ceilings + peak FLOP/s). Methodology: [Yang20]. Machine characterization: NERSC Empirical Roofline Toolkit (ERT). Application characterization: Nsight Compute metrics.

***Why hierarchical and not classical:*** the classical roofline has one bandwidth ceiling (DRAM). The hierarchical version adds L1 and L2 ceilings. Why? Because a kernel can be *L2-bandwidth-bound* — its arithmetic per byte is high enough to be DRAM-unbound but not high enough to hit the L2 ceiling. Without the L2 line on the plot, that kernel looks "memory-bound" and you can't tell which level. With it, you do.

**RTX 2000 Ada-specific ceilings:**

- Peak FP32: 12.0 TFLOPS.
- L2 bandwidth ceiling: measure with ERT.
- DRAM bandwidth ceiling: 224 GB/s (validate with ERT; empirical maxes are 80–90% of spec).

***Why measure ceilings with ERT rather than trust the spec:*** vendor specs are theoretical peaks under ideal conditions. ERT runs microbenchmarks that get as close to peak as actual silicon allows. The ERT-measured value is what your kernels can *actually* achieve; the spec value is the impossible-to-reach ceiling. Compare your kernels to what's achievable.

#### (b) Per-kernel access-pattern fingerprint

For each profiled kernel, classify exactly one of:

| Class | Description | PiM payoff | ISP payoff |
|---|---|---|---|
| Dense regular | Image conv, pyramid | Low | None |
| Strided gather-scatter | Descriptors at sparse keypoints | Moderate | None |
| Indirect pointer-chasing | Map → covis graph → keyframe → 3D points | High | None |
| Random hash lookup | BoW vocabulary / loop closure | Ideal | Moderate |
| Sequential bulk | Keyframe serialization | Low | Ideal |

***Why these five classes:*** they're the standard taxonomy from the data-movement and PiM literature ([Oliveira21], [Gomez-Luna22]). Dense regular and sequential bulk lie at opposite ends — both have high spatial locality but very different reuse. Strided gather-scatter has spatial locality only along the stride. Indirect pointer-chasing has no spatial or temporal locality but each access depends on the previous. Random hash lookup is the worst case: no locality of any kind.

***Why PiM/ISP payoff varies across classes:*** PiM benefits scale with how much *data movement* you're saving. Dense regular data is already streamed efficiently by the GPU's prefetcher — PiM saves little. Random hash lookups are the opposite: every access is a DRAM trip, and PiM eliminates the round-trip to the CPU/GPU. ISP is the same logic at the storage level — sequential scans through huge datasets benefit, scattered small reads don't.

Classification method: read the source, then *confirm with the NVBit address trace* — compute stride histograms and gather/scatter signatures numerically. Don't rely on intuition.

#### (c) Reuse-distance histograms — and the cache-simulation pipeline

***Why reuse distance is the right metric:*** cache hit rate depends on cache size; reuse distance is a property of the *access pattern alone*. If you know the reuse-distance distribution, you can compute the hit rate for any cache size by integrating up to that size. It generalizes across hardware.

**Two ways to get hit rates:**

- *Nsight Compute counters* — fast, easy, but underestimate steady-state hit rate.
- *Accel-Sim simulation* — slow, fiddly, but accurate for steady-state.

**Step 1 — Compute reuse-distance directly from NVBit traces.** Cutoffs on your hardware:

- Reuse distance < 1 MB → fits in L1.
- Reuse distance < 12 MB → fits in L2.
- Reuse distance > 12 MB → DRAM access.
- Reuse distance > 16 GB → unified-memory paging.

***Why these cutoffs are not generic:*** they're *your hardware's*. Cite the RTX 2000 Ada spec. A reviewer can verify them against the published architecture; with generic numbers they couldn't.

**Step 2 — Build an Accel-Sim config for AD107.** Accel-Sim ships configs for Volta, Turing, Ampere. Ada (AD107) is not officially supported. You build one:

```bash
cd ~/research/accel-sim-framework
mkdir -p configs/tested-cfgs/SM89_RTX2000Ada
cp configs/tested-cfgs/SM75_RTX2060/gpgpusim.config \
   configs/tested-cfgs/SM89_RTX2000Ada/gpgpusim.config
```

Edit to: 22 SMs, 12 MB L2, 128 KB per-SM L1, GDDR6 at 224 GB/s, 128-bit bus, locked-clock domains.

***Why start from Turing and not Ampere:*** Turing (TU102) is the closest architectural ancestor to Ada in terms of cache hierarchy structure (banked L2, similar L1 organization). Ampere introduced async copy paths and the unified L1/shared-memory partition, which Ada inherits but with different sizes. Adapting from Turing requires fewer structural changes.

**Step 3 — Calibrate against real hardware.**

Build a micro-benchmark suite of known-AI kernels and run them both on your RTX 2000 Ada (Nsight Compute) and through Accel-Sim. Tune the config until per-kernel cycles, DRAM bytes, and L2 hit rate agree within ~5%.

***Why calibrate at all:*** because your config is community-built, not officially validated. A community config can be off in numerous subtle ways (cache associativity, banking, replacement policy, throttling logic). Without calibration you have no idea how off it is. With calibration against microbenchmarks of known behaviour, you have a credible error bar.

***Why microbenchmarks and not cuVSLAM itself for calibration:*** cuVSLAM is complex; if simulated and real disagree, you can't tell whether the disagreement is in the cache model, the warp scheduler, the memory controller, or somewhere else. Microbenchmarks exercise one mechanism at a time. Calibrate each mechanism independently, then trust the composite on cuVSLAM.

**Step 4 — Report deltas, not absolutes.** Reviewers will accept:

- "L2 hit rate for kernel `cuvslam_local_ba_kernel` is 14 points lower than for `cuvslam_feature_detect_kernel`."
- "Pinning the covisibility graph into L2 raises that kernel's hit rate by 22 points."

They will not accept:

- "The absolute L2 hit rate is 71.3%."

***Why deltas survive review and absolutes don't:*** an absolute number from an unvalidated simulator is just an opinion of the simulator. A delta — "X minus Y" — has any systematic simulator error cancel out. If the simulator overestimates absolute hit rate by 8 points, it overestimates both X and Y by 8 points, and the delta is unchanged. This is a general principle in simulation methodology: when you can't validate absolute numbers, *the right play is to design your experiments around comparisons*.

#### (d) Per-stage bandwidth breakdown

Bar chart: one bar per cuVSLAM stage, height = average GB/s from DRAM, error bar = 95th percentile. Compute from Nsight Compute `dram__bytes_read.sum + dram__bytes_write.sum` per kernel, attributed to a stage via NVTX ranges.

***Why this chart matters disproportionately:*** it's the one that goes on slide 2 of your defense. It shows *at a glance* which stages move the most data, which is the visual core of the "memory wall" argument. Everything else in the paper backs up this one chart.

### 12.2 Sweeps to run (using only real data — see Section 14)

| Variable | How to sweep without leaving real data | What you learn |
|---|---|---|
| Image resolution | Downsample real frames (VGA / 720p / 1080p) | Streaming working-set scaling |
| Sequence length | Truncate or concatenate real sequences | Cold-persistent working-set growth |
| Loop-closure rate | Pick sequences with different revisit patterns | Where back-end traffic spikes |
| Frame rate | Decimate real frames (10 / 30 / 60 Hz) | Temporal pressure on front-end |
| Feature density | cuVSLAM's `max_features` parameter | Front-end working set scaling |
| Map size pressure | Disable / enable map serialization | When you cross the 16 GB VRAM threshold |

***Why these specific sweeps:*** each isolates one variable that the thesis predicts will move a working set across a hardware boundary. Image resolution scales the streaming working set linearly — at what point does it stop fitting in L2? Sequence length grows the cold-persistent state — at what point does it spill to host memory? Feature density inflates the front-end gather load — at what point does that gather become the bottleneck? These are not random sweeps; each one is designed to make a *specific* effect appear.

### 12.3 DAMOV-style classification

Apply the DAMOV taxonomy [Oliveira21] to your cuVSLAM kernels. Output: a table — *cuVSLAM stage → DAMOV bottleneck class → PiM/ISP affinity*. This is the synthesis chart.

### 12.4 Phase 3 expected output

A 15–25 page workload-characterization technical report containing:

- Annotated DAG (refined).
- Hierarchical roofline plots.
- Per-stage access-pattern fingerprints.
- Reuse-distance histograms for top kernels.
- Calibrated Accel-Sim hit-rate deltas.
- Bandwidth breakdown chart.
- DAMOV-style classification table.
- A shortlist of PiM/ISP candidates with rationale.

***Why this size is right:*** an ISPASS or IISWC submission is typically 10–12 pages double-column. A 15–25 page internal report compresses to that range; the extra pages are appendix material (calibration data, detailed methodology) that reviewers will check.

---

## 13. Phase 3.5 — Extension to Jetson AGX Orin

When workstation profiling stabilizes, port the entire harness to a Jetson AGX Orin and re-run.

***Why Jetson matters for the thesis:*** the workstation result, on its own, is suggestive but not definitive. A reviewer can always say "you measured on a discrete GPU; would it look the same on the deployment platform?" By repeating the characterization on Jetson — which has a *unified* memory architecture, different cache sizes, and different bandwidth — you answer that question affirmatively. If the three-class taxonomy survives the discrete-to-unified shift, it's clearly a property of the workload, not of the hardware.

### 13.1 Why Jetson is interesting

Jetson AGX Orin:

- **Unified memory architecture** — CPU and GPU share the same LPDDR5 pool. No discrete VRAM; no PCIe transfer.
- **GPU: Ampere `sm_87`** — same generation family as A100.
- **DRAM: LPDDR5** at up to 204.8 GB/s (32 GB variant) — within range of your RTX 2000 Ada.
- **Power budget: 15–60 W** configurable.
- **The exact deployment target for cuVSLAM as marketed by NVIDIA.**

***Why the unified memory angle is interesting scientifically:*** PiM proposals usually assume a host↔device split (CPU + DRAM-with-PiM, or GPU + HBM-with-PiM). Unified memory blurs that line — there is no "host vs device" memory; there's just one pool. Does PiM still make sense in that world? This is an open question, and your characterization on Jetson provides empirical input to it.

### 13.2 What to redo on Jetson

- cuVSLAM build for aarch64 + Jetson CUDA.
- Phase 2 + Phase 3 measurement pipeline, identical methodology.
- *Unified-memory-specific analysis:* where do allocations end up (CPU-side, GPU-side, migrated)? `cudaMemRangeGetAttribute` and the `pageable_memory_access` device property.
- Re-do roofline with Jetson LPDDR5 bandwidth ceiling.
- Re-do reuse-distance with Jetson L2 size.

### 13.3 What translates without rework

- The address-to-data-structure mapping pipeline. Same allocator wrapper, same vocabulary.
- The dataset suite.
- The DAMOV classification framework.

***Why portability of the pipeline is a feature:*** it means the workstation work isn't throwaway. Every hour you invested in Phase 2 pays off again on Jetson.

### 13.4 Time budget

3–4 weeks if the workstation harness is solid.

### 13.5 Tooling adjustments for Jetson

- **Nsight Systems** — works identically.
- **Nsight Compute** — works, but some metric sets are reduced on Jetson.
- **NVBit** — supported on aarch64 since NVBit 1.5.
- **Accel-Sim** — your AD107 config does *not* port to Jetson Orin's GPU. Either build a separate `sm_87` config (more work) or rely on Nsight Compute counters (lower fidelity). State the choice in the writeup.

***Why state the choice explicitly:*** reviewers will ask, and the right answer is "we chose X for reason Y; if the result held up under Y's limitations, it would hold up under stronger ones." That's defensible. "We did X" without the why is not.

---

## 14. Why Isaac Sim is deferred out of Phases 0–3

Earlier-version plans suggested Isaac Sim for sensitivity sweeps. That introduces a confound.

**The problem.** Isaac Sim's rendered images have different noise characteristics, sub-pixel feature distributions, and motion-blur profiles than real cameras. Sweeping a parameter in Isaac Sim and observing a memory-behaviour change conflates *the parameter sweep* with *the sim-vs-real gap*.

***The general principle this illustrates:*** when an experimental design conflates two variables, the result has zero scientific value, regardless of how good either variable's measurement is. The fix is to either remove one variable or measure it independently. Section 12.2 picks "remove" — use only real data with real cuVSLAM knobs.

**The fix in this plan.** Section 12.2 sweeps every variable you need using real data + cuVSLAM's own configuration knobs.

**When Isaac Sim could return.** Phase 4+ may need extreme regimes that no real dataset captures (e.g. 32-camera rigs, 24-hour continuous mapping). At that point Isaac Sim is appropriate *if* you first publish a "reality alignment" sub-study showing that a matched Sim sequence reproduces real-data memory behaviour within X%.

***Why a reality-alignment sub-study is itself a research contribution:*** it answers a question the community cares about — when does simulated data faithfully reproduce hardware-level behaviour? This is a missing piece in the simulation literature; doing it well would be publishable on its own.

---

## 15. Deliverables and milestones

| Month | Milestone | Concrete artifact |
|---|---|---|
| 1 | Literature foundation | 5–10 page literature synthesis identifying the gap |
| 1.5 | Workstation operational | cuVSLAM runs on EuRoC `V1_01_easy` with ATE < 10 cm |
| 2 | Phase 1 complete | Annotated DAG with 3-class persistence tagging |
| 3 | Profiling harness | Run scripts, NVTX-annotated cuVSLAM, versioned results dir, forked ncu_tooling/ |
| 4 | **Mapping pipeline complete** | Labelled traces with < 5% `unmapped` |
| 4.5 | **Sieve clustering operational** | Trace volume cut by ≥ 30× |
| 5 | First measurement pass | Nsight Systems timelines + Nsight Compute per-kernel |
| 6 | NVBit trace pass | Memory address traces, reuse-distance histograms |
| 7 | **Accel-Sim calibration** | AD107 config matching microbenchmarks within ~5% |
| 8–9 | Phase 3 measurements | Calibrated hit rates, bandwidth breakdowns, DAMOV classification |
| 10 | Workload characterization report | 15–25 page draft |
| 11 | Phase 3.5 — Jetson port | Same characterization on Jetson AGX Orin |
| 12 | Cross-platform comparison | Side-by-side workstation vs Jetson |

***Why this timeline isn't aggressive:*** you'll lose time to things that aren't in the timeline (debugging hardware issues, a paper deadline you decide to submit to, holidays). The milestones above are achievable working steadily; padding-free timelines slip.

***Why the mapping pipeline (month 4) is the gating milestone:*** every subsequent number depends on labelled traces. If the mapping pipeline isn't producing trustworthy labels, you can't say *anything* about which data structures are PiM candidates — you can only say something about kernels, which is much weaker. Hit this milestone or the project loses its thesis.

---

## 16. Annotated bibliography

Items marked **[critical]** must be read in detail; others can be skimmed unless directly relevant.

### 16.1 SLAM and Visual SLAM

- **[Korovko25] [critical]** Korovko, A., Slepichev, D., Efitorov, A., et al. *cuVSLAM: CUDA accelerated visual odometry and mapping*. arXiv:2506.04359, 2025. https://arxiv.org/abs/2506.04359.
- **[Cadena16] [critical]** Cadena, C., Carlone, L., et al. *Past, Present, and Future of Simultaneous Localization And Mapping: Towards the Robust-Perception Age*. IEEE Transactions on Robotics, 32(6):1309–1332, 2016. arXiv:1606.05830.
- **[Campos21]** Campos, C., et al. *ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multimap SLAM*. IEEE Transactions on Robotics, 37(6):1874–1890, 2021. arXiv:2007.11898. Code: https://github.com/UZ-SLAMLab/ORB_SLAM3.

### 16.2 Datasets

- **[Burri16]** Burri, M., et al. *The EuRoC micro aerial vehicle datasets*. International Journal of Robotics Research, 35(10), 2016.
- **[Geiger12]** Geiger, A., Lenz, P., Urtasun, R. *Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite*. CVPR 2012.
- **[Sturm12]** Sturm, J., et al. *A benchmark for the evaluation of RGB-D SLAM systems*. IROS 2012.
- **[Schubert18]** Schubert, D., et al. *The TUM VI Benchmark for Evaluating Visual-Inertial Odometry*. IROS 2018. arXiv:1804.06120.
- **[Grupp17]** Grupp, M. *evo*. https://github.com/MichaelGrupp/evo.

### 16.3 Profiling tools and methodology

- **[Villa19] [critical]** Villa, O., Stephenson, M., Nellans, D., Keckler, S. *NVBit: A Dynamic Binary Instrumentation Framework for NVIDIA GPUs*. MICRO 2019.
- **[NVIDIA-NSC] [critical]** NVIDIA. *Nsight Compute Profiling Guide*. https://docs.nvidia.com/nsight-compute/ProfilingGuide/.
- **[NVIDIA-NSS]** NVIDIA. *Nsight Systems documentation*. https://docs.nvidia.com/nsight-systems/.
- **[Yang20] [critical]** Yang, C., Kurth, T., Williams, S. *Hierarchical Roofline analysis for GPUs*. CCPE 2020.
- **ERT.** NERSC Empirical Roofline Toolkit.

### 16.4 GPU simulation, sampling, and characterization workflows

- **[Khairy20] [critical]** Khairy, M., et al. *Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling*. ISCA 2020.
- **[Naderan21] [critical]** Naderan-Tahan, M., Eeckhout, L. *Cactus: Top-Down GPU-Compute Benchmarking using Real-Life Applications*. IISWC 2021. Code: https://github.com/gpubench/cactus.
- **[Naderan23] [critical]** Naderan-Tahan, M., SeyyedAghaei, H., Eeckhout, L. *Sieve: Stratified GPU-Compute Workload Sampling*. ISPASS 2023. Code: https://github.com/gpubench/sieve.
- **[Cao23] [critical]** Cao, J., Sen, R., Interlandi, M., Arulraj, J., Kim, H. *Revisiting Query Performance in GPU Database Systems*. arXiv:2302.00734, 2023. Code: https://github.com/jiashenC/gpudb-char-and-opt.
- **[Tang26]** *MSched: GPU Multitasking via Proactive Memory Scheduling*. arXiv:2512.24637.
- **[Yu25]** *cuThermo: Understanding GPU Memory Inefficiencies with Heat Map Profiling*. arXiv:2507.18729.

### 16.5 Memory-system characterization and PiM/ISP context

- **[Oliveira21] [critical]** Oliveira, G.F., Gómez-Luna, J., Orosa, L., et al. *DAMOV: A New Methodology and Benchmark Suite for Evaluating Data Movement Bottlenecks*. IEEE Access, 2021. arXiv:2105.03725.
- **[Gomez-Luna22] [critical]** Gómez-Luna, J., et al. *Benchmarking a New Paradigm: Experimental Analysis and Characterization of a Real Processing-in-Memory System*. IEEE Access, 2022. arXiv:2105.03814.
- **[Mutlu24]** Mutlu, O., Ghose, S., Gómez-Luna, J., Ausavarungnirun, R. *A Modern Primer on Processing in Memory*. arXiv:2012.03112.

### 16.6 SLAM accelerator prior art

- **[Suleiman19]** Suleiman, A., Zhang, Z., Carlone, L., Karaman, S., Sze, V. *Navion: A 2-mW Fully Integrated Real-Time Visual-Inertial Odometry Accelerator*. IEEE JSSC, 2019.
- **[Liu19]** Liu, R., et al. *eSLAM*. DAC 2019.
- **[Vemulapati22]** Vemulapati, V., Chen, D. *FSLAM*. ICFPT 2022.

### 16.7 Reading order with rationale

1. [Korovko25] — what you're profiling. *First because nothing else makes sense without knowing what cuVSLAM is.*
2. [Cadena16] — what SLAM is. *Survey-level orientation; only need to read deeply if SLAM is new to you.*
3. [Campos21] (skim) — the comparison point. *For contrast with cuVSLAM's design choices.*
4. [NVIDIA-NSC] — your microscope. *Read sections on Memory Workload Analysis, Range Replay, Sets and Sections.*
5. [Villa19] — your address-trace gun. *Conceptual model for how NVBit works.*
6. [Yang20] — bandwidth and roofline tooling. *Methodology for the roofline artifact.*
7. **[Cao23]** — concrete example of NCU-based GPU characterization. *Direct tooling reuse.*
8. **[Naderan21]** — methodological justification for application-driven characterization. *Why cuVSLAM is the right vehicle.*
9. **[Naderan23]** — Sieve sampling. *Mandatory before NVBit trace pass.*
10. **[Khairy20]** — the cache simulator. *Mandatory before Phase 3.*
11. **[Tang26]** and **[Yu25]** — address-to-data-structure mapping prior art. *For Phase 2 methodology.*
12. [Oliveira21] — methodology template. *Read deeply; your characterization mirrors theirs.*
13. [Gomez-Luna22] — what the output should look like. *Reference for paper structure.*
14. [Suleiman19] — accelerator-side context. *For positioning the work against compute-side accelerators.*
15. [Mutlu24] (deep-read later) — why anyone cares. *Read before Phase 4+; for now, skim.*

---

## 17. Glossary

(Same as v4 — see that document or read individual entries as they appear in context.)

---

## 18. Common pitfalls and FAQ

**Q: `nvidia-smi` doesn't show the GPU after Ubuntu install.**
Secure Boot. Disable in BIOS and reinstall the driver. *The reason this happens: Secure Boot requires signed kernel modules, and the default NVIDIA driver isn't signed by a key your firmware trusts.*

**Q: `ncu` says "ERR_NVGPUCTRPERM" when running.**
Profiler permissions. Apply the kernel module fix in Section 5.6. *Why: NVIDIA restricts GPU performance counters to root by default to prevent side-channel leaks; you disable the restriction because you're the only user.*

**Q: I get `ImportError: invalid ELF header` when importing `cuvslam`.**
Missed `git lfs pull`. Run it, reinstall the wheel. *Why this happens: without LFS, the `.so` file is a text pointer, not a binary. The Python import sees something that isn't an ELF header and refuses.*

**Q: ATE on EuRoC is huge.**
Debug in order: (1) calibration, (2) IMU-camera time sync, (3) frame-drop rate, (4) GPU clock state, (5) ECC errors. *Why this order: most-likely cause first. Calibration errors are by far the most common; ECC errors are by far the rarest.*

**Q: Nsight Compute reports 0% cache hit rate.**
L1/L2 flush between replay passes. Use Range Replay, or Accel-Sim for steady-state. *Why this counter-intuitive result: the flush is by design, not a bug, but it means the reported number is cold-start rather than steady-state.*

**Q: NVBit traces are 100s of GB.**
Expected. Mitigations: (1) profile 20–50 frame windows; (2) gate emission with env vars; (3) on-the-fly `zstd`; (4) provision 1 TB SSD.

**Q: My `unmapped` bucket in the post-trace join is 30%.**
Layer 1 audit is incomplete. Use Layer 2 NVBit hook output to find missing call sites. *Why this is the right diagnostic move: Layer 2 sees every `cuMemAlloc` regardless of source; the diff between Layers 1 and 2 is exactly the set of allocations you missed.*

**Q: Accel-Sim disagrees with Nsight Compute by 30%.**
Calibration. Run more microbenchmarks. *Why 5% should be achievable but 30% means structural mismatch: 30% indicates the simulator and hardware are modelling fundamentally different things (wrong cache geometry, wrong clock, wrong scheduler). 5% is the noise floor of microbenchmark variability — getting closer than that requires effort disproportionate to the benefit.*

**Q: GPU clocks won't stay locked after reboot.**
`nvidia-smi -lgc` doesn't persist. Put `scripts/lock_clocks.sh` in a systemd service. *Why this isn't fixed in the driver: locking is intended as a runtime knob, not a configuration setting; the assumption is that you re-lock before each measurement run.*

**Q: ECC is reported as "disabled".**
Re-enable: `sudo nvidia-smi -e 1`. Requires reboot.

**Q: My RTX 2000 Ada thermal-throttles during long traces.**
Lower `-lgc` to 1500 MHz. *Why: the 70 W TDP and small-form-factor airflow in the Precision 7875 can throttle sustained 1620 MHz. The point of locking is reproducibility, not peak performance — choose a frequency you can sustain.*

**Q: How do I deal with cuVSLAM being non-deterministic?**
RANSAC. Seed RNGs where possible; report distributions over ≥ 5 runs. *Why architecture metrics are still stable: bandwidth, hit rate, instruction count are determined by the workload's memory access pattern, which RANSAC changes only marginally. ATE swings because pose estimates compound; cache hit rate doesn't.*

**Q: When does Jetson AGX Orin show up?**
After Phase 3 stabilizes on the workstation.

**Q: I want to use ROS later.**
Fine. Layer `ros-humble-ros-base` and `isaac_ros_visual_slam` on top. Doesn't change the profiling work.

**Q: How do I cite this onboarding doc?**
You don't. Cite the primary sources in Section 16.

---

*End of onboarding package v5 (Annotated Edition). ~17,500 words. Native Ubuntu 22.04 LTS on Dell Precision 7875 with RTX 2000 Ada Generation. Methodology core unchanged from v3/v4. New in v5: pedagogical rationale woven through every section, a new Section 0 framing the research at the conceptual level, and "Why" prose blocks on every major decision. Supersedes v1, v2, v3, v4.*
