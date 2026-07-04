# cuVSLAM-stack — Full Onboarding (start here, no prior knowledge assumed)

**Who this is for:** someone who has just been handed this repository and needs to
continue a computer-architecture research project **without** the previous
human or any AI assistant available. By the end you will understand *why* the
project exists, *what* has been measured, *how* every number is produced, *which
machines* do the work and how to reach them, and *what to do next* — with the
exact commands.

Read it top to bottom once. Then keep it open as a reference. Nothing here
assumes you know SLAM, GPU profiling, or memory architecture; all of that is
explained. Where a claim is subtle or was learned the hard way, it says so.

> **Companion docs** (read after this one, in this order):
> `profiling/PROJECT_STATUS.md` (living status board), `profiling/METHODOLOGY.md`
> (how each number is made), `profiling/WALKTHROUGH.md` (guided tour of results),
> `profiling/PUBLISHABILITY.md` (the reviewer-objection register), and the
> original design bible `suggestions_and_summuries/cuVSLAM_Profiling_Onboarding_v5 (1).md`.
> This file is the on-ramp that ties them together.

---

## Part 0 — The 60-second version

We are building the empirical case for a new kind of computer hardware. Robots
and other "Physical AI" systems are limited less by raw compute than by the
energy and time spent **moving data** between memory/storage and the processor.
Two hardware ideas attack that: **Processing-in-Memory (PiM)** — put small
compute units inside the DRAM — and **In-Storage Processing (ISP)** — put compute
inside the SSD. To justify designing such hardware, you first have to *prove*
that a real, important workload is bottlenecked on data movement and show
*which parts* of it would benefit. Our workload is **cuVSLAM**, NVIDIA's
production visual-SLAM system (the thing that lets a robot figure out where it
is from camera images). We profile it kernel-by-kernel and, now,
data-structure-by-data-structure, on real GPUs, and sort each piece into a
three-way "where should this live" taxonomy. That measurement study is a paper
(venue: ISPASS or IISWC). The hardware design that follows is the bigger paper
(MICRO/ASPLOS/ISCA/HPCA).

**Status:** the measurement study is essentially done and committed. The newest
result — attributing memory traffic to named data structures — just landed, and
a campaign re-running it across all 27 dataset sequences is executing on the
workstation right now.

---

## Part 1 — The science, explained from scratch

### 1.1 The data-movement bottleneck

A modern GPU can do arithmetic far faster than it can fetch the operands from
DRAM. Fetching a 32-bit word from off-chip DRAM costs roughly two orders of
magnitude more energy than the arithmetic you then do on it. So for any workload
whose kernels spend their time *reading and writing memory* rather than
*computing*, the processor sits idle waiting, and most of the energy is burned
in the wires and DRAM, not the math. This is the "memory wall." It is worse for
Physical AI (robots, drones, embodied agents) because they must run in real
time, at low power, on a battery.

### 1.2 The two hardware answers

- **Processing-in-Memory (PiM):** integrate simple compute units *inside or next
  to the DRAM dies*, so operations happen where the data already is. You avoid
  hauling data across the memory bus. Good for data that is **hot** (touched
  constantly) and **large** (doesn't fit in cache). Real examples: UPMEM DPUs,
  Samsung HBM-PIM, Newton.
- **In-Storage Processing (ISP):** integrate compute *inside the SSD controller*,
  so you can scan/filter a large dataset without streaming all of it into host
  DRAM first. Good for data that is **cold** (touched rarely) and **huge** (lives
  on disk). Real examples: Samsung SmartSSD, computational-storage drives.

Neither is free — they need silicon, memory controllers, and system software.
That is exactly why you need a rigorous characterization first: to point at
specific parts of a real workload and say "*this* belongs in PiM, *that* belongs
in ISP, and here is the measured evidence."

### 1.3 What Visual SLAM is, and what cuVSLAM is

**SLAM** = Simultaneous Localization And Mapping. A robot with a camera must
answer two coupled questions at once: *where am I?* (localization) and *what does
the world around me look like?* (mapping). **Visual** SLAM does this from camera
images (mono, stereo, or RGB-D = color + depth).

A visual-SLAM system has two loosely-coupled halves:

- **Front-end / odometry (per frame, every frame):** for each incoming image,
  detect visual features (corners), track them from the previous frame, and
  estimate the small camera motion since the last frame. This is a firehose:
  it runs at camera frame rate and touches the image pyramid, feature lists,
  and a **local map** of recently-seen 3-D points.
- **Back-end / SLAM (occasionally):** maintains a longer-term **map** of
  **keyframes** (selected representative frames) and their features, runs
  **bundle adjustment** (a big least-squares optimization that jointly refines
  camera poses and 3-D points), and performs **loop closure** — recognizing
  "I've been here before" and correcting accumulated drift by searching a
  **keyframe database**.

**cuVSLAM** is NVIDIA's production, GPU-accelerated implementation of this,
shipped inside Isaac ROS and used on real robots. We profile it because it is
*real production code*, not a toy — reviewers can't dismiss it.

### 1.4 The thesis: a three-way persistence taxonomy

This is the intellectual core of the project. We claim cuVSLAM's memory has
**three qualitatively different persistence classes**, and each maps to a
different hardware substrate:

| Class | What lives here | When touched | Size behavior | Hardware target |
|---|---|---|---|---|
| **Streaming** | per-frame images, pyramids | every frame, then discarded | fixed, small | near-sensor SRAM (on the camera/SoC) |
| **Hot-persistent** | the local map (recent 3-D points), BA linear system | every frame, read+written | fixed, medium | **DRAM-PiM** |
| **Cold-persistent** | the keyframe database (loop-closure store) | only on loop closure (rare) | **grows with session length** | **ISP** (in the SSD) |

The whole measurement effort exists to *test and quantify* this taxonomy: does
cuVSLAM's memory really cluster into these three classes, and can we put numbers
(bandwidth, footprint, reuse distance, which named allocation) behind each cell?

### 1.5 The published-methodology lineage

We are not inventing profiling from nothing; we adapt two established methods so
reviewers recognize the rigor:

- **DAMOV** (Oliveira et al., 2021) — a systematic method for finding
  data-movement bottlenecks. It runs a workload, measures locality, and sorts
  kernels into classes (their paper has classes like "temporal-locality-bound,"
  "compute-bound," "DRAM-bandwidth-bound," etc.). DAMOV was written for CPUs;
  we adapt it to GPUs (there's a documented "GPU-DAMOV adaptation study" that
  defines our G1–G7 classes — see `profiling/METHODOLOGY.md`).
- **Cao23 roofline** — the NCU-counter-based roofline methodology for deciding
  whether a GPU kernel is compute-bound or memory-bound, and by how much.

---

## Part 2 — The methodology, so you can read (and defend) the numbers

### 2.1 Compute-bound vs memory-bound, and the roofline

Every kernel has an **arithmetic intensity** = (FLOPs performed) ÷ (bytes moved
from DRAM). Plot achieved GFLOP/s against arithmetic intensity and you get the
**roofline**: a diagonal line (the memory-bandwidth ceiling — you can't go faster
than DRAM feeds you) that bends into a horizontal line (the compute ceiling).
A kernel sitting on the diagonal is **memory-bound** (data movement is the
limit — a PiM/ISP candidate). A kernel on the flat top is **compute-bound**
(arithmetic is the limit — leave it on the GPU). The whole PiM/ISP argument
lives on the diagonal.

The **ceilings must be measured on the actual chip at locked clocks**, not taken
from the spec sheet — spec sheets quote boost clocks you never sustain. We
measure them with `profiling/env/measure_ceilings.py` (an ERT-style microbench):
on our workstation GPU, locked, they are **205.0 GB/s DRAM bandwidth** and
**5445 GFLOP/s FP32**.

### 2.2 Locality and reuse distance (the DAMOV core)

Bandwidth tells you *how much* data moved; **locality** tells you *whether a cache
could have avoided it*. The key metric is **reuse distance** (a.k.a. LRU stack
distance): for each memory access, how many *distinct* other locations were
touched since this same location was last touched? If that number is small, a
cache of that size would have held the data (a hit); if it's large, the access
misses and hits DRAM no matter how big your cache. Compute the reuse-distance
histogram and you can predict the hit rate at *any* cache size — an
**architecture-independent** locality curve. A curve that stays flat as you grow
the cache means the data is **cache-immune streaming**: it will hit DRAM on any
real machine → a strong near-data (PiM/ISP) signal.

We compute this from **real per-warp address traces** captured with NVBit (see
Part 4), in `profiling/analysis/locality.py`.

### 2.3 Coalescing vs divergence (two things counters conflate)

When a warp (32 GPU threads) issues a memory instruction, the 32 addresses can
be:
- **coalesced** (contiguous → few cache sectors touched, efficient), or
- **scattered** (spread out → many sectors, a "gather," inefficient); and
  separately the warp can be
- **converged** (all 32 lanes active) or **divergent** (some lanes masked off).

NCU's `sectors/request` counter blends these. The address trace lets you
separate them exactly (sectors per *active* lane vs number of active lanes).
**This distinction caught a real error** (see 2.6).

### 2.4 The GPU-DAMOV G-classes → PiM/ISP affinity

`profiling/analysis/classify.py` implements a decision tree that reads each
kernel's counters and assigns a G-class (G1–G7) and a PiM/ISP affinity
(`strong` / `conditional` / `none`). Roughly: memory-bound + poor locality +
large footprint → strong PiM; rarely-touched + grows with session → ISP;
compute-bound or cache-friendly → none. The exact rules are in `classify.py` and
`METHODOLOGY.md`. The taxonomy was **validated by clustering** (k-means over the
metric vectors independently prefers 7–8 clusters, matching the 7 G-classes —
so the classes fall out of the data, they aren't just asserted by the tree).

### 2.5 Memory spaces (critical — this is how you avoid a wrong answer)

A GPU has several memory spaces, and NVBit's tracer records accesses to **all**
of them. Only one space can hit a named data structure in DRAM:

- **Global** (`LDG`/`STG`): the real DRAM-backed heap. **This is the only space
  where "which data structure?" is a meaningful question.**
- **Shared** (`LDS`/`STS`): on-chip scratchpad, per-thread-block. Fast SRAM,
  *never* touches DRAM. Convolution/sort kernels are dominated by this.
- **Local** (`LDL`/`STL`): the per-thread "spill" window — when a kernel needs
  more registers than exist, the compiler spills to local memory. It *is*
  DRAM-backed, but it is compiler scratch, **not a data structure**.

**If you attribute all accesses indiscriminately, 88–98% of a kernel's traffic
looks "unmapped" (it's shared tiles and register spill), and you draw the wrong
conclusion.** The attribution join therefore buckets by opcode first
(shared → `shared_onchip`, local → `local_spill`) and only runs data-structure
lookup on global accesses. After this split, "unmapped" global traffic drops to
≤7% (usually <1%). **Remember this**; it is the single most important correctness
rule in the newest analysis, and it also means the older Slice-3 locality
numbers need re-deriving with the same filter (open task, Part 10).

### 2.6 Methodology self-corrections (a feature, not a bug)

Twice now, a finer measurement overturned a coarser one. Reviewers *love* this
because it shows the method policing itself:

1. **The st_track proxy correction (Slice-3):** NCU's `sectors/request` counter
   labeled the loop-closure scan kernel `st_track_with_cache` a scattered
   gather (which would be a PiM/ISP candidate for random access). The real
   address trace showed it is fully *coalesced and converged* streaming — the
   counter was misled. The ISP argument then re-grounded from "random gather in
   the kernel" to "the keyframe **database grows with session length**" — a
   different and stronger claim.
2. **The memory-space correction (attribution):** the very first
   data-structure joins showed ~95% "unmapped," which would have looked like a
   broken tool. It wasn't — it was shared/local traffic being counted. Splitting
   by memory space (2.5) fixed it and revealed that `st_track_with_cache`'s DRAM
   traffic is **94% register spill**, only 5% actual keyframe-descriptor reads.

Both are documented in `PUBLISHABILITY.md` as "self-correcting method" rows.

---

## Part 3 — The machines and how to reach them (the crown jewels)

**None of this is recoverable from the git repo. Guard it.** Two machines:

### 3.1 The laptop (`iNOMAL`, CachyOS/Arch, GeForce MX450)

Development + CPU-side analysis + writing. Small 2 GB GPU, not for real numbers.

- **Known problem:** the CUDA install is corrupt (a disk-full pacman upgrade on
  2026-07-01 zero-filled ~118 files including `libnvJitLink.so.13`,
  `libnvrtc.so.13`). `import cuvslam` fails with "invalid ELF header" unless you
  prefix workload commands with:
  ```bash
  LD_LIBRARY_PATH=$HOME/.local/cuda-repair/lib <command>
  ```
  (`~/.local/cuda-repair/lib` holds clean `.so` copies extracted from the pacman
  cache with soname symlinks.) **Permanent fix** (needs the human's sudo):
  `sudo pacman -S cuda`, then delete `~/.local/cuda-repair`. Verify with
  `LC_ALL=C pacman -Qkk cuda | grep -c "SHA256 checksum mismatch"` → should be 0.
  Note: the analysis layer (`python3 -m analysis.*`) is pure-stdlib and does
  **not** need CUDA — you can re-run all synthesis on the laptop from committed
  CSVs without touching this.

### 3.2 The workstation (`ssh ndpvslam@dell-workstation`, RTX 2000 Ada) — the results box

**Every real number and every trace comes from here.**

- **Access:** SSH key auth from the laptop; **passwordless sudo**
  (`/etc/sudoers.d/ndpvslam`). SSH goes over **Tailscale SSH**, which
  periodically demands re-authentication — it prints a `login.tailscale.com/...`
  URL and blocks until someone opens it. **Only the human can clear that.** If
  your SSH suddenly prints a login URL and hangs, that's what happened; you're
  blocked until they re-auth.
- **The unified driver/CUDA/tool stack (learned the hard way — do not "fix" it):**
  - Driver **575.64.05** + CUDA **12.9.1**, booting **linux-lts 6.12.39** (GRUB
    default). This specific combination is load-bearing: **NVBit needs driver
    ≤ 575**, so you can't just upgrade.
  - Because CUDA-13's Nsight tools *reject* a 575 driver, `ncu` must be the
    CUDA-12.9 build **ncu 2025.2.1.3** at `~/ncu2025/` (point tools at it with
    the `NCU_BIN` env var), and `nsys` is 2025.3.2. All three — NVBit,
    ncu 2025.2, nsys 2025.3.2 — coexist on the one 575 driver. `profile.py`
    honors `NCU_BIN` / `NSYS_BIN` for exactly this reason.
  - There is a rollback to a 610 driver at `~/driver-rollback/revert.sh` — **do
    not run it**, it re-breaks NVBit. It exists only as an escape hatch.
- **Two cuVSLAM Python venvs on the workstation:**
  - `~/Projects/cuvslam-stack/cuvslam_venv` — the **baseline** cu12 wheel
    (rebuilt from source). Use for normal profiling.
  - `~/Projects/cuvslam-stack/cuvslam_venv_tagged` — the **instrumented** wheel
    (TaggedAllocator + NVTX, see Part 5). Use for attribution captures.
  - (`cuvslam_venv_cu13` is the original cu13 wheel, kept for reference.)
- **The cuVSLAM source tree** is at `~/Projects/cuvslam_src_cu12` — this is what
  the wheels are built from, and where the instrumentation patch is applied.
- **Locking clocks (do this before every measurement run):**
  ```bash
  sudo nvidia-smi -pm 1            # persistence mode
  sudo nvidia-smi -lgc 1620,1620   # lock graphics clock
  sudo nvidia-smi -lmc 7001,7001   # lock memory clock
  ```
  At this lock the measured ceilings are 205.0 GB/s / 5445 GFLOP/s.
  **Locks reset on every reboot or power cut — re-apply.** Unlocked, run-to-run
  variance is ~50%; locked, it's 0.14%. (Traces don't need locked clocks — they
  aren't timed — but ncu/nsys captures absolutely do.)
- **Freeing the GPU for captures:** the machine BIOS powers on with AC and
  auto-logs-into KDE, so after a power cut a desktop compositor is sitting on the
  GPU and perturbs ncu/nsys captures. The human's `~/free_gpu.zsh` kills the KDE
  compositor; `~/restore_gui.zsh` brings it back. Run free_gpu before a capture
  campaign — **note it resets the clocks, so re-lock afterward.**
- **Datasets** live on the 3.6 TB drive **sda2, mounted at `/mnt/data`**:
  KITTI (color, seq 00–21), EuRoC (×11 under `/mnt/data/EuRoC/...`),
  TUM RGB-D fr3 (under `/mnt/data/TUM_RGBD/extracted/...`), TUM-VI.
  - **NTFS gotcha (learned this session):** sda2 is NTFS and fstab-mounted
    **read-only**, and its "dirty bit" survives power cuts, so a plain
    `mount -o rw` fails with *"Volume is dirty and force flag is not set."* To
    get it writable (needed to dump outputs there):
    ```bash
    sudo umount /mnt/data
    sudo mount -t ntfs3 -o rw,force /dev/sda2 /mnt/data
    ```
    (`ntfsfix` is not installed. The attribution campaign script re-asserts this
    mount automatically in its preamble.)
- **THE IRON RULE for long jobs:**
  ```bash
  setsid nohup <your-command> > ~/some.log 2>&1 &
  ```
  Power cuts and SSH drops have killed multi-hour jobs several times. `setsid`
  detaches from the terminal, `nohup` ignores hangups, `&` backgrounds it, and
  the log file is your window into it. **Never run a multi-hour job in a plain
  foreground SSH session.** Also make every long script **resumable** (skip any
  step whose output already exists) so a relaunch after a crash costs nothing.

---

## Part 4 — The profiling tools, each explained

You will use four measurement tools plus a symbolizer. Know what each gives you:

| Tool | What it does | Granularity | Perturbs timing? |
|---|---|---|---|
| **nsys** (Nsight Systems) | timeline of every CUDA API call, kernel launch, memcpy, and (if enabled) NVTX ranges | whole-run timeline | low |
| **ncu** (Nsight Compute) | hardware performance counters per kernel launch (bandwidth, FLOPs, cache hit rates, sectors/request, roofline inputs) | per kernel, replays each kernel | high (kernel replay) — **timed separately** |
| **NVBit** | binary instrumentation that emits the **actual memory addresses** every warp touches | per-warp per-access address stream | massive (10–1000×) — **NEVER use for timing** |
| **addr2line** | turns a program-counter value into `file:line` (needs a build with debug info) | — | — |

- **NVTX** is not a tool but an annotation: source code can mark named ranges
  ("VIO:track()", "SLAM:LC & optimization") that nsys then attributes kernels to.
  cuVSLAM's own profiler emits these when built with `-DUSE_NVTX=ON`. This is how
  we map kernels to sub-frame stages *measured* rather than by name-guessing.

**Why NVBit traces are gated and windowed:** a full-run address trace is
terabyte-scale and runs 100–1000× slower. So we never trace the whole run. Two
gates (in our patched `mem_trace` tool, `blocked/mem_trace_windowing.patch`):
- `LAUNCH_BEGIN=N LAUNCH_END=M` — only instrument grid-launch ids in `[N, M)`.
  Outside the window, kernels run at near-native speed. `LAUNCH_END=0` means
  "instrument nothing" — used to get the fast full-run *launch map* (the
  launch-id ↔ kernel-name table) without paying the trace cost.
- `KERNEL_FILTER=substr` — only instrument kernels whose demangled name contains
  `substr` (e.g. `st_` catches every keyframe-scan kernel across the whole run
  without instrumenting the thousands of other launches between them).

---

## Part 5 — The instrumentation we added (TaggedAllocator + NVTX)

The characterization up to mid-2026 made **kernel-level** claims ("st_track is
memory-bound"). Reviewers rightly object: you conclude things about *data
structures* ("the keyframe database belongs in ISP") but you only ever observed
*kernels*. Closing that gap is the "TaggedAllocator" milestone (onboarding §11.2
in the design bible). It is a **three-layer** pipeline so that independent
signals cross-check each other:

- **Layer 1 (source, primary):** patch cuVSLAM so every GPU allocation records
  its pointer, size, and a **host backtrace** to a journal file
  (`CUVSLAM_ALLOC_LOG=<path>`). The backtrace says *which line of cuVSLAM source*
  asked for the buffer → which data structure it is. Implemented in
  `patches/0002-tagged-allocator-nvtx.patch`: a small `alloc_log.{h,cpp}` plus
  `LogAlloc`/`LogFree` calls in the RAII wrapper classes (`GPUArray`,
  `GPUArrayPinned`, `GPUOnlyArray`, `GPUImage`) and the 3 direct `cudaMalloc`
  sites. Enabled only when the env var is set — otherwise it's one branch on a
  cached flag, on allocation paths only. Pinned-host halves are journaled too
  (they're device-visible under UVA).
- **Layer 2 (driver, verification):** the NVBit tool *also* logs every
  `cuMemAlloc*`/`cuMemHostAlloc`/`cuMemFree*` with the grid-launch id it precedes
  (`MEM_TRACE_ALLOC_LOG=<path>`). This catches allocations Layer 1 missed (CUB,
  cuSOLVER internals) and gives allocation lifetimes in the trace's own clock.
  Implemented in `blocked/mem_trace_alloc_events.patch`.
- **Layer 3 (kernel-args, optional refinement):** correlate access addresses
  with the pointers passed as kernel arguments. Not needed so far — Layer-1
  backtraces already disambiguate every observed allocation.

**The join** (`analysis/attribution.py`, subcommand `join`) streams the address
trace, keeps the live allocation set in launch-id order, buckets each access by
memory space (Part 2.5), and for global accesses looks up which allocation
(→ which data-structure **tag**) it falls into. Output: per-kernel × per-tag
byte/sector tables. The **tag vocabulary** (a fixed set: `keyframe_descriptors`,
`ba_linear_system`, `pyramid_levels`, `images_raw`, `feature_tracks`, etc.) is
defined by `TAG_RULES` in `attribution.py`, mapping owner call-sites to tags.

**The NVTX half:** the same patch flips `profiler_enable.h` so cuVSLAM's own
profiler domains become active under `-DUSE_NVTX=ON`, and nsys's
`nvtx_kern_sum` report then gives the measured kernel → stage table.

**Building the instrumented wheel** (on the workstation, inside the podman
builder because the host gcc is too new for nvcc 12.9):
```bash
cd ~/Projects/cuvslam_src_cu12
# patch is already applied on the ws; to re-apply on a fresh tree:
#   patch -p1 < ~/Projects/cuvslam-stack/patches/0002-tagged-allocator-nvtx.patch
podman run --rm --userns=keep-id --network host -v "$PWD:/cuvslam:Z" -w /cuvslam \
    cuvslam-wheel-builder bash -c '
      set -e; export CUDA_HOME=/usr/local/cuda HOME=/tmp
      cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_ARCHITECTURES=OFF \
            -DUSE_NVTX=ON -S /cuvslam -B /cuvslam/build
      make -j"$(nproc)" -C /cuvslam/build cuvslam
      export CUVSLAM_BUILD_DIR=/cuvslam/build
      python3 -m build --wheel --no-isolation --outdir /cuvslam/dist /cuvslam/python'
~/Projects/cuvslam-stack/cuvslam_venv_tagged/bin/pip install --force-reinstall \
    ~/Projects/cuvslam_src_cu12/dist/cuvslam-*.whl
```
Key build flags: **`RelWithDebInfo`** keeps DWARF debug info (a plain `Release`
links `-s` and strips it, breaking `addr2line`); **`CMAKE_CUDA_ARCHITECTURES=OFF`**
lets the kernels use their own GPU-agnostic gencode; **`USE_NVTX=ON`** turns on
the profiler ranges. The baseline (uninstrumented) wheel is preserved at
`~/Projects/cuvslam_src_cu12/dist/baseline/`, and the original source files have
`.orig` backups from the `patch` run.

Rebuilding the NVBit tool after editing `mem_trace.cu` (also in the container —
host gcc16 breaks nvcc 12.9):
```bash
cd ~/Projects/cuvslam-stack/external_repos/nvbit_release_x86_64
podman run --rm -v "$PWD:/nvbit:Z" -w /nvbit/tools/mem_trace \
    cuvslam-wheel-builder make -j8
# produces tools/mem_trace/mem_trace.so
```

---

## Part 6 — The repository, directory by directory

Top level (`/home/m_inomal/Projects/cuvslam-stack` on the laptop;
`~/Projects/cuvslam-stack` on the workstation):

```
goal.md                     the research thrust, one paragraph (Part 1.1 quotes it)
HANDOFF.md                  the previous cross-account handoff (still useful context)
README.md                   the cuVSLAM *runner* readme (how run.py works)
run.py                      the workload entrypoint: `python run.py <config.toml>`
cuvslam_runner/             the Python runner package (loads a TOML, drives cuVSLAM)
configs/                    hand-written runner configs (not the profiling ones)
cuvslam_src/                a source checkout (laptop copy; the real one is on ws)
cuvslam_venv/               laptop venv (currently CUDA-broken; see Part 3.1)
external_repos/             NVBit and Accel-Sim live here (on the ws)
patches/
  0001-podman-wheel-build-cuda13.patch   the wheel-build patch
  0002-tagged-allocator-nvtx.patch       <-- OUR instrumentation (Part 5)
docs/
  ONBOARDING.md             <-- THIS FILE
  agent-memory/             the previous agent's memory notes (machine access etc.)
suggestions_and_summuries/
  cuVSLAM_Profiling_Onboarding_v5 (1).md   the original design bible (§11.2 = TaggedAllocator)
profiling/                  <-- EVERYTHING for the characterization lives here
```

Inside `profiling/`:

```
PROJECT_STATUS.md    living status board (read after this file)
METHODOLOGY.md       how every number is produced + the G-class rules
WALKTHROUGH.md       guided tour of the results
PUBLISHABILITY.md    the reviewer-objection register (what's closed / open)
PROFILING_PLAN.md    the milestone plan (M1..M7)
README.md            command reference for the profiling tools
run_characterization.sh   one-command pipeline for a single sequence

harness/
  profile.py         the nsys/ncu capture wrapper. Honors NCU_BIN / NSYS_BIN.
                     Has inline METRIC_SETS (curated ncu counters per Cao23).

analysis/            the headless, stdlib-only, GPU-free analysis layer.
                     Everything reruns from committed CSVs with no GPU.
  common.py          shared helpers (CSV IO, kernel-name demangling, hw loader)
  stages.py          kernel-name -> pipeline-stage mapping
  build_dag.py       kernels-per-frame, launch structure
  screen.py          first-pass memory-bound vs compute-bound verdicts
  roofline.py        the roofline math (arithmetic intensity, ceilings)
  bandwidth.py       achieved DRAM bandwidth per kernel
  transfers.py       host<->device copy accounting
  variance.py        run-to-run variance / CoV analysis
  classify.py        the GPU-DAMOV G1-G7 decision tree -> PiM/ISP affinity
  compare.py         cross-sequence agreement
  cluster.py         k-means taxonomy validation
  locality.py        NVBit trace -> reuse distance, footprint, coalescing
  attribution.py     <-- NEW: address -> data-structure join (Part 5)
  campaign.py        27-sequence synthesis
  make_report.py     renders a report from the derived CSVs
  svgfig.py          dependency-free SVG figures (no matplotlib, runs anywhere)

campaign/
  gen_configs.py             generate one odom + one slam config per sequence
  run_campaign.sh            the original 27-seq characterization campaign
  ws_slice3_campaign.sh      the Slice-3 NVBit locality campaign
  ws_attribution_capture.sh  single-sequence attribution capture (TUM long_office)
  ws_attribution_all.sh      <-- NEW: FULL-MATRIX attribution campaign (Part 8)

env/
  check_env.sh          verify the toolchain
  measure_ceilings.py   ERT-style microbench -> measured roofline ceilings
  gen_hw_descriptor.py  build a hw/*.toml from the live GPU
  lock_clocks.sh        the nvidia-smi clock-lock sequence
  fetch_datasets.sh     dataset download helper
  setup_perms.sh        perf-counter permissions
  gpu_warmup.py         warm the GPU to steady state before a capture

hw/                     hardware descriptors (measured ceilings, cache sizes)
  dellworkstation_sm89.toml  <-- the workstation (RTX 2000 Ada) - use this
  rtx2000ada_sm89.toml       (same GPU, generic)
  mx450_sm75.toml            the laptop
  jetson_orin_sm87.toml      future target

configs/               profiling runner configs (hand-written singles)
  configs/campaign/    the 27-sequence generated configs (gen_configs.py output)

blocked/               the NVBit / Accel-Sim track (was "driver-gated", now unblocked)
  check_capability.sh          gate: is this host NVBit-capable?
  run_nvbit_memtrace.sh        run a windowed address trace
  mem_trace_windowing.patch    LAUNCH_BEGIN/END + KERNEL_FILTER for mem_trace
  mem_trace_alloc_events.patch <-- NEW: the Layer-2 alloc sidecar (Part 5)
  run_accelsim.sh              Accel-Sim runner (future: Phase 4)

tests/test_analysis.py   15 GPU-free tests. Run: python3 profiling/tests/test_analysis.py

reports/               committed reports (reproduce findings with NO GPU)
  2026-07-04_campaign/          the 27-sequence characterization
  2026-07-04_slice3_locality/   the NVBit locality result
  2026-07-04_attribution/       <-- NEW: the data-structure attribution report
  (older per-machine reports too)

results/, results_ws/, results_ws_campaign/   raw + derived run outputs
                     (results_ws_campaign is gitignored, 35 MB, on the laptop;
                      needed to re-run analysis.campaign from scratch)
```

---

## Part 7 — What has been done (the results, so you can cite them)

All committed and pushed to `github.com/mij001/cuvslam-stack`, branch `main`.

1. **Locked-clock production pass** (RTX 2000 Ada): 5-repeat coefficient of
   variation **0.14%** (vs ~50% unlocked); measured ceilings 205 GB/s / 5445
   GFLOP/s. This makes every subsequent number a stable quantity.
2. **Full-scale 27-sequence campaign** (`reports/2026-07-04_campaign/`): KITTI
   00–10, EuRoC ×11, TUM fr3 ×4, TUM-VI — each in odometry and SLAM mode, 0
   failures. Cross-sequence **modal class consistency 91%**; pooled k-means
   prefers **k=7–8** (validates the taxonomy); **61% of GPU time carries PiM
   affinity**.
3. **Slice-3 locality** (`reports/2026-07-04_slice3_locality/`): the first
   architecture-independent locality from real NVBit address traces. Front-end
   reuse CDF is flat from 64 KiB to 48 MiB (cache-immune streaming). This is
   where the st_track counter-proxy correction happened (Part 2.6 #1).
4. **Data-structure attribution** (`reports/2026-07-04_attribution/`, the newest,
   commit `0e237fa`): the TaggedAllocator + NVTX pipeline (Part 5). Headline
   findings on TUM fr3 long_office (SLAM):
   - The **GPU memory budget is static**: 240 device allocations totaling
     108.65 MB, plus 34 pinned-host buffers (17.1 MB), and it does **not grow**
     over 2500 loop-closing frames. The keyframe state on the GPU is a fixed
     **6.7 MB** descriptor buffer. → The session-scale database that the ISP
     argument is about grows **host-side** (LMDB), not on the GPU. This confirms
     the Slice-3 ISP re-grounding *from the allocator itself*.
   - `st_track_with_cache` (the loop-closure scan) DRAM traffic is **94%
     register spill**, only 5% actual keyframe-descriptor reads. So it's
     spill-bandwidth-bound, and its data structure is a fixed small buffer —
     the ISP target is the host database, and the device kernel would benefit
     from larger register files, not near-data placement.
   - The bundle-adjustment kernels stream **`ba_linear_system` at 97%** of their
     global traffic — one named, pre-sized, contiguous structure → the clean
     **PiM** candidate.
   - Front-end kernels (conv, GFTT, sort) are **89–98% shared-memory tiles**;
     their small global residue is exactly the pyramid/track structures the
     streaming class predicts.
   - The **NVTX kernel→stage map is now measured** (`nvtx_kern_sum.csv`):
     `st_track_with_cache` sits under "SLAM:LC & optimization", `st_build_cache`
     under keyframe ingest — closing the "you mapped kernels to stages by regex"
     objection.

**Reviewer-objection register** (`PUBLISHABILITY.md`): issues **1, 2, 3, 4, 7, 8
closed**; issue 5 (no PiM-side simulation), 6 (no energy numbers), 9 (closed
source), 10 (LICENSE), 11 (inter-kernel movement) open or partial; plus two
"NEW" self-correction rows (Part 2.6).

---

## Part 8 — What is running RIGHT NOW (the campaign)

A **full-matrix attribution campaign** launched on the workstation on
2026-07-04 ~16:34 IST: `profiling/campaign/ws_attribution_all.sh`. It repeats the
Part-5 attribution across **all 27 SLAM sequences**. Per sequence it runs:
1. **pass1** — NVBit injected, empty window → the full launch map + the Layer-1
   journal + the Layer-2 sidecar (near-native speed);
2. **window** — auto-picked from pass1 (≈3 frames of launches mid-sequence);
3. **nsys NVTX** — the kernel→stage table;
4. **pass2a** — all-kernel trace over the steady-state window;
5. **pass2b** — `KERNEL_FILTER=st_` → every keyframe-database scan, whole run
   (this is the slow step);
6. **resolve + join** → per-kernel × data-structure CSVs.

Properties: **resumable** (any step whose output exists is skipped — relaunch
after a crash costs nothing), 3-hour timeout per capture step, and it
**re-asserts the rw NTFS mount** in its preamble.

- **All outputs go to sda:** `/mnt/data/attribution_out/<sequence>/`
- **Watch progress from your dev box:**
  ```bash
  ssh ndpvslam@dell-workstation tail -f '~/attribution_campaign.log'
  ssh ndpvslam@dell-workstation cat /mnt/data/attribution_out/PROGRESS
  ```
- **Relaunch if it dies** (power cut, Tailscale re-auth, etc.):
  ```bash
  ssh ndpvslam@dell-workstation
  cd ~/Projects/cuvslam-stack
  setsid nohup profiling/campaign/ws_attribution_all.sh > /dev/null 2>&1 &
  ```
- **Estimated total:** 15–30 hours. When it finishes, `PROGRESS` reads
  `CAMPAIGN DONE` and the log lists ✓/✗ per sequence.

---

## Part 9 — How to actually DO the work (step-by-step recipes)

### Recipe A — Re-run all analysis with no GPU (sanity check, on any machine)
```bash
cd /home/m_inomal/Projects/cuvslam-stack
python3 profiling/tests/test_analysis.py          # expect: OK, 15 tests
# re-synthesize the 27-seq campaign from committed derived CSVs:
cd profiling && python3 -m analysis.campaign results_ws_campaign \
    --hw hw/dellworkstation_sm89.toml --out /tmp/campaign_check
# compare /tmp/campaign_check/CAMPAIGN.md to reports/2026-07-04_campaign/CAMPAIGN.md
```
This proves the pipeline works without touching a GPU. Do it first, always.

### Recipe B — Profile one sequence on the workstation (counters + timeline)
```bash
ssh ndpvslam@dell-workstation
cd ~/Projects/cuvslam-stack
# lock clocks (measurements need this):
sudo nvidia-smi -pm 1; sudo nvidia-smi -lgc 1620,1620; sudo nvidia-smi -lmc 7001,7001
# run the one-command pipeline (nsys + ncu + analysis):
NCU_BIN=~/ncu2025/ncu setsid nohup \
  profiling/run_characterization.sh profiling/configs/campaign/tum_fr3_long_office_household_slam.toml \
  --hw profiling/hw/dellworkstation_sm89.toml > ~/char.log 2>&1 &
tail -f ~/char.log
```
(Read `profiling/run_characterization.sh` first — it shows the exact nsys/ncu
invocations and where derived CSVs land.)

### Recipe C — Capture a data-structure attribution trace (the new pipeline)
The scripted version is `profiling/campaign/ws_attribution_capture.sh` (single
sequence) or `ws_attribution_all.sh` (all). To do it by hand for one sequence:
```bash
ssh ndpvslam@dell-workstation
cd ~/Projects/cuvslam-stack
export PATH=/opt/cuda/bin:$PATH        # nvdisasm for NVBit
TOOL=$PWD/external_repos/nvbit_release_x86_64/tools/mem_trace/mem_trace.so
PY=./cuvslam_venv_tagged/bin/python    # the INSTRUMENTED wheel
CFG=profiling/configs/campaign/tum_fr3_long_office_household_slam.toml

# pass 1: launch map + journals, near-native (LAUNCH_END=0 = instrument nothing)
LAUNCH_BEGIN=0 LAUNCH_END=0 CUDA_INJECTION64_PATH=$TOOL \
  MEM_TRACE_ALLOC_LOG=/mnt/data/attribution_out/manual/nvbit_allocs.csv \
  CUVSLAM_ALLOC_LOG=/mnt/data/attribution_out/manual/cuvslam_allocs.csv \
  $PY run.py $CFG 2>/dev/null | zstd -3 -T0 -o /mnt/data/attribution_out/manual/launchmap.txt.zst

# pass 2: trace a window (pick LAUNCH_BEGIN/END from the launch map)
LAUNCH_BEGIN=99200 LAUNCH_END=99960 CUDA_INJECTION64_PATH=$TOOL \
  MEM_TRACE_ALLOC_LOG=/mnt/data/attribution_out/manual/pass2_nvbit.csv \
  CUVSLAM_ALLOC_LOG=/mnt/data/attribution_out/manual/pass2_cuvslam.csv \
  $PY run.py $CFG 2>/dev/null | zstd -3 -T0 -o /mnt/data/attribution_out/manual/trace.txt.zst

# resolve (symbolize the journal; needs addr2line + the exact .so, so run on ws)
cd profiling
python3 -m analysis.attribution resolve /mnt/data/attribution_out/manual/pass2_cuvslam.csv \
    --out /mnt/data/attribution_out/manual/alloc_table.csv
# join (attribute the trace; stdlib, runs anywhere)
python3 -m analysis.attribution join /mnt/data/attribution_out/manual/trace.txt.zst \
    /mnt/data/attribution_out/manual/alloc_table.csv \
    /mnt/data/attribution_out/manual/pass2_nvbit.csv \
    --out /mnt/data/attribution_out/manual/join
# read /mnt/data/attribution_out/manual/join/attribution.csv
```
**Interpreting `attribution.csv`:** columns are
`kernel, tag, warp_accesses, sectors, bytes, pct_kernel_traffic`. `shared_onchip`
and `local_spill` are memory-space buckets (not DRAM data structures);
everything else is a named data structure; `unmapped` is global traffic that
fell in no live allocation (should be <7%, usually <1% — if it's high, your
alloc_table doesn't match the trace's process, e.g. wrong run or ASLR mismatch).

### Recipe D — Commit and push (match the existing history)
Commits are authored as `mij001`, **no AI-attribution trailers**:
```bash
git -c user.name=mij001 -c user.email=mij001@users.noreply.github.com \
    commit -m "profiling: <what and why>"
git push origin main
```
The local `git config user.email` is a throwaway (`no@way`); always override per
commit as above. Big raw traces stay on the workstation / sda; commit only the
small derived CSVs and reports.

---

## Part 10 — What to do next (open work, with enough detail to execute)

In priority order:

1. **When the campaign finishes: synthesize it.** Each
   `/mnt/data/attribution_out/<seq>/join_*/attribution.csv` is per-sequence.
   Write a small synthesizer (mirror `analysis/campaign.py`'s style) that pools
   them into a cross-sequence data-structure attribution table: for each named
   tag, its share of global traffic across sequences, and its stability. Then
   extend `reports/2026-07-04_attribution/ATTRIBUTION.md` with the multi-sequence
   result and copy the small CSVs into the report dir (raw traces stay on sda).
   This turns the single-sequence attribution into a generalization claim, the
   same way the 27-seq campaign generalized the kernel classification.

2. **Re-derive Slice-3 locality with the memory-space filter**
   (PUBLISHABILITY row "NEW2"; task tracker #8). `analysis/locality.py` currently
   parses *all* opcodes, so its footprints/reuse-CDFs blend shared-memory tiles
   and register-spill with global traffic — the same error Part 2.5 describes,
   but in the *locality* analysis instead of attribution. Add the opcode-space
   filter to `locality.py` (bucket LDS/STS and LDL/STL out of the global
   locality computation, exactly like `attribution.py` does), re-run it on the
   kept traces (`~/slice3/*.zst` on the workstation) and the new campaign traces,
   and update `reports/2026-07-04_slice3_locality/` plus the PUBLISHABILITY row.
   Expect the st_track "coalesced streaming" verdict to be revealed as
   **spill-dominated** — consistent with, but sharper than, the current wording.

3. **Get a LICENSE decision** (PUBLISHABILITY #10). This is a *human* decision
   (the cuVSLAM wheel EULA interacts with repo licensing) — surface it, don't
   pick one unilaterally. Artifact evaluation for a paper requires it.

4. **Phase 4 — the architecture paper track** (PUBLISHABILITY #5, #6):
   - **Accel-Sim NDP config:** feed the NVBit traces into Accel-Sim
     (`blocked/run_accelsim.sh`) configured as a near-data-processing substrate
     (reduced L2, bank-level bandwidth) and report the *steady-state hit-rate and
     performance deltas* vs the baseline GPU config.
   - **AccelWattch energy:** get per-kernel joules, so the PiM/ISP argument has
     an energy number (PiM's headline win). NVML whole-run power sampling is a
     cheaper first step you can add to `harness/profile.py` today.
   - Then the substrate design itself (the MICRO/ASPLOS paper).

---

## Part 11 — Glossary (every acronym in one place)

- **PiM** — Processing-in-Memory: compute inside/near DRAM.
- **ISP** — In-Storage Processing: compute inside the SSD.
- **NDP** — Near-Data Processing: umbrella term for PiM + ISP.
- **SLAM** — Simultaneous Localization And Mapping.
- **VO / odometry** — Visual Odometry: per-frame motion estimation (front-end).
- **BA** — Bundle Adjustment: the big least-squares pose+point optimization.
- **loop closure** — recognizing a revisited place, correcting drift.
- **keyframe** — a selected representative frame stored in the long-term map.
- **DAMOV** — the data-movement-bottleneck analysis methodology we adapt.
- **roofline** — the compute-vs-bandwidth ceiling plot.
- **arithmetic intensity** — FLOPs per byte moved from DRAM.
- **reuse distance / stack distance** — distinct locations touched since a
  location's previous access; drives cache-hit-rate prediction.
- **warp** — 32 GPU threads that execute together.
- **coalesced / scattered** — whether a warp's 32 addresses are contiguous.
- **converged / divergent** — whether all 32 lanes of a warp are active.
- **memory spaces:** global (DRAM heap), shared (on-chip scratchpad), local
  (per-thread register-spill window).
- **grid launch id** — a monotonically increasing id NVBit assigns to each
  kernel launch; used for trace windowing.
- **nsys / ncu / NVBit** — the timeline / counter / address-trace tools (Part 4).
- **NVTX** — source-level named ranges nsys can attribute kernels to.
- **CoV** — coefficient of variation (stddev/mean); our locked-clock stability.
- **G1–G7** — the GPU-DAMOV kernel classes (see METHODOLOGY.md).
- **UVA** — Unified Virtual Addressing: pinned host memory is device-addressable.
- **cu12 / cu13** — CUDA-12 / CUDA-13 build variants of the cuVSLAM wheel.

---

## Part 12 — Gotchas learned the hard way (read before you debug)

- **NVBit needs driver ≤ 575.** Do not upgrade the workstation driver. The whole
  575 / CUDA-12.9 / ncu-2025.2 / nsys-2025.3 stack is a carefully balanced set;
  see Part 3.2. `~/driver-rollback/revert.sh` re-breaks NVBit — don't run it.
- **Never trace for timing.** NVBit slows execution 10–1000×. Timing comes from
  nsys/ncu at locked clocks; traces are for *addresses only*.
- **Always window/filter a trace.** A full-run trace is TB-scale. Use
  `LAUNCH_BEGIN/END` and/or `KERNEL_FILTER`. `LAUNCH_END=0` = fast launch map.
- **Memory spaces (Part 2.5).** If your attribution or locality shows most
  traffic "unmapped," you are almost certainly counting shared/local accesses.
  Filter by opcode.
- **`addr2line` needs a `RelWithDebInfo` build.** A `Release` build strips
  symbols (`-s`) and every backtrace resolves to `??`.
- **Re-lock clocks after `free_gpu.zsh` or any reboot/power cut.** They reset.
- **sda2 NTFS dirty bit** blocks `mount -o rw`; use `mount -t ntfs3 -o rw,force`
  (Part 3.2). The campaign script does this for you.
- **`setsid nohup … &` for anything over a few minutes** (Part 3.2, the iron
  rule) and make it resumable.
- **Tailscale re-auth is a human-only unblock.** If SSH prints a
  `login.tailscale.com/...` URL and hangs, escalate to the human.
- **Commit as `mij001`, no AI trailers** (Recipe D).
- **The instrumented venv is `cuvslam_venv_tagged`, not `cuvslam_venv`.** Using
  the wrong one gives you traces with no allocation journal.

---

## Part 13 — First 30 minutes as the new maintainer (do this now)

1. Read this file (done) and skim `profiling/PROJECT_STATUS.md`.
2. Recipe A on the laptop — prove the analysis layer runs GPU-free and the tests
   pass. If they don't, fix that before anything else.
3. `ssh ndpvslam@dell-workstation` and confirm you can reach it. If it prints a
   Tailscale login URL, get the human to clear it.
4. Check the campaign: `cat /mnt/data/attribution_out/PROGRESS` and
   `tail -n 40 ~/attribution_campaign.log`. If it stopped before `CAMPAIGN DONE`,
   relaunch it (Part 8).
5. Open `reports/2026-07-04_attribution/ATTRIBUTION.md` and read it against
   Part 7 here — that's the newest science and the shape of what you'll extend.
6. When the campaign finishes, start open task #1 (Part 10): synthesize the
   per-sequence attribution CSVs into a cross-sequence table and extend the
   report.

You now have everything: the *why*, the *how*, the *machines*, the *tools*, the
*results*, and the *next steps*. Keep this file updated as the project moves —
the next person (or you, in six months) will thank you.
```
