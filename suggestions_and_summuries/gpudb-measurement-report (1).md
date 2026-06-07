# How the GPU Databases Are Measured — A Self-Contained Lecture on `gpudb-char-and-opt`

*Read this as if you were sitting in a course. I assume you are an EEE undergraduate: you know clocks, registers, caches, pipelines, and what a hardware performance-counter (PMU) is on a CPU. I do not assume you know GPUs or databases. Everything is taken from the code in the repository you uploaded. Where the repo deliberately omits something, I say so.*

**How to use this document.** Sections 1–3 are the foundation (hardware, the tools, the pipeline). **Section 4 is the heart**: it teaches you to *read* any NVIDIA metric name and then walks through **all 110 metrics** the harness collects, grouped into nine families. **Section 5** turns those raw counters into the paper's numbers, with the formulas *derived* and then *worked with real values*. Sections 6–9 explain the per-system measurement, the insights, the optimization, and the concurrency study, each taught step by step.

---

## 1. Hardware and profiling primer

### 1.1 The GPU as a machine
A CPU has a few powerful cores. A GPU has thousands of weak ALUs organised so that *throughput* (work per second over a huge batch) is enormous, even though any single thread is slow. The chip in this study is an **NVIDIA A100**. The vocabulary you must own:

- **SM (Streaming Multiprocessor)** — the GPU's "core." The A100 has **108 SMs**. Each SM is itself divided into **4 sub-partitions**, and each sub-partition has its **own warp scheduler** and its own register file. *This 4-way split is why so many metric names start with `smsp` (SM sub-partition) rather than `sm`.*
- **Thread** — the smallest unit of work; runs one query's worth of arithmetic on one tuple (roughly).
- **Warp** — a bundle of **32 threads** that execute *the same instruction together*, each on its own data. This is SIMT (Single Instruction, Multiple Threads). If a branch is true for some lanes and false for others, the false lanes are switched off for that instruction — they are **predicated off** and do no useful work. (Remember this word; it appears in metric names.)
- **Warp scheduler** — every clock cycle it tries to pick one *ready* warp and issue its next instruction. If nothing is ready (all warps waiting on memory), the cycle is wasted. The health of this scheduler is a central theme.
- **Block (CTA)** — a group of warps that share an SM and can synchronise via a barrier (`__syncthreads`). A kernel launches a **grid** of blocks.
- **Kernel** — one GPU function launch. *One SQL query becomes anywhere from 4 kernels (Crystal) to ~190 (TQP).* The profiler measures **each kernel separately** — a fact that forces a key design decision in §5.

### 1.2 The memory hierarchy (same idea as CPU caches, different sizes)
From fastest/smallest to slowest/largest:

| Level | Scope | Rough size (A100) | EEE analogy |
|---|---|---|---|
| Registers | per thread | 256 KB file/SM | the register file in your datapath |
| L1 cache / shared memory | private to one SM | ~192 KB/SM | L1 cache |
| L2 cache | **shared by all 108 SMs** | **~40 MB** | shared L2/L3 |
| DRAM (HBM2) | whole GPU | 40 GB | main memory, but ~1.5 TB/s |

Two granularity facts you must internalise, because metric names depend on them:
- A **cache line is 128 bytes**, made of **4 sectors of 32 bytes each**. Caches are looked up *per sector*, so "hit rate" is a *sector* hit rate.
- A memory **request** is one memory instruction's access; one request may touch several sectors.

### 1.3 What a "metric" is
Exactly like a CPU's PMU exposes counters (cache misses, mispredicts, cycles), the GPU exposes hundreds of hardware event counters. A profiler reads them out of the silicon while your program runs. **The counters count physical events, so they don't care whether the software was hand-written CUDA, JIT-compiled, or PyTorch.** That neutrality is precisely what lets us compare four very different databases fairly.

### 1.4 The three NVIDIA tools (and which the repo really uses)
| Tool | Role | Used here? |
|---|---|---|
| **Nsight Compute** (`ncu`) | Deep per-*kernel* counter profiling. | **Yes — the entire pipeline.** |
| **Nsight Systems** (`nsys`) | System-wide *timeline*. | Vestigial only (see §10). |
| **`nvidia-smi`** | Admin/control: lock clocks, slice the GPU (MIG), list devices. *Not a profiler.* | **Yes — clock-lock + concurrency.** |

### 1.5 Two models you need before §5
- **Arithmetic Intensity (AI)** `= compute_operations / bytes_read` (units: ops/byte). Low AI ⇒ you fetch a lot and compute little ⇒ **memory-bound**. High AI ⇒ **compute-bound**.
- **Roofline model.** Plot AI on the x-axis, achieved throughput (ops/s) on the y-axis. Two ceilings bound the plot: a *sloped* line = peak memory bandwidth (α), and a *flat* line = peak compute (β). They cross at AI = β/α. A query left of the crossover is memory-bound; right of it, compute-bound. The vertical gap between your dot and the ceiling above it = how much performance you're leaving on the table. *This paper draws **two** rooflines — one for DRAM, one for L2 — because the same query has different AI at each memory level (proved numerically in §5.5).*

---

## 2. The experimental skeleton and its controls

- **One GPU, one workload (SSB's 13 queries `q11…q43`), four systems.** The query list is hard-coded identically in every script.
- **Scale factor (data size).** `#define SF 16` in `crystal/crystal_src/src/ssb/ssb_utils.h` for Crystal; a `[sfph]` placeholder in the SQL for the others. **SF = 16 is the headline.** Scripts can sweep SF = 1…16.
- **Warm vs. cold.** The study targets *warm* execution (data already on the GPU, query already compiled) to isolate raw GPU efficiency; the code implements this per system (§6).
- **The crucial control — clock locking.** GPU clocks normally boost/throttle, which would make per-cycle counters non-reproducible. The repo pins the clock to **1410 MHz** with `nvidia-smi -lgc 1410`, and tells the profiler to keep its hands off with the switch **`--clock-control none`**. Why it matters: many counters are *per cycle*; to convert to *per second* you need the clock to be a known constant.

---

## 3. The measurement pipeline, end to end

```
characterization_script/run_<system>.sh     # loops queries, calls ↓
   └─ <system>/run_query.py  --ncu           # starts DB, hands the binary to ↓
        └─ utility/ncu_profiler.py            # builds the real `ncu` command
             │  (metric list from utility/counter_config.py)
             └─ ncu ... -o gpudb-perf <bin>   # profiler runs, writes report
                  └─ gpudb-perf.ncu-rep       # one binary report per query
                       └─ report_parser/ncu_parser.py   # `ncu --import ... --csv`, parse
                            ├─ stats/ncu_export.py       # human-readable per-kernel dump
                            └─ stats/flush_ncu_csv.py    # → res/*.txt (the figures' numbers)
```

### 3.1 The exact `ncu` command and every switch
From `utility/ncu_profiler.py`:
```
ncu  [-s <skip>] [-c <count>] [<extra-flags>] \
     --metrics <union of all 110 counters> \
     --apply-rules yes  --clock-control none \
     -f  -o gpudb-perf  <database binary>
```
| Switch | Meaning | Rationale |
|---|---|---|
| `--metrics <list>` | collect exactly these counters | naming a fixed list keeps replays (see below) cheap and guarantees identical treatment of every system |
| `--apply-rules yes` | run Nsight's built-in analysis rules | enriches the human-readable export with NVIDIA's own derived guidance |
| `--clock-control none` | profiler must NOT lock/reset the clock | the clock is already pinned to 1410 MHz externally; keeps per-cycle→per-second conversions valid |
| `-f` | force-overwrite the report | each query overwrites then is moved aside |
| `-o gpudb-perf` | output file `gpudb-perf.ncu-rep` | the parser looks for this exact name |
| `-s <N>` | **skip** the first N kernels | skip start-up/data-loading kernels so only the warm query is measured (§6) |
| `-c <N>` | profile only N kernels | bound capture to a slice of execution |
| `--target-processes all` (BlazingSQL) | follow into child processes | BlazingSQL's real GPU work runs in spawned subprocesses |

**A subtlety: kernel replay.** The GPU can only count a limited number of events simultaneously. `ncu` therefore *replays each kernel several times*, collecting a different subset of the 110 metrics each pass, then stitches them together. This is fine here because database kernels are deterministic given fixed cached data — replays produce identical work. It is also *why* you don't ask for "all metrics": each extra group can add replay passes and slow profiling dramatically.

The parser reads the report back with a second call, `ncu --import gpudb-perf.ncu-rep --csv`, and normalises every value to base units (bytes, seconds, cycles/second) so the formulas in §5 are dimensionally consistent.

---

## 4. The metric catalogue — every counter, explained

### 4.0 First, learn to *read* a metric name

NVIDIA metric names look terrifying but follow a strict grammar:

```
   unit  __  counter             .  rollup   .  submetric
   sm        throughput             avg          pct_of_peak_sustained_elapsed
   lts       t_sector_hit_rate                   pct
   smsp      warps_eligible         avg          per_cycle_active
```

**(a) The `unit` prefix — *where* on the chip the event is counted:**

| Prefix | Where | Think of it as |
|---|---|---|
| `gpu` | the whole GPU | the chip |
| `gpc` | a Graphics Processing Cluster (a group of SMs) | a cluster of cores |
| `sm` | one Streaming Multiprocessor (whole) | one core |
| `smsp` | an SM **sub-partition** (4 per SM, each with its own scheduler) | one lane-group/scheduler inside a core |
| `l1tex` | the combined **L1 cache + texture + load/store** unit (per SM) | the L1 + LSU |
| `lts` | an **L2** cache slice ("L-Two-Slice") | shared L2 |
| `dram` | device DRAM | main memory |
| `launch` | a property of the kernel *launch* (not a counter you replay) | the launch configuration |

**(b) The `counter` — *what* is counted.** Common stems:
- `cycles_elapsed` / `cycles_active` — clock cycles (see "elapsed vs active" below).
- `inst_executed` / `inst_issued` — instructions executed / issued (warp-level).
- `sass_thread_inst_executed_op_<TYPE>` — *thread-level* machine (SASS) instructions of a given op type (`integer`, `fadd`, `ffma`, `dfma`, …). "SASS" = the GPU's real assembly (below PTX).
- `t_requests` / `t_sectors` — memory *requests* / *sectors* at a cache's tag (`t`) stage.
- `warps_active` / `warps_eligible` / `issue_active` — scheduler states.
- `bytes_read` / `bytes_write` / `bytes` — DRAM traffic.

**(c) The `rollup` — how multiple instances combine:** `.sum` (add across all SMs/sub-partitions), `.avg` (average), `.max`, `.min`.

**(d) The `submetric` — a final transform:**
| Suffix | Meaning |
|---|---|
| `.peak_sustained` | the hardware's *theoretical peak* for this counter, per cycle |
| `.per_cycle_elapsed` | achieved rate per **elapsed** cycle |
| `.per_cycle_active` | achieved rate per **active** cycle |
| `.per_second` | achieved rate per second |
| `.pct_of_peak_sustained_elapsed` / `_active` | achieved as a % of peak |
| `.pct` | a percentage (e.g. a hit rate) |
| `.ratio` | a dimensionless ratio |
| `.lookup_hit` / `.lookup_miss` | cache hit/miss outcomes |

**Four concepts that recur (learn these once):**
1. **Elapsed vs. active cycles.** *Elapsed* = every cycle the kernel was running, including cycles a unit sat idle. *Active* = only cycles that unit was doing something. So `per_cycle_elapsed` gives **real throughput** (idle dilutes it — what you actually got), while `per_cycle_active` gives **efficiency when busy**. The roofline uses *elapsed* on purpose, so idle time correctly drags the dot below the ceiling.
2. **Warp-instruction vs. thread-instruction.** `inst_executed` counts one per warp per issue (up to 1 instruction for 32 lanes). `thread_inst_executed` counts at the lane level (up to 32 per warp-instruction). Their ratio = average live lanes per instruction (max 32). `pred_on` variants count only lanes that were *not* predicated off — i.e. *useful* lanes.
3. **Request vs. sector.** A *request* is one memory instruction's trip to the cache; a *sector* is a 32-byte chunk it moved. Hit rate is per sector; bandwidth is per byte (= sectors × 32, or, as the authors approximate for L2, requests × 128).
4. **`peak_sustained` vs. achieved.** Peak = what the silicon *could* do per cycle at best; achieved = what it actually did. The roofline ceilings come from peak; the dot's height comes from achieved.

Now the nine families. For each metric I give: **identifier → label → what it physically counts → why it's collected / how it's used.**

---

### 4.1 Family 1 — `metric_sol()` "Speed of Light" (8 metrics)
*Purpose:* a one-glance "how close to peak is each part of the chip, and how long did the kernel take." "Speed of Light" is NVIDIA's term for "% of the theoretical maximum."

| Identifier | Label | What it counts / why |
|---|---|---|
| `gpu__time_duration.sum` | Duration | Wall-clock time the kernel ran. **The backbone of every aggregation in §5** — it converts per-second rates back to totals and weights kernels. |
| `gpc__cycles_elapsed.max` | Elapsed Cycles | Clock cycles spanned by the kernel (max across clusters). Duration measured in cycles. |
| `sm__cycles_active.avg` | SM Active Cycles | Cycles an SM had ≥1 warp resident. *Elapsed − active = idle*; a big gap means the SM was starved. |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Compute (SM) Throughput | The busiest SM sub-unit as a % of its peak. The headline "compute utilisation." |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | Memory Throughput | Combined memory-pipeline utilisation as % of peak. |
| `l1tex__throughput.avg.pct_of_peak_sustained_active` | L1/TEX Cache Throughput | How hard the L1/LSU worked vs. its peak. |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | L2 Cache Throughput | How hard L2 worked vs. peak. High here = L2-bandwidth-bound (a key finding). |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | DRAM Throughput | DRAM bandwidth as % of peak. High = DRAM-bound. |

*Teaching point:* if `lts` throughput is ~100% while `dram` throughput is low, the query lives in L2 — exactly the situation that motivates the second roofline.

---

### 4.2 Family 2 — `metric_roofline()` (13 metrics) — the roofline inputs
*Purpose:* supply the achieved and peak operation rates and the clock so §5 can place a dot. There are FP32, FP64, and INT variants; **only the INT and clock ones drive the model** (OLAP is integer work), but the FP ones are collected to *prove* the FP units are idle.

| Identifier | Label | Role |
|---|---|---|
| `sm__sass_thread_inst_executed_op_integer_pred_on.sum.peak_sustained` | **Peak INT** | integer thread-ops/cycle the whole GPU can sustain ⇒ the **compute ceiling β** |
| `smsp__sass_thread_inst_executed_op_integer_pred_on.sum.per_cycle_elapsed` | **Achieved INT** | integer thread-ops/cycle actually achieved (useful lanes only) ⇒ feeds total-ops |
| `smsp__cycles_elapsed.avg.per_second` | **Cycles Per Second** | the measured clock (~1.41×10⁹) ⇒ converts per-cycle to per-second |
| `dram__bytes.sum.per_second` | DRAM Bytes Per Second | achieved DRAM bandwidth (cross-check on the memory axis) |
| `sm__...op_ffma_pred_on...peak_sustained` | Peak FP32 (FFMA) | peak fused-multiply-add rate (FP32) — *collected, not used in model* |
| `sm__...op_fmul_pred_on...peak_sustained` | Peak FP32 (FMUL) | peak FP32 multiply |
| `sm__...op_dfma_pred_on...peak_sustained` | Peak FP64 (DFMA) | peak FP64 fused-multiply-add |
| `smsp__...op_fadd_pred_on...per_cycle_elapsed` | Achieved FP32 FADD | achieved FP32 add (≈0 for OLAP) |
| `smsp__...op_fmul_pred_on...per_cycle_elapsed` | Achieved FP32 FMUL | achieved FP32 multiply |
| `smsp__...op_ffma_pred_on...per_cycle_elapsed` | Achieved FP32 FFMA | achieved FP32 FMA |
| `smsp__...op_dadd_pred_on...per_cycle_elapsed` | Achieved FP64 DADD | achieved FP64 add |
| `smsp__...op_dmul_pred_on...per_cycle_elapsed` | Achieved FP64 DMUL | achieved FP64 multiply |
| `smsp__...op_dfma_pred_on...per_cycle_elapsed` | Achieved FP64 DFMA | achieved FP64 FMA |

*Why integer-only is the right call:* an FP32-FMA ceiling on the A100 is ~19.5 TFLOP/s — astronomically above what an integer-only query can reach. Putting that ceiling on the chart would make every query look hopelessly compute-underutilised. The integer ceiling (`Peak INT`) is the relevant one, and the achieved FP counters near zero confirm no floating-point work is happening.

---

### 4.3 Family 3 — `metric_memory()` (24 metrics) — where the bytes live
*Purpose:* explain *why* a query is memory-bound — which level serves the traffic, how wide the accesses are, and the hit rates.

**(i) DRAM traffic**
| Identifier | Label | Use |
|---|---|---|
| `dram__bytes_read.sum` | Total Bytes Read | **the byte count for the DRAM roofline** and for Figure 8 (bottom) |
| `dram__bytes_write.sum` | Total Bytes Write | writes (tiny for aggregating queries) |
| `dram__bytes_read.sum.pct_of_peak_sustained_elapsed` | DRAM Read Peak | read bandwidth as % of peak |

**(ii) Memory-instruction widths** — how many bytes each memory instruction moved. Counts of warp memory-instructions by access width:
`smsp__sass_inst_executed_op_memory_8b.sum` (8-bit), `…_16b`, `…_32b`, `…_64b`, `…_128b` → labels **8/16/32/64-/128-bit Warp Insts**. *Why:* narrow accesses (8/16-bit) waste bandwidth (you fetch a 32-byte sector to use 1–2 bytes); wide, aligned 128-bit accesses *coalesce* well. The width histogram reveals how friendly the access pattern is to the memory system.

**(iii) L1/LSU requests & sectors** — split by global vs. local memory and load vs. store. *Global* = the shared data arrays; *local* = per-thread spill (registers that didn't fit). Requests = number of LSU trips; sectors = 32-byte chunks moved.
`l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` (Global Load Requests), `…_global_op_st` (Global Store Requests), `…_local_op_ld` (Local Load Requests), `…_local_op_st` (Local Store Requests); and the sector versions `l1tex__t_sectors_pipe_lsu_mem_global_op_ld/st.sum`, `…_local_op_ld/st.sum`. *Why:* lots of **local** load/store traffic is a red flag — it means register spilling. The request-vs-sector gap shows how scattered the accesses are (many sectors per request = poor coalescing).

**(iv) L2 requests, sectors, hits/misses**
| Identifier | Label | Use |
|---|---|---|
| `lts__t_requests_srcunit_tex_op_read.sum` | **L2 Read Requests** | **the L2 byte estimate for the L2 roofline** (× 128, §5.5) |
| `lts__t_requests_srcunit_tex_op_write.sum` | L2 Write Requests | L2 write traffic |
| `lts__t_sectors_srcunit_tex_op_read.sum` | L2 Read Sectors | 32-byte chunks read at L2 |
| `lts__t_sectors_srcunit_tex_op_write.sum` | L2 Write Sectors | chunks written |
| `lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum` | L2 Lookup Hit | sectors that hit in L2 |
| `lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum` | L2 Lookup Miss | sectors that missed (went to DRAM) |

**(v) Hit rates (the two numbers in Table 6)**
| Identifier | Label | Use |
|---|---|---|
| `l1tex__t_sector_hit_rate.pct` | **L1/TEX Hit Rate** | fraction of L1 sector lookups that hit — *Crystal's is ~0.47%, the bug the optimization fixes* |
| `lts__t_sector_hit_rate.pct` | **L2 Hit Rate** | fraction of L2 sector lookups that hit |

*Teaching point:* `srcunit_tex` means "requests arriving at L2 *from the L1/texture unit*" — i.e. ordinary global loads that missed L1 and fell through to L2. That's exactly the traffic we want for the L2 roofline.

---

### 4.4 Family 4 — `metric_compute()` (19 metrics) — which compute pipes are busy
*Purpose:* once you know a query is *not* purely memory-bound, find which execution pipe limits it. Each SM has several specialised pipelines; a metric reports each one's utilisation as % of peak.

**(i) Instruction throughput / IPC**
| Identifier | Label | Meaning |
|---|---|---|
| `sm__inst_executed.avg.per_cycle_elapsed` | Executed IPC Elapsed | instructions/cycle including idle (real IPC) |
| `sm__inst_executed.avg.per_cycle_active` | Executed IPC Active | IPC while the SM is busy |
| `sm__instruction_throughput.avg.pct_of_peak_sustained_active` | SM Busy | overall instruction throughput vs. peak |
| `sm__inst_issued.avg.pct_of_peak_sustained_active` | Issue Slots Busy | how full the issue stage is |
| `sm__inst_issued.avg.per_cycle_active` | Issued IPC Active | issued IPC while busy |

*Reading it:* on the A100 each SM can issue up to ~4 instructions/cycle. If "Executed IPC Elapsed" is far below that, the SM spends most cycles issuing nothing — the *issue-bound* story you'll see in §4.7 and §7.

**(ii) Per-pipe utilisation** (each `% of peak sustained active`). You need to know what each pipe does:
| Identifier (suffix `…avg.pct_of_peak_sustained_active`) | Label | Pipe's job | Expect for OLAP? |
|---|---|---|---|
| `sm__pipe_alu_cycles_active` | ALU Utilization | integer & logic arithmetic | **busy** (integer work) |
| `sm__pipe_fma_cycles_active` | FMA Utilization | fused multiply-add; also integer multiply | moderately busy |
| `sm__inst_executed_pipe_lsu` | LSU Utilization | load/store unit (issues memory ops) | **busy** (memory-bound) |
| `sm__inst_executed_pipe_adu` | ADU Utilization | address/branch divergence unit | some |
| `sm__inst_executed_pipe_cbu` | CBU Utilization | convergence/branch unit (sync, reconverge) | some |
| `sm__inst_executed_pipe_xu` | XU Utilization | special-function/conversion unit | low |
| `sm__inst_executed_pipe_uniform` | Uniform Utilization | uniform datapath (scalar ops shared by a warp) | low |
| `sm__inst_executed_pipe_tex` | Tex Utilization | texture pipe | low |
| `sm__inst_executed_pipe_fp16` | FP16 Utilization | half-precision FP | ~0 |
| `sm__inst_executed_pipe_fp64` | FP64 Utilization | double-precision FP | ~0 |
| `sm__inst_executed_pipe_tensor_op_dmma` | Tensor (DP) Utilization | tensor-core FP64 matrix | ~0 |
| `sm__inst_executed_pipe_tensor_op_hmma` | Tensor (FP) Utilization | tensor-core FP16/TF32 matrix | ~0 |
| `sm__inst_executed_pipe_tensor_op_imma` | Tensor (INT) Utilization | tensor-core INT matrix | ~0 |

*Teaching point:* seeing **ALU + LSU busy, every FP/Tensor pipe ≈ 0** is independent confirmation that an integer-only roofline is the correct model — the floating-point and tensor hardware is simply unused by SQL analytics.

---

### 4.5 Family 5 — `metric_occupancy()` (8 metrics) — how full the SMs are
*Background you need:* **occupancy** = (resident warps) / (the hardware maximum, 64 warps/SM on A100). Higher occupancy = more warps to hide memory latency. But occupancy is *capped* by whichever resource runs out first: registers, shared memory, warp slots, or block slots.

| Identifier | Label | Meaning |
|---|---|---|
| `sm__maximum_warps_per_active_cycle_pct` | Theoretical Occupancy | the best occupancy this kernel *could* reach given its resource use |
| `sm__maximum_warps_avg_per_active_cycle` | Theoretical Active Warps per SM | same, expressed as warps |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved Occupancy | the occupancy actually realised |
| `sm__warps_active.avg.per_cycle_active` | Achieved Active Warps per SM | actual resident warps |
| `launch__occupancy_limit_registers` | Block Limit Register | max blocks/SM allowed by register pressure |
| `launch__occupancy_limit_shared_mem` | Block Limit Shared Mem | max blocks/SM allowed by shared-memory use |
| `launch__occupancy_limit_warps` | Block Limit Warp | max blocks/SM allowed by warp slots |
| `launch__occupancy_limit_blocks` | Block Limit SM | max blocks/SM allowed by the block-slot hardware limit |

*Reading it:* whichever "Block Limit …" is **smallest** is the binding constraint. If `Block Limit Register` is smallest, the kernel uses too many registers/thread — connecting directly to `launch__registers_per_thread` (next family) and to why HeavyDB (64 regs/thread) may achieve lower occupancy than Crystal (40).

---

### 4.6 Family 6 — `metric_launch()` (5 metrics) — the launch configuration
*Purpose:* relate the *software's* launch choices to the hardware limits above. These are read straight from the launch (no replay needed).

| Identifier | Label | Meaning |
|---|---|---|
| `launch__grid_size` | Grid Size | number of blocks launched |
| `launch__block_size` | Block Size | threads per block |
| `launch__thread_count` | Threads | total threads = grid × block |
| `launch__registers_per_thread` | **Register Per Thread** | registers each thread uses — **Table 6: Crystal 40, HeavyDB 64**. More registers ⇒ fewer resident warps ⇒ lower occupancy ⇒ less latency hiding. |
| `launch__waves_per_multiprocessor` | Waves Per SM | how many full "waves" of blocks the SM processes; a fractional last wave = **tail effect** (some SMs idle at the end) |

*Teaching point:* a register count is not "good" or "bad" in isolation — it trades off against occupancy. The harness collects both so you can see the trade rather than guess.

---

### 4.7 Family 7 — `metric_warp()` (9 metrics) — scheduler health (Table 5)
*Purpose:* the numbers behind the paper's Table 5. Each SM sub-partition's scheduler can issue **one warp per cycle** at best. These metrics tell you, *per scheduler per cycle*, how many warps were in each state.

| Identifier | Label | Meaning |
|---|---|---|
| `smsp__warps_active.avg.per_cycle_active` | **Active Warps Per Scheduler** | warps resident & assigned to this scheduler |
| `smsp__warps_eligible.avg.per_cycle_active` | **Eligible Warps Per Scheduler** | warps *ready to issue* this cycle |
| `smsp__issue_active.avg.per_cycle_active` | **Issued Warps Per Scheduler** | warps actually issued (≤1) |
| `smsp__warps_active.avg.peak_sustained` | GPU Max Warps Per Scheduler | hardware max (16 on A100: 64 warps/SM ÷ 4 schedulers) |
| `smsp__maximum_warps_avg_per_active_cycle` | Theoretical Warps Per Scheduler | max this kernel could keep resident |
| `smsp__average_warp_latency_per_inst_issued.ratio` | Warp Cycles Per Issued Instruction | average cycles a warp waits per issued instruction (latency) |
| `smsp__average_warps_active_per_inst_executed.ratio` | Warp Cycles Per Executed Instruction | warp-cycles spent per executed instruction |
| `smsp__thread_inst_executed_per_inst_executed.ratio` | Avg Active Threads Per Warp | live lanes per instruction (≤32) |
| `smsp__thread_inst_executed_pred_on_per_inst_executed.ratio` | Avg Not-Predicated-Off Threads Per Warp | *useful* lanes per instruction (≤32) |

*The key reading (Table 5):* the paper finds **Active ≈ 11, but Eligible ≈ 0.16** for Crystal. Interpretation: there are plenty of warps resident (good occupancy), but only ~0.16 are *ready* on any given cycle, so the scheduler has nothing to issue ~5 cycles out of 6. The SM is **issue-bound, starved by memory latency**, not compute-bound. (`Active − Eligible` ≈ stalled warps.)

---

### 4.8 Family 8 — `metric_detail_warp()` (18 metrics) — *why* warps stall
*Purpose:* when a warp can't issue, the hardware records *why*. Each metric is `…average_warps_issue_stalled_<REASON>_per_issue_active.ratio` — the average number of warps stalled for that reason, normalised per issue-active cycle. **This is the family the stall pie-charts (Figure 9) are built from.** Here are all 18, with a plain-English cause:

| Reason (identifier `…_<reason>_…`) | Label | What it means |
|---|---|---|
| `long_scoreboard` | Stall Long Scoreboard | **waiting on a long-latency memory load** (data coming from L2/DRAM). *The signature of memory-bound code.* |
| `short_scoreboard` | Stall Short Scoreboard | waiting on a short-latency result (shared memory / MIO / L1) |
| `wait` | Stall Wait | waiting on a **fixed-latency arithmetic** result (e.g. an ALU op's known latency) — i.e. genuine compute dependency |
| `lg_throttle` | Stall LG Throttle | the **load/global instruction queue is full** — too many memory ops in flight |
| `mio_throttle` | Stall MIO Throttle | the memory-input/output queue is full (shared mem / special ops) |
| `math_pipe_throttle` | Stall Math Pipe Throttle | the math pipe is saturated — can't accept more |
| `tex_throttle` | Stall Tex Throttle | texture pipe queue full |
| `drain` | Stall Drain | warp finished, **draining outstanding memory writes** before exit (write-back) |
| `membar` | Stall Membar | waiting at a memory fence (`__threadfence`) |
| `barrier` | Stall Barrier | waiting at `__syncthreads()` for sibling warps |
| `branch_resolving` | Stall Branch Resolving | waiting for a branch target to be computed |
| `no_instruction` | Stall No Instruction | waiting for instruction fetch (I-cache miss / fetch latency) |
| `imc_miss` | Stall IMC Miss | miss in the immediate-constant cache (constant memory) |
| `sleeping` | Stall Sleep | warp is sleeping (e.g. after `__nanosleep`, or all lanes exited) |
| `dispatch_stall` | Stall Dispatch Stall | a resource needed at the dispatch stage is unavailable |
| `not_selected` | Stall Not Selected | warp *was eligible* but the scheduler chose another — a **healthy** "stall" (means oversubscription) |
| `selected` | Stall Selected | the warp that *was* issued this cycle (not really a stall) |
| `misc` | Stall Misc | everything else |

**Which 5 the chart uses, and why.** `stats/flush_ncu_csv.py` plots only five of these, regrouped into a story:

| Counter chosen | Chart label | Story it tells |
|---|---|---|
| `long_scoreboard` | **Mem LD** | waiting on memory loads (the dominant slice almost everywhere) |
| `wait` | **Compute** | genuine arithmetic dependency (small except in BlazingSQL) |
| `lg_throttle` | **Mem Queue Full** | memory pipe backed up |
| `drain` | **Mem WB** | end-of-kernel write-back |
| `barrier` | **Branch** | synchronisation waits |

The five were chosen because together they separate *memory-load latency* (Mem LD) from *compute dependency* (Compute) from *memory pipe pressure* (Mem Queue Full / Mem WB) from *control/sync* (Branch) — which is exactly the axis along which the four systems differ. The remaining 13 are collected but are negligible for these queries.

---

### 4.9 Family 9 — `metric_inst()` (6 metrics) — raw instruction counts
*Purpose:* totals used both for reporting and as the **weights** when averaging stalls across kernels (§5.7).

| Identifier | Label | Meaning |
|---|---|---|
| `smsp__inst_executed.sum` | **Executed Insts** | total warp-instructions executed — **the per-kernel weight for stall aggregation** |
| `smsp__inst_executed.avg` | Avg Executed Insts Per Scheduler | per-scheduler average |
| `smsp__inst_issued.sum` | Issued Insts | total instructions issued |
| `smsp__inst_issued.avg` | Avg Issued Insts Per Scheduler | per-scheduler average |
| `smsp__thread_inst_executed_per_inst_executed.ratio` | Avg Threads Per Inst | live lanes per instruction (≤32) |
| `smsp__thread_inst_executed_pred_on_per_inst_executed.ratio` | Avg Active Predicated-On Threads Per Inst | *useful* lanes per instruction |

*Teaching point:* if `Avg Active Predicated-On Threads Per Inst` is well below 32, lanes are being wasted to divergence/predication — precisely the inefficiency the **thread-termination** optimization (§8) attacks.

---

## 5. From raw counters to the paper's numbers — derivations and worked examples

### 5.0 The aggregation principle (read this first)
A query is many kernels, but a roofline dot or a "total bytes" bar is **one number**. The rule the code follows in `stats/flush_ncu_csv.py`:

> **Sum *extensive* quantities (time, bytes, operations, requests) across all of a query's kernels. Then form ratios (AI, throughput) from the sums. Never average ratios directly.**

*Why (an EEE-style argument):* bytes and operations are like charge — they add. Arithmetic intensity is like a *ratio of charges*; if you average two kernels' AIs you'd weight a tiny kernel equally with a huge one, which is wrong. Summing the numerators and denominators separately, then dividing once, is the correct weighting (it weights each kernel by its actual contribution). This is identical to how you'd compute an overall efficiency from several stages — total useful out over total in, not the mean of per-stage efficiencies.

Notation: for a query with kernels *k* = 1…K, let
`D_k` = `gpu__time_duration.sum` (s), `B_k` = `dram__bytes_read.sum` (bytes),
`a_k` = `Achieved INT` (`…per_cycle_elapsed`, ops/cycle), `f_k` = `Cycles Per Second` (cycle/s),
`R_k` = `lts__t_requests_srcunit_tex_op_read.sum` (requests), `I_k` = `smsp__inst_executed.sum`.

### 5.1 Total integer operations
For each kernel, ops = (ops/cycle) × (cycles/s) × (s):
```
ops_k = a_k · f_k · D_k                         [ops/cycle · cycle/s · s = ops]
total_ops = Σ_k ops_k
```
The units cancel cleanly to operations — that's the dimensional check. → `res/inst.txt`, Figure 8 (top).

### 5.2 Total DRAM bytes
```
total_bytes = Σ_k B_k                            [bytes]
```
→ `res/bytes.txt`, Figure 8 (bottom). (Reads only — writes are negligible for queries that output small aggregates.)

### 5.3 DRAM arithmetic intensity — with a worked example
```
AI_dram = total_ops / total_bytes               [ops/byte]
```
**Worked example — Crystal, Q21 (real numbers from the paper's Table 7):**
total_ops = 2704 M = 2.704×10⁹, total_bytes = 1653 MB = 1.653×10⁹ B.
```
AI_dram = 2.704e9 / 1.653e9 = 1.64 ops/byte
```
So Crystal's Q21 does ~1.6 integer operations for every byte it pulls from DRAM — a *low* AI, i.e. memory-leaning. → `res/roofline_dram_*.txt` x-coordinate; also `res/ai.txt`.

### 5.4 Achieved throughput (the y-coordinate)
```
total_kernel_time = Σ_k D_k                      [s]
T = total_ops / total_kernel_time / 1e9          [Gops/s]
```
The `/1e9` just converts ops/s to Gops/s for the chart.
**Illustrative number:** Crystal's Q21 is dominated by one kernel of ~3.96 ms (Table 4), so total_kernel_time ≈ 3.99 ms ≈ 0.00399 s.
```
T ≈ 2.704e9 / 0.00399 / 1e9 ≈ 0.68 ×10³ = 678 Gops/s
```
That ~678 Gops/s is the dot's height; you'd compare it to the integer-compute ceiling (β, from `Peak INT × clock`) to see how far below peak it sits.

### 5.5 L2 bytes and the L2 roofline — and *why AI differs by level*
`ncu` gives L2 **requests**, not bytes. The authors model **one L2 read request ≈ one 128-byte cache line**:
```
total_L2_bytes = (Σ_k R_k) · 128                 [bytes]
AI_l2 = total_ops / total_L2_bytes               [ops/byte]
```
*(A more literal byte count would be `L2 read sectors × 32`; both `…op_read.sum` (requests) and `…sectors…op_read.sum` are collected, but the roofline uses requests × 128 as its model.)*

**Worked example — Crystal, Q21:** L2 read requests = 127 M (Table 7).
```
total_L2_bytes = 127e6 × 128 = 1.6256e10 B ≈ 16.3 GB
AI_l2 = 2.704e9 / 1.6256e10 = 0.166 ops/byte
```
**Now compare the two intensities for the *same query*:**
```
AI_dram = 1.64 ops/byte   (only 1.65 GB reached DRAM)
AI_l2   = 0.166 ops/byte  (16.3 GB were served by L2)
```
This is the whole reason for two rooflines: **far more bytes move at L2 than at DRAM (16.3 GB vs 1.65 GB), so the L2-level AI is ~10× lower.** A query can look comfortably memory-light at the DRAM level yet be slammed against the L2-bandwidth ceiling. One chart would hide that; two charts reveal it. The throughput y-value `T` is the same on both charts — only the x-position (AI) changes.

### 5.6 Reading a dot on the roofline (the interpretation skill)
- The **sloped ceiling** is memory bandwidth: `throughput ≤ AI × peak_bandwidth`. A dot *on* this line is bandwidth-saturated.
- The **flat ceiling** is `Peak INT × clock` (compute).
- They cross at `AI* = β/α`. Left of `AI*` ⇒ memory-bound; right ⇒ compute-bound.
- The **vertical distance** from a dot up to the nearest ceiling = unrealised performance. The whole paper is about closing that gap, either by raising AI (compute less per byte → §8) or by saturating a higher ceiling.

### 5.7 Stall breakdown — instruction-weighted, with a worked example
You cannot just average two kernels' stall ratios; a kernel that ran 100× more instructions should count 100× more. So `flush_stall()` weights each kernel's stall ratio by its instruction count `I_k`:
```
for reason c:
   stall_sum[c]      = Σ_k  I_k · stall_ratio_k(c)
   per_query[c]      = stall_sum[c] / Σ_k I_k          # instruction-weighted average ratio
   share[c]          = per_query[c] / Σ_c' per_query[c']   # normalise the 5 reasons to 100%
```
**Worked example (two kernels, the "Mem LD" reason):**
Kernel A: I_A = 90 M insts, Mem-LD stall ratio = 0.80.
Kernel B: I_B = 10 M insts, Mem-LD stall ratio = 0.20.
Naïve average = (0.80+0.20)/2 = 0.50 — *wrong*, it pretends B matters as much as A.
Weighted:
```
stall_sum = 90e6·0.80 + 10e6·0.20 = 72e6 + 2e6 = 74e6
per_query = 74e6 / (90e6+10e6) = 0.74
```
0.74, correctly dominated by the big kernel. Then divide by the sum over the five reasons to get the pie-slice percentage. → `res/stall_<system>.txt`, Figure 9.

### 5.8 Top-kernel share — with a worked example
`flush_top_kernel()` sums each kernel-name's duration, sorts, takes the top 3, and reports each as a fraction of total time:
```
share(kernel) = time(kernel) / Σ time(all kernels)
```
**Worked example — Crystal Q21 (Table 4):** the `probe_ht` kernel ran 3.96 ms out of a 3.99 ms total ⇒ share = 3.96/3.99 = **0.9935 = 99.35%**. One fused kernel is essentially the whole query — the fingerprint of aggressive kernel fusion. Contrast BlazingSQL/TQP, where the top kernel is only ~38–41% because work is smeared over ~170–190 kernels.

### 5.9 Engineering aside — unit normalisation
Before any of the above runs, `report_parser/ncu_parser.py` converts every value `ncu` emits into base units: `Gbyte→byte`, `usecond→second`, `cycle/nsecond→cycle/second`, etc. This is what makes the multiplications in §5.1 dimensionally valid (you're always multiplying ops/cycle by cycle/second by seconds). Ratios like AI are immune to a consistent scale error anyway, since the scale cancels in numerator and denominator.

---

## 6. Per-system measurement — same counters, different plumbing

The 110 counters are identical for every system. What changes is **how you start the database** and **how you isolate the warm query from start-up noise.** That isolation is the part that most affects correctness.

### 6.1 Crystal — `crystal/run_query.py`
- **One binary per query.** Crystal compiles each SSB query to its own executable. The harness rewrites `#define SF` and *recompiles* for the chosen scale factor first.
- **End-to-end timing is done by the program itself.** Inside `q11.cu`, `runQuery(...)` is wrapped in a C++ `std::chrono::high_resolution_clock` timer and prints `Execution time: X ms`. With `--t=5` it runs 5 times; the **first** timed run includes loading columns onto the GPU, **runs 2–5 are warm** (only `runQuery`). The harness reads these lines from stdout (Figure 5).
- **Microarchitecture profiling is direct.** Crystal is a single process with no warm-up server, so `ncu` profiles it immediately — no kernel skipping needed. It launches few kernels, so capturing all of them is cheap.

### 6.2 HeavyDB — `heavydb/run_query.py` (client–server ⇒ the kernel-skip trick)
- **Warm vs. cold encoded as two SQL files.** `ssb_q21_uncached.sql` *drops* the tables, recreates them, `COPY`s the Parquet in, then runs the `SELECT` — the **cold** path (load + first compile). `ssb_q21_cached.sql` is just the `SELECT` — the **warm** path.
- **End-to-end timing** uses HeavyDB's own client: the script sends `\timing`, runs the query 5×, parses the timing line.
- **Isolating the warm query under `ncu` — step by step (the important trick):**
  1. Run the warm-up (uncached) query *under `ncu`*; `ncu` prints how many kernels ran — call it `N`.
  2. Relaunch the server under `ncu` with **`-s N`** (skip the first N kernels) and run the real query. Now only the steady-state query kernels are profiled — the data-loading and table-creation kernels are skipped.

  ```
  warm-up run under ncu → counts N start-up kernels
            │
            ▼
  real run under ncu  -s N → profiles only kernels N+1, N+2, … (the warm query)
  ```
- **Why HeavyDB's counters look distinctive.** It uses a *warp execution model* (a fixed number of warps, each handling a variable tuple count) and JIT-compiles via LLVM. The counters capture the consequences: 64 registers/thread (vs Crystal's 40) and — surprisingly — *better* L1/L2 hit rates than Crystal, which becomes the optimization target (§8).

### 6.3 BlazingSQL — `blazingsql/run_query.py`
- **Runs inside a conda Python env** (`execute_query.py`).
- **Same skip trick** (run warm-up, count `N`, relaunch with `-s N`).
- **Extra switch `--target-processes all`** — BlazingSQL spawns worker subprocesses; without following children, `ncu` would profile the parent and miss the actual GPU work.
- **Why ~172 kernels per query:** it composes queries from Thrust/cuDF library calls, each its own kernel, materialising intermediate results between them. The top-kernel and op/byte counters make this visible.

### 6.4 TQP — *not in the repo*
TQP's code is **not released** (closed-source at the time) and the **analytical model is withheld** (Microsoft patent), per the README. In the paper TQP is measured with the *same* counters; it runs relational operators as PyTorch tensor programs, hence ~190 kernels and heavy intermediate-tensor traffic. From this repo you can reproduce Crystal/HeavyDB/BlazingSQL; TQP's numbers must come from the paper.

---

## 7. How each measurement becomes an insight (the reasoning chain)

Think of this as "given counter X, what may I conclude, and what is the alternative explanation I must rule out."

1. **DRAM roofline shows most dots well below the sloped ceiling.** ⇒ DRAM bandwidth is under-used. *Why?* Hash joins scatter memory accesses (random, uncoalesced), so the achieved bandwidth is poor. *Ruled-in exception:* Crystal's Q11–Q13 sit *on* the ceiling because Crystal rewrites those hash joins as a plain filter (a sequential table scan), which coalesces and saturates DRAM. The byte/op counters confirm the scan reads contiguous data.
2. **L2 roofline shows the optimised systems hugging the L2 ceiling.** ⇒ they're L2-bandwidth-bound, not DRAM-bound. *Why believable?* The SSB dimension tables fit in 40 MB L2 (see §5.5: 16.3 GB served by L2 vs 1.65 GB by DRAM). Without the second roofline you'd misdiagnose these as memory-light.
3. **`inst.txt` / `bytes.txt` (Figure 8): BlazingSQL/TQP do far more ops and read far more bytes.** ⇒ overhead from no kernel fusion + index-based materialisation + heavier hashing. *Cross-check:* the top-kernel counter shows their time spread over ~170–190 kernels.
4. **Scheduler counters (Table 5): Active≈11 but Eligible≈0.16.** ⇒ issue-bound, memory-latency-starved — *not* compute-bound. *Cross-check that rules out "low occupancy":* occupancy (Active) is fine; the problem is readiness (Eligible).
5. **Stall pie (Figure 9): dominant slice = Mem LD everywhere except BlazingSQL.** ⇒ confirms (4) with an independent counter. BlazingSQL's larger Compute/queue slices confirm it is the compute-bound outlier seen on the roofline.
6. **Memory table (Table 6): Crystal L1 hit ≈ 0.47%.** ⇒ Crystal is throwing away L1 reuse. *This single number is what the optimization targets.*

Notice the discipline: each conclusion is backed by one counter and *checked* against another that would have implied a different cause. That cross-checking is the method.

---

## 8. Optimization (Section 6) — what changed, why, and how it was measured

The optimized code lives in the `crystal/crystal-opt_src` submodule, which is **empty in your zip** (it's a separate repo). But the mechanism is fully visible because the baseline `crystal_src` already contains the two new primitives *and* the compiler flag that gets flipped.

### 8.1 Enable the L1 cache — a one-flag change with real cache-policy meaning
The baseline `crystal/crystal_src/Makefile` compiles with:
```
NVCCFLAGS += ... -Xptxas="-dlcm=cg -v" ...
```
`-dlcm` = **d**evice **l**oad **c**ache **m**ode. `cg` = "**c**ache **g**lobal" = *cache the load in L2 only, bypass L1*. Crystal historically chose this because, on older GPUs, L1 was tiny and random-access workloads thrashed it. The optimization changes it to **`-dlcm=ca`** ("**c**ache **a**ll" = use L1 too). *Why this helps here:* Crystal processes a column in **tiles** of a few tuples; the same tile is touched repeatedly within a kernel, so caching it in the (now larger) L1 gives temporal reuse instead of re-fetching through L2. This is the literal "compilation flag" the paper refers to.

### 8.2 Predicated loading (`crystal/load.cuh`, `BlockPredLoad`)
The baseline tile-load fetched *every* tuple in a tile from DRAM. The predicated version guards each load with the tuple's validity bit:
```cpp
for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (selection_flags[ITEM]) {                       // load only if still valid
        items[ITEM] = thread_itr[ITEM * BLOCK_THREADS];
    }
}
```
A tuple already rejected by an earlier `WHERE` predicate is **never fetched** ⇒ fewer DRAM bytes. *EEE intuition:* it's the difference between clocking data through a pipeline you'll then discard, versus gating the fetch with an enable signal.

### 8.3 Voluntary thread termination (`crystal/term.cuh`, `Term`)
```cpp
int count = 0;
for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) count += selection_flags[ITEM];
if (count == 0) { return; }                            // all tuples dead → stop this thread early
```
If none of a thread's tuples survive the predicates, the thread exits instead of running the rest of the query on dead data ⇒ fewer integer ops, especially for highly selective queries. When a *whole warp* dies, the warp frees its scheduler slot.

### 8.4 How the gain is *measured*: the same `ncu` pipeline, before vs. after
You compile both `crystal_src` and `crystal-opt_src`, profile each with `run_query.py --ncu`, and compare the `res/*.txt` outputs. Table 7 (Q21) is exactly that comparison:

| Metric (same counters as §4) | Crystal | Crystal-Opt | What moved |
|---|---|---|---|
| Total DRAM bytes (`dram__bytes_read.sum`) | 1653 M | 902 M | predicated load fetched fewer tuples |
| Total integer ops (§5.1) | 2704 M | 2291 M | thread termination killed dead work |
| L1 hit % (`l1tex__t_sector_hit_rate.pct`) | 0.47 | 41.31 | L1 enabled (`dlcm=ca`) |
| L2 hit % (`lts__t_sector_hit_rate.pct`) | 51 | 67 | better locality |
| L2 requests (`lts__t_requests_srcunit_tex_op_read.sum`) | 127 M | 66 M | more served by L1 ⇒ fewer fall through to L2 |

**Recompute the AI to see the roofline move (using §5.3/5.5 on the new numbers):**
```
Crystal      : AI_dram = 2704/1653 = 1.64 ;  AI_l2 = 2704e6/(127e6·128) = 0.166
Crystal-Opt  : AI_dram = 2291/902  = 2.54 ;  AI_l2 = 2291e6/(66e6·128)  = 0.271
```
Both intensities **rose** (you now do more ops per byte fetched, because you fetch fewer useless bytes), so the dots shift right and up toward the ceilings — matching the paper's Figure 11 and producing the ~1.9× average speedup (Figure 10, from the program's own chrono timer). The lesson worth keeping: **the optimization is verified with the very counters that diagnosed the problem** — a closed diagnostic loop.

---

## 9. Concurrency (Section 7) — slicing the GPU and counting queries/second

The question changes from "how efficient is one query" to "how many queries can the GPU run at once." The instrument is now `nvidia-smi` and the MPS daemon — *not* `ncu`.

### 9.1 MIG — physical partitioning (`concurrency_script/*_part_mig.py`)
Setup/teardown, verbatim from `heavydb_part_mig.py`:
```
sudo nvidia-smi -mig 1            # enable MIG mode
sudo nvidia-smi mig -cgi 9,9 -C   # create two GPU instances of profile 9, + compute instances
sudo nvidia-smi -lgc 1410         # lock clock (same control used everywhere)
...
sudo nvidia-smi mig -dci ... ; mig -dgi ; -mig 0   # tear down
```
**Reading the MIG arithmetic.** The A100 has **7 compute slices** and **8 memory slices**. A MIG *profile* names a slice size: **profile `9` = `3g.20gb`** (3 of 7 compute slices, 20 GB). `-cgi 9,9` makes **two** such instances ⇒ **Degree of Concurrency = 2** (using 6 of 7 compute slices; one slice is unavoidably idle). `-C` also creates the compute instances inside them. *Why DoC = 2 tends to win:* the 7/8 split doesn't divide evenly into 3, so DoC = 3 strands compute/memory slices and can perform *worse* than 2 — exactly what the paper observes.

**Running it.** `nvidia-smi -L` lists the MIG slice UUIDs. The script launches **one database process per slice**, each pinned with `CUDA_VISIBLE_DEVICES=<MIG-UUID>`, all started as parallel Python threads. Each runs `bin/ssb/all --t=<iter>` (the `all` binary runs all 13 queries `iter` times). `time.perf_counter()` brackets the batch to get wall-clock time.

**The metric is throughput, not a hardware counter:**
```
QPS = (slices × queries_per_slice) / wall_clock_seconds
    = (DoC × iter × 13) / total_time
```
→ paper Figure 14 (top). There are no `ncu` counters here; slicing and timing *is* the measurement.

### 9.2 MPS — logical sharing (`concurrency_script/crystal_part_mps.py`)
```
nvidia-cuda-mps-control -d              # start the MPS daemon
... launch --num-worker N processes, all sharing one GPU ...
echo quit | nvidia-cuda-mps-control     # stop the daemon
```
Unlike MIG, MPS does **not** isolate: all N workers share L2 and DRAM, but they can overlap each other's compute and data transfer over the *full* PCIe link. The script launches `N` Crystal workers as threads, times the batch, reports QPS the same way.

### 9.3 MIG vs MPS — the trade-off, with the bandwidth reasoning
- **Small data (e.g. SF = 2):** data movement (PCIe transfer) dominates. **MPS wins** — all workers share the *full* PCIe bandwidth and overlap transfer with compute, because no single worker saturates the link. Under MIG, each slice gets only a *fraction* of the PCIe bandwidth, so transfer-bound workers are throttled.
- **Large data (e.g. SF = 16):** GPU compute dominates. **MIG matches or wins** — physical isolation prevents the workers from fighting over shared L2/DRAM, and there's little transfer overlap left to exploit.
This is paper Figure 15. The general rule: **MIG for compute-dominated, isolation-sensitive workloads; MPS for transfer-dominated workloads that benefit from overlap.**

### 9.4 The estimation model is *not* in the repo
The analytical model that *predicts* QPS for a given allocation (so you needn't try every configuration) is **withheld under a Microsoft patent**. You can reproduce the *measured* concurrency numbers here; the *predicted* curves and the accuracy study come from the paper only.

---

## 10. Honest caveats — what the repo does *not* contain

- **`nsys` is effectively unused.** `crystal/crystal_src/crystal/profiler.cuh` defines `nsys start/stop` macros, but the calls are **commented out**. The reproducible pipeline is **100% `ncu`**, and end-to-end time comes from each program's own timer (Crystal's `chrono`, HeavyDB's `\timing`, BlazingSQL's loop). The paper's `nsys`-style cold-execution breakdown is illustrative.
- **TQP is absent** (closed-source); its paper numbers used the same counters but its code isn't here.
- **The analytical estimation model is absent** (Microsoft patent).
- **`crystal/crystal-opt_src` is an empty submodule in your zip** — but the optimization mechanism is fully visible in `crystal_src` (the `BlockPredLoad`/`Term` primitives and the `-dlcm=cg` flag it flips to `-dlcm=ca`).
- **Default build target** in the released Makefile is `sm_52` (old GPU) with newer targets commented; for the A100 you'd build `sm_80`. Methodology is unaffected.

---

## 11. Reproduction quickstart

```bash
# 1. Generate SSB data at SF=16 (binary for Crystal, Parquet for the others)
python3 -m venv data_venv && source data_venv/bin/activate
pip install pandas pyarrow fastparquet
mkdir -p data/storage
./data/generate_ssbm.py --sf 16
deactivate

# 2. Build a Crystal query (run_query.py rewrites #define SF and recompiles automatically)
cd crystal/crystal_src && make && make bin/ssb/q11 && cd ../../

# 3. End-to-end warm time (program's own chrono timer → stdout)
export PYTHONPATH="."
./crystal/run_query.py --bin="./crystal/crystal_src/bin/ssb/q11 --t=5" --profile-run 4 --sf 16

# 4. Microarchitecture profiling (this fires the `ncu` command from §3.1)
./crystal/run_query.py --bin="./crystal/crystal_src/bin/ssb/q11" --profile-run 1 --sf 16 --ncu
# ...or every query for all three open systems:
./characterization_script/run_crystal.sh
./characterization_script/run_heavydb.sh
./characterization_script/run_blazingsql.sh

# 5. Turn the .ncu-rep reports into the figures' numbers
mkdir res
./stats/ncu_export.py     --path tmp/res/sf16/q11/   # human-readable per-kernel dump
./stats/flush_ncu_csv.py  --path tmp/res --sf 16     # → res/roofline_*, ai.txt, inst.txt, bytes.txt, stall_*, top_kernel_*

# 6. Concurrency (needs MIG-capable GPU + sudo)
./concurrency_script/crystal_part_mig.py --sf 16 --iter 1000
./concurrency_script/crystal_part_mps.py --sf 16 --iter 1000 --num-worker 2
```

---

## 12. One-paragraph summary

Every system is measured by one tool, **Nsight Compute (`ncu`)**, invoked with a fixed list of **110 hardware performance counters** (`utility/counter_config.py`) and the switches `--metrics … --apply-rules yes --clock-control none -f -o gpudb-perf`, while the GPU clock is externally pinned to 1410 MHz with `nvidia-smi -lgc`. Because the counters record physical events, the comparison is fair across Crystal, HeavyDB, BlazingSQL, and TQP regardless of how each produces kernels. Since one query is many kernels, `stats/flush_ncu_csv.py` **sums the extensive quantities — time, bytes, integer-ops, L2-requests — across kernels and only then forms ratios**, giving arithmetic intensity (ops/byte) and throughput (Gops/s) for two rooflines: DRAM (`ops / dram_bytes`) and L2 (`ops / (L2_requests × 128)`), the latter revealing L2-bandwidth limits the DRAM view hides (worked: Crystal Q21 has AI_dram = 1.64 but AI_l2 = 0.166 because 16.3 GB are served by L2 vs 1.65 GB by DRAM). Microarchitectural counters then explain the under-use — scheduler readiness (Eligible ≈ 0.16 ⇒ issue-bound) and stall reasons (dominated by `long_scoreboard` = memory loads). The fix — enable L1 (`-dlcm=cg`→`ca`), predicated loads, early thread termination — is validated by the **same** counters showing fewer bytes/ops and higher hit rates (AI rises 1.64→2.54), yielding 1.9×. Concurrency is measured not with counters but by physically slicing the GPU with **MIG** (`nvidia-smi mig -cgi 9,9`, profile `9` = 3g.20gb ⇒ DoC 2) or sharing it logically with **MPS**, then timing 1000-query batches for queries/second. The only pieces missing from the repo are TQP, the patented estimation model, and (in your zip) the optimized-Crystal submodule — whose mechanism is nonetheless fully visible in the baseline source.
