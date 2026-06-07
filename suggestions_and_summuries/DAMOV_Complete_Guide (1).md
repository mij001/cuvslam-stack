# DAMOV, From Zero: A Complete Walkthrough of the Paper *and* the Code

*A self-contained lecture for an electrical-engineering undergraduate. No prior knowledge of computer architecture, simulators, or near-data processing is assumed. Read it top to bottom; each idea is built on the previous one.*

---

## How to use this guide

I will teach you DAMOV the way I'd teach it at a whiteboard. We start with **why anyone cares** (the physics and economics of moving data), build up **all the vocabulary** you need (DRAM, caches, locality, prefetchers, 3D memory), then walk through the **paper's idea** (a method to diagnose *why* a program is slow at the memory), and finally open up the **actual source code** in the repository you gave me and show how every concept in the paper is turned into running C++ and Python.

Whenever I write something like *(paper §3.3.1)* I'm pointing at a section of the DAMOV paper. Whenever I show a file path like `simulator/templates/template_pim_ooo.cfg`, that file is in the repository you uploaded, and I have read it — the explanations are based on the real contents, not a guess.

A reading map:

- **Parts 1–2**: the world before DAMOV. The problem and all background concepts.
- **Parts 3–5**: the paper's contribution — the metrics, and the three-step method.
- **Part 6**: the simulator (DAMOV-SIM) in detail, with code.
- **Part 7**: the six "bottleneck classes" — the central result.
- **Parts 8–9**: robustness checks and the four case studies.
- **Part 10**: a hands-on "how would I actually run this" walkthrough.
- **Part 11**: a one-page cheat sheet to keep.

---

# PART 1 — The big picture: why "data movement" is the enemy

## 1.1 The shape of a computer

Almost every computer you've used is built on the same plan, sometimes called the *von Neumann* architecture. There are two big pieces:

1. The **processor** (CPU): a small, fast chip that does arithmetic and logic — add, multiply, compare, branch.
2. The **main memory** (DRAM): a large, comparatively slow chip (or set of chips) that stores the program's data and instructions.

The CPU cannot compute on data that lives in memory directly. To add two numbers, it must first **bring those numbers from memory into the CPU**, do the addition inside the CPU, and (if needed) **send the result back to memory**. That back-and-forth shuttling of bytes between memory and the CPU is what we call **data movement**.

Here's the uncomfortable truth that motivates the entire DAMOV paper: for a huge fraction of modern programs, the time and energy are *not* dominated by the arithmetic. They are dominated by the *waiting and shuttling* — the data movement. The CPU sits idle, twiddling its thumbs, waiting for memory to deliver the next chunk of data.

## 1.2 The "memory wall"

Why does the CPU wait? Two reasons, both worth feeling in your bones:

**(a) Latency.** When the CPU asks DRAM for a piece of data, the answer takes a long time to arrive — on the order of *hundreds of CPU clock cycles*. In those hundreds of cycles, the CPU could have done hundreds of additions. If the program constantly needs new data from memory, the CPU spends most of its life stalled.

**(b) Bandwidth.** "Bandwidth" is *how many bytes per second* you can stream between memory and CPU. DRAM chips connect to the CPU through a fixed, limited set of physical wires/pins on the package. There are only so many pins, and they can only switch so fast, so there's a hard ceiling on bytes/second. The paper points to exactly this: *the external memory bandwidth is bounded by the limited number of I/O pins available in the DRAM device.* If a program wants more bytes/second than the pins can deliver, it is **bandwidth-bound** — it cannot go faster no matter how fast the CPU's arithmetic is.

Historically, CPU speed grew much faster than DRAM speed and bandwidth. The gap between "how fast the CPU can consume data" and "how fast memory can supply it" kept widening. Architects named this widening gap the **memory wall**. Modern data-hungry programs (graph analytics, machine learning, databases, genomics) slam straight into it.

## 1.3 Data movement also burns energy

It's not just slow — it's expensive in *joules*. Reading a byte from DRAM and dragging it across the chip package to the CPU costs far more energy than the arithmetic operation you'll perform on it. The DAMOV simulator literally encodes these costs; from the paper's Table 1 and the code's energy model, rough per-access energies are:

- L1 cache hit/miss: **15 / 33 pJ**
- L2 cache hit/miss: **46 / 93 pJ**
- L3 cache hit/miss: **945 / 1904 pJ**
- DRAM: **~2 pJ/bit** inside the array, plus **~8 pJ/bit** in the logic layer, plus **~2 pJ/bit** on the off-chip link.

Notice the staircase: touching L1 is cheap; touching the big shared L3 is ~60× more expensive than L1; and crossing the off-chip link to DRAM is more expensive still. **Every time data has to travel farther from the CPU, it costs more time and more energy.** So reducing data movement helps performance *and* battery life *and* electricity bills in data centers.

## 1.4 Two philosophies for fighting data movement

Architects have two broad strategies. Keep these two camps in your head — the whole DAMOV paper is a referee between them.

**Camp A — Compute-centric ("bring the data to the compute, but smarter").**
Keep the classic design (CPU here, DRAM there), but add clever hardware to *hide* or *avoid* the memory traffic:
- **Caches**: small, fast memories near the CPU that keep recently/likely-used data so you don't have to go to DRAM every time.
- **Prefetchers**: hardware that *predicts* what data you'll need next and fetches it early, so it's already nearby when you ask.

**Camp B — Memory-centric ("bring the compute to the data").**
Instead of dragging data to the CPU, put *some computing ability right next to (or inside) the memory*. This is called **Near-Data Processing (NDP)**, or **Processing-in-Memory (PIM)**. The DAMOV paper uses "NDP" and "PIM" interchangeably. Near the memory you have access to far more bandwidth and much lower latency, because you've eliminated the trip across the package pins.

Camp A is what every laptop and phone already does. Camp B is the emerging, exciting idea. **The central question of DAMOV is: for any given program, which camp will actually help, and why?** The paper's answer is a *diagnostic method* that tells you, for each function in a program, the precise reason it's memory-bound and therefore which fix will work.

---

# PART 2 — Background you actually need (built from scratch)

Before we can read the paper, we need a working mental model of six things: DRAM, caches, locality, prefetchers, 3D-stacked memory, and NDP. I'll keep each tight but complete.

## 2.1 How DRAM works (just enough)

DRAM (Dynamic Random Access Memory) stores each bit as a tiny charge on a capacitor. Bits are organized into a 2-D grid of **rows** and **columns**, grouped into **banks** (independent grids that can work in parallel).

To read data, the DRAM must:
1. **Activate** a row — copy that entire row out of the capacitor array into a fast buffer called the **row buffer**. (This is the slow step.)
2. **Read/write columns** from the row buffer (fast, once the row is "open").
3. **Precharge** — close the row before opening a different one.

Two consequences matter for DAMOV:
- If your next access hits the *same already-open row* (a **row hit**), it's fast. If it needs a *different row* (a **row miss / row conflict**), you pay the activate+precharge cost again. So *access patterns* strongly affect DRAM speed.
- Because the capacitor charge leaks, DRAM must be periodically **refreshed** (rewritten), which steals some time and energy. (DAMOV considers refresh-related effects but doesn't make them a primary classifier.)

The DRAM connects to the CPU through a **memory controller** (the traffic cop that orders requests, opens/closes rows, and obeys timing rules) and then across the **off-chip link/channel** (the package pins). That off-chip link is the bandwidth bottleneck from §1.2.

## 2.2 The cache hierarchy (Camp A's main weapon)

Going to DRAM is slow, so CPUs keep a **cache hierarchy** — a series of progressively larger, slower memories between the core and DRAM:

```
CPU core ── L1 (tiny, ~32 KB, ~4 cycles) ── L2 (~256 KB, ~7 cycles) ── L3 (big, ~8 MB, ~27 cycles, shared) ── DRAM (~hundreds of cycles)
```

(Those exact numbers are the DAMOV "Host CPU" configuration — we'll see them again.)

Vocabulary you must own:

- **Cache line / block**: caches don't store single bytes; they store fixed-size chunks, here **64 bytes** at a time. When you ask for one byte, the cache grabs the whole 64-byte line containing it. (This is itself a bet on *spatial locality* — see §2.3.)
- **Hit / miss**: if the data you want is in a given cache level, that's a **hit** (fast). If not, a **miss** — you go to the next level down, eventually to DRAM.
- **L1 / L2 are private** (each core has its own); **L3 is shared** across all cores. The numbers `32 KB`, `256 KB`, `8 MB` are typical sizes.
- **Associativity** (e.g., "8-way"): a flexibility knob for where a line may be placed. Higher associativity → fewer accidental evictions. You can treat it as "how cleverly the cache avoids throwing away useful data."
- **Replacement policy** (e.g., **LRU**, Least Recently Used): when a cache is full and you need room, LRU evicts the line you haven't touched for the longest time.
- **Inclusive** hierarchy: anything in L1/L2 is also kept in L3. (DAMOV's L3 is inclusive.)

**Two caches, two kinds of misses you must distinguish — this is the seed of DAMOV's key new metric:**
- An **L1 miss** means "not in L1." But it might still be found in L2 or L3 (cheap-ish). 
- An **L3 miss (= Last-Level-Cache miss, LLC miss)** means "not in *any* cache" → you *must* go to DRAM (expensive). 

So *the number of L3/LLC misses* is the real measure of how much you actually hammer DRAM. Hold that thought.

**Cache coherence (MESI).** With multiple cores each holding private copies of data, the hardware must ensure they all agree on the current value. The protocol that does this (DAMOV uses **MESI**) generates extra "coherence traffic" between caches. You don't need the protocol details; just know coherence traffic is *another* possible source of data movement, and the paper tested it as a candidate metric but decided it wasn't a clean classifier.

## 2.3 Locality: the reason caches work at all

Caches are a bet that programs reuse data in predictable ways. Two flavors:

- **Temporal locality**: if you used a memory address recently, you're likely to use *that same address* again soon. (Example: a loop counter, or a running sum.) Caches exploit this by *keeping* recently used data around.
- **Spatial locality**: if you used an address, you're likely to use *nearby addresses* soon. (Example: walking through an array element by element.) Caches exploit this by fetching a whole 64-byte line at once, and prefetchers exploit it by grabbing the *next* lines.

A program with **high** temporal and spatial locality is a cache's dream: almost everything hits in L1/L2, DRAM is rarely touched. A program with **low** locality (random jumps through a huge dataset — think graph traversal) is a cache's nightmare: caches barely help, and you constantly hit DRAM.

DAMOV measures these two as numbers between 0 and 1 (we'll see the exact formulas and the code in Part 4). For now: **low locality ⇒ caches don't help ⇒ a program that might love NDP**; **high locality ⇒ caches do help ⇒ a program that loves the classic CPU**.

## 2.4 Prefetchers (Camp A's second weapon)

A **prefetcher** is hardware that watches your memory access pattern and *guesses* what you'll need next, fetching it into cache *before* you ask. DAMOV uses a **stream prefetcher**: it detects sequential or regular-stride streams (addresses going 100, 164, 228, … i.e., +64 each time) and runs ahead, pulling in the next lines.

When does a prefetcher help? When the access pattern is **regular** (high spatial locality) and there are **enough** requests to learn the pattern. When does it fail or even hurt?
- If accesses are **irregular/random**, the prefetcher's guesses are wrong; it wastes bandwidth fetching garbage (it can *slow you down*).
- If memory requests are **infrequent** (low memory intensity), the prefetcher never sees enough of a pattern to train on.

You'll see both failure modes show up as findings in specific DAMOV classes.

## 2.5 3D-stacked memory and the Hybrid Memory Cube (HMC)

Now the enabling technology for NDP. A normal DRAM chip is a flat 2-D die connected to the CPU by package pins (few wires → limited bandwidth). A **3D-stacked memory** stacks several DRAM dies on top of each other and connects them *vertically* with thousands of tiny wires called **Through-Silicon Vias (TSVs)**. Because there are *thousands* of vertical wires (versus a few package pins), the internal bandwidth is enormous.

The **Hybrid Memory Cube (HMC)** — the specific device DAMOV models — adds a crucial twist: at the *bottom* of the stack there is a **logic layer**, a die that can hold actual computing circuits. The memory is divided into **vaults** (vertical slices, like 32 of them), each vault being a column of banks with its own controller in the logic layer.

The punchline:
- The **CPU outside** the cube sees the cube through the narrow off-chip link → limited bandwidth.
- A **processor placed *in the logic layer*** sees all the vaults through the fat internal TSVs → massive bandwidth, and it's physically right next to the data → low latency.

That logic-layer processor is exactly where you'd put an **NDP core**.

## 2.6 NDP/PIM: bringing compute to the data

So **Near-Data Processing** = put compute units (simple CPU cores, SIMD engines, or custom accelerators) in the logic layer of the 3D-stacked memory, and *offload* the memory-heavy parts of a program to them. The benefits:
- Much higher bandwidth (internal TSVs vs. external pins).
- Lower latency (no trip across the package; no need to climb the L2/L3 hierarchy).
- Lower energy (you skip the expensive L2/L3 lookups and the off-chip link).

The paper opens with a concrete measurement to make this vivid. Running the **STREAM Copy** micro-benchmark (which just copies a big array — pure bandwidth stress):
- the host CPU sustains a peak of **115 GB/s**;
- an NDP engine in the logic layer of *one* HMC sustains **431 GB/s** — that's **3.7×** more bandwidth.

That 3.7× is the prize. But — and this is the whole point of DAMOV — **NDP is not free and not always better.** NDP cores are simple and small (tight area/power budget in the logic layer), and crucially they **lack the deep L2/L3 cache hierarchy**. So for a program whose data *does* fit nicely in caches, moving it to NDP *removes* the caches it was relying on and makes it *slower*. Whether NDP helps depends entirely on *why* the program was memory-bound in the first place — which is the question DAMOV answers.

## 2.7 Why we can't just measure this on real hardware

You might ask: why all the modeling — just run the programs on a real NDP chip and an Intel CPU and compare? The paper explains why that's impossible to do *cleanly* (paper §2.1):
1. There are almost no real general-purpose NDP machines you can buy and configure, and the few that exist are specialized and non-tweakable.
2. Even comparing real CPUs is messy: a 1-core chip and a 256-core chip differ in dozens of ways besides core count, so you can't isolate the effect you care about.

To run a *controlled* experiment — change *only* the cache hierarchy, or *only* the core count, and hold everything else fixed — you need a **simulator**. That's why DAMOV builds **DAMOV-SIM** (Part 6).

---

# PART 3 — The problem the paper attacks

## 3.1 The two popular "shortcuts" for predicting NDP suitability

Before DAMOV, researchers used two simple metrics as rules of thumb for "should this program go to NDP?":

**(1) Arithmetic Intensity (AI).** Roughly, *how many arithmetic operations you do per byte of data you fetch from memory.* (DAMOV's precise version: operations per L1 cache line accessed.)
- High AI = "compute-heavy": you do lots of math per byte, so memory isn't your problem. Caches/CPU are great for you; NDP won't help.
- Low AI = "memory-heavy": you barely compute per byte, so you're mostly waiting on memory; a candidate for NDP.

**(2) LLC MPKI (Last-Level-Cache Misses Per Kilo-Instruction).** *How many times per 1000 instructions you miss in the last cache and have to go to DRAM.* The conventional wisdom (used by many prior works): MPKI **> 10** = "memory-intensive" = good NDP candidate.

## 3.2 The roofline model (explained from zero)

The **roofline model** is a famous one-picture way to ask "is my program limited by compute or by memory?" Picture a graph:
- **x-axis**: Arithmetic Intensity (operations per byte) — log scale.
- **y-axis**: achieved performance (operations per second) — log scale.

There are two "roofs" (ceilings) your program can't exceed:
- A **slanted roof** on the left: `performance ≤ Peak_Bandwidth × Arithmetic_Intensity`. This is the **memory roof** — when you do little compute per byte (left side), your speed is capped by how fast memory can feed you. Its slope is the memory bandwidth.
- A **flat roof** on the right: `performance ≤ Peak_Compute_Throughput`. This is the **compute roof** — once you do enough math per byte, you're capped by how fast the CPU can compute, not by memory.

Where the two roofs meet is the "ridge point." If your program sits under the slanted part, it's **memory-bound**; under the flat part, **compute-bound**. Many prior works used "memory-bound on the roofline ⇒ good for NDP."

## 3.3 Why both shortcuts fail (the paper's opening jab — Figure 1)

The paper plots 44 memory-bound applications on both a roofline and an "MPKI vs. NDP speedup" chart and color-codes each app by what *actually* happened when run on NDP:
- **yellow = faster on NDP**, **blue = faster on CPU**, **red = about the same**, **green = it depends** (NDP helps only at some core counts).

The findings that demolish the shortcuts:
- On the roofline, *most* memory-bound apps benefit from NDP (yellow) — as expected. **But** some memory-bound apps are **blue** (NDP makes them *worse*) and some are **green** (NDP helps only sometimes). The roofline can't tell these apart — it lumps them all as "memory-bound."
- On MPKI-vs-speedup, yes, *all* apps with high MPKI (>10) benefit from NDP. **But** plenty of apps with **low MPKI (<10) also benefit** from NDP. So "MPKI < 10 ⇒ skip NDP" is plain wrong; you'd miss real opportunities.

Conclusion (paper's words, paraphrased): AI/roofline and MPKI each capture *one* facet of memory behavior, so each alone **cannot comprehensively identify the source of a memory bottleneck**, and therefore cannot reliably predict NDP suitability.

## 3.4 The actual goal

So the paper sets two goals:
1. **Understand** the *root causes* of data-movement bottlenecks by finding the right *set* of metrics (not just one).
2. **Package** representative programs for each root cause into an open **benchmark suite** (DAMOV) so future researchers have a standard, diverse test set.

And the method must be able to look at a *new, never-seen* function and say "this one is bandwidth-bound; NDP will help" or "this one is compute-bound; NDP will hurt it." That requires more than one number — which brings us to the metrics.

---

# PART 4 — The metrics (the vocabulary of the whole study)

DAMOV's diagnosis rests on a small set of carefully chosen numbers. I'll define each, give the intuition, the *rationale for why it's included*, and — where the repo computes it — show the actual code. There are **two architecture-independent** metrics (properties of the program itself, no hardware assumed) and **three architecture-dependent** metrics (measured on a specific simulated machine).

## 4.0 Why split into "architecture-independent" vs "architecture-dependent"?

This split is itself a design decision worth dwelling on (paper §2.3). If you only measure things on one particular machine, you can't tell whether a bottleneck is **inherent to the program** ("this code *fundamentally* has no reuse") or just an **artifact of that machine** ("the cache was too small"). So DAMOV measures:
- **inherent** properties (spatial & temporal locality) with a tool that uses *only the program's address stream*, no cache sizes assumed; and
- **machine** effects (AI, MPKI, LFMR) on the simulated systems.

Combining both lets you say *why* a bottleneck exists and *whether better hardware (caches, prefetchers) or NDP* would remove it.

## 4.1 Memory Bound % (the screening metric for Step 1)

Before classifying *how* a function is memory-bound, you first need to find functions that *are* memory-bound at all. DAMOV uses Intel's **VTune** profiler and its **top-down analysis**, which assigns each slice of the CPU's time to a category. One category, **"Memory Bound,"** is the percentage of CPU pipeline "slots" wasted because the CPU was waiting on the memory system.

DAMOV's rule: a function is a candidate if **Memory Bound > 30%** *and* it takes **≥ 3% of total clock cycles**.
- *Why 30%?* The authors found empirically that below ~30%, applying any data-movement fix (caches, prefetch, NDP) barely changed performance or energy — so it's not worth studying.
- *Why ≥3% of cycles?* If a function is a negligible slice of runtime, optimizing it can't matter to the whole program (Amdahl's law). No point profiling noise.

## 4.2 Spatial locality (architecture-independent) — definition *and* code

**Definition (paper, Eq. 1).** Slide a window of `W` consecutive memory references. Within the window, for each reference compute the *smallest stride* (distance to the nearest other accessed address). Build a histogram ("stride profile") of how often each stride length occurs. Then:

```
Spatial Locality = Σ_i  ( stride_profile(i) / i )
```

i.e., weight each stride by the *reciprocal* of its length. Small strides (you stay close) → big contribution → score near **1** (perfectly sequential). Large/random strides → tiny contribution → score near **0**.

**Now the actual implementation** — from `simulator/src/locality.cpp`. The window size is literally 32 (`past_32`), matching the paper's `W = 32`:

```cpp
// Histogram of stride access (for each new address, look back over the last 32 addresses)
if (past_32.size() >= 32){
    uint64_t stride = 1048576;                 // start "huge"; we'll shrink to the minimum
    for (auto const& item : past_32) {
        if (abs((long)item - (long)addr) < stride)
            stride = abs((long)item - (long)addr);   // smallest distance to any of last 32 addrs
    }
    if (stride != 0) {
        s_histogram[pow(2, ceil(log2(stride)))] += 1;  // bucket the min-stride by power of two
        stride_access += 1;
    }
    past_32.pop_front();
}
past_32.push_back(addr);
```

and the score:

```cpp
float locality::get_spatial_locality(){
  for (int i = 0; i < 21; i++){
    percent = s_histogram[pow(2,i)] * 1.0 / stride_access;   // fraction of accesses with this stride bucket
    spatial_locality_score += percent * 1.0 / pow(2, i);     // weighted by 1/stride  ← Eq. 1
  }
  return spatial_locality_score;
}
```

Read it slowly: for each new address it finds the **minimum distance** to any of the previous 32 addresses, buckets that distance into a power-of-two histogram bin, and at the end sums `fraction_in_bin / stride_length`. That's Equation 1, faithfully.

## 4.3 Temporal locality (architecture-independent) — definition *and* code

**Definition (paper, Eq. 2, conceptual).** Track how often each address is *reused*. Build a "reuse profile" and combine it so that **high reuse ⇒ score near 1**, **no reuse ⇒ score 0**.

**Implementation** — also `locality.cpp`. The code uses the closely related **reuse-distance** idea: for each address, how many *other* accesses happened since you last touched this address? Short reuse distance = strong temporal locality.

```cpp
// Reuse Distance: distance (in #accesses) since this address was last seen
if (last_access.find(addr) != last_access.end()) {
    uint64_t stride = addr_id - last_access[addr];     // reuse distance
    if (stride > 1048576) stride = 1048576;
    t_histogram[pow(2, ceil(log2(stride)))] += 1;      // bucket reuse distance by power of two
}
last_access[addr] = addr_id;
```

```cpp
float locality::get_temporal_locality(){
  for (int i = 0; i < 21; i++){
    percent = t_histogram[pow(2,i)] * 1.0 / mem_accesses;
    temporal_locality_score += percent * 1.0 * (21 - i) / 21;  // short reuse distance (small i) → big weight
  }
  return temporal_locality_score;
}
```

So bin `i = 0` (reuse distance 1 — you touched the same address immediately again) gets weight `21/21 = 1`, and very long reuse distances (large `i`) get near-zero weight. Sum it up → high score means "you keep coming back to the same data quickly."

> **Nitty-gritty / honest note:** the *paper's* Equation 2 is written as `Σ 2^i · reuse_profile(i) / total`, a slightly different formulation. The shipped code instead uses this reuse-distance, `(21−i)/21`-weighted histogram. Both implement the same *idea* ("more reuse ⇒ higher score") and follow the architecture-independent locality methodology the paper cites (Weinberg et al. and Shao & Brooks). When you run the repo, the number you get is the code's version. This is a real, observable difference between the paper's clean equation and the engineering artifact — exactly the kind of detail worth knowing.

**A subtle but important cleanup step.** Right before computing locality, the code throws away **stack addresses**:

```cpp
// Remove stack addresses from the stream.
// Empirically, stack addresses appear more than 2^21 times. (cf. WIICA, ISPASS'13)
int index = int(ceil(log2(it->second)));
if (index >= 21) to_remove.insert(it->first);
```

*Why?* The call stack (local variables, return addresses) is touched constantly and would swamp the histogram with fake "locality," hiding the program's true data-access behavior. So any address referenced an absurd number of times (≥ 2²¹) is treated as stack and dropped. This keeps the metric about the *data* the algorithm actually works on.

**Word granularity.** Both metrics are computed at *word* granularity (the address is shifted by `log2(size)`), deliberately *not* at cache-line granularity, so the metric stays independent of any particular cache's line size.

## 4.4 Arithmetic Intensity (architecture-dependent)

**Definition.** Number of arithmetic/logic operations performed **per L1 cache line accessed**. High AI ⇒ compute-heavy ⇒ memory is not the bottleneck. Low AI ⇒ memory-heavy.
- *Why "per cache line" and not "per byte"?* Because that matches how the hardware and the VTune profiler actually move data (in 64-byte lines), and because it's the same definition the Step-1 profiler uses, keeping the pipeline consistent.
- DAMOV calls AI *architecture-dependent* precisely because it depends on the cache line size — a hardware parameter.

In the data: low-temporal-locality functions all have **low AI** (< 2.2 ops/line, avg ~1.3). High-temporal-locality functions range from low AI (0.3) up to **44** ops/line. The threshold separating "high AI" later turns out to be ~8.5.

## 4.5 MPKI (architecture-dependent) — definition *and* code

**Definition.** LLC (L3) **Misses Per Kilo-Instruction** = number of last-level-cache misses per 1000 instructions executed. It's a proxy for **memory intensity** — the rate at which the program slams DRAM. "High" means **MPKI > 10** (the long-standing threshold from prior work).

**Implementation** — from `simulator/scripts/get_stats_per_app.py`:

```python
l3_mpki = l3_misses / float(instructions / 1000.0)
```

Dead simple: count L3 misses (from the simulator's stats), divide by (instructions/1000). High MPKI ⇒ frequent DRAM trips ⇒ high pressure on the off-chip bandwidth.

## 4.6 LFMR — the paper's *new* metric (definition, code, intuition)

This is the star of the show, the metric DAMOV invents. **LFMR = Last-to-First Miss Ratio**:

```
LFMR = (number of LLC / L3 misses) / (total number of L1 misses)
```

**Implementation** — same Python file:

```python
if (l1_misses):
    lfmr = float(l3_misses / float(l1_misses))
else:
    lfmr = 0
```

**Intuition — why this single ratio is so powerful.** Remember §2.2: an L1 miss might still be caught by L2/L3, while an L3 (LLC) miss *must* go to DRAM. So:
- **LFMR ≈ 0** → almost no L1 misses reach DRAM; the deep L2/L3 caches *catch* nearly everything. ⇒ **The cache hierarchy is doing its job. This program loves caches.**
- **LFMR ≈ 1** → nearly *every* L1 miss sails right through L2 and L3 to DRAM. ⇒ **The deep cache hierarchy is useless here.** Those big L2/L3 caches add latency and energy for nothing.
- **Medium LFMR (0.1–0.7)** → caches help with some, but not most, misses.

Thresholds the paper uses: **low < 0.1**, **high > 0.7** (with the validation threshold around 0.56).

In plain English, LFMR answers: *"Of the times the closest cache failed, how often did the *entire* cache hierarchy also fail?"* That tells you exactly whether the L2/L3 caches are pulling their weight — and therefore whether removing them (which is what NDP does!) would hurt or be free.

This is the metric the roofline and MPKI shortcuts were missing. It's the difference between "memory-bound because the caches are too small/too contended (fixable with caches or with NDP at scale)" and "memory-bound because no caching scheme could ever help (a pure NDP win)."

## 4.7 Why *these* metrics and not others?

The authors explicitly tried other data-movement signals — **raw cache miss counts, coherence traffic, DRAM row hits/misses/conflicts** — and found that while they're useful for *describing* a program, they **don't cleanly map to a specific bottleneck type** (paper §2.4). The four-metric set (temporal locality, AI, MPKI, LFMR) plus spatial locality turned out to be the *minimum* set that *cleanly separates* every bottleneck class. (Spatial locality, interestingly, ends up *not* defining any class — see §7.0 — because the L1 cache, present in both CPU and NDP, already captures most spatial locality.)

**The five metrics, at a glance:**

| Metric | Independent or dependent? | One-line meaning | "High" means |
|---|---|---|---|
| Temporal locality | Independent | Do I reuse the same data soon? | Caches *can* help |
| Spatial locality | Independent | Do I use nearby data soon? | Prefetcher *can* help |
| Arithmetic Intensity (AI) | Dependent | Ops per cache line | Compute-bound, not memory-bound |
| MPKI | Dependent | DRAM trips per 1000 instr | High memory intensity / bandwidth pressure |
| **LFMR** (new) | Dependent | Fraction of L1 misses that reach DRAM | Deep caches are *useless* here |

---

# PART 5 — The three-step methodology

Now we assemble the metrics into a procedure. Input: a program's source code + its input datasets. Output: for each important function, a label saying *which of six bottleneck classes* it belongs to (and therefore whether NDP, caches, or prefetching will help). The method has three steps (paper Figure 2).

## Step 1 — Find the memory-bound functions (the screening net)

Run the program under **Intel VTune** on a real Intel Xeon (the paper used an E3-1240, 4 cores, hyper-threading disabled for clean counts). Use top-down analysis to get each function's **Memory Bound %**. Keep functions with **Memory Bound > 30%** and **≥ 3% of clock cycles** (§4.1). 

*Why a real machine here and a simulator later?* Step 1 is a *cheap, broad first pass* over a *huge* number of programs — you want speed, and a real CPU with hardware counters is fastest. It just answers "is this function worth examining at all?" The careful, controlled comparison comes later in the simulator.

This step is marked **optional** in the framework: if you already know your hotspot functions (your "regions of interest"), you can skip straight to instrumenting them. In the code, the "region of interest" is marked with the hooks we'll meet in §6.3.

**Scale of this pass:** 345 applications, from 37 benchmark suites, **77,000 functions** analyzed. From those, **144 functions** (across 74 applications, 16 suites) survive the filter and become the **DAMOV benchmark suite**. For readability, 44 of those are used as "representative" examples in the figures, and 12 are drilled into for the performance-scaling plots.

## Step 2 — Locality-based clustering (split into two big families)

Take the surviving functions and measure their **spatial and temporal locality** with the architecture-independent tool (§4.2–4.3, the `locality.cpp` code). Plot every function on a spatial-vs-temporal scatter and run **k-means clustering** (a standard algorithm that groups points by similarity) (paper Figure 3).

Two clusters fall out naturally:
- **Low temporal locality** functions (little data reuse).
- **High temporal locality** functions (lots of reuse).

This top-level split is the first branch of the final decision tree. Intuition: functions in the bottom-left (low spatial *and* temporal) are least able to use a multi-level cache, so they're the prime NDP candidates — *but* Step 3 will show suitability depends on more than this.

## Step 3 — Scalability analysis + bottleneck classification (the careful experiment)

Here's the controlled simulation. For each function, run it on **three system configurations** and **sweep the core count** from **1 → 4 → 16 → 64 → 256**, collecting AI, MPKI, and LFMR. The three configs (paper §2.4.2, and the templates we'll dissect in Part 6):

1. **Host CPU** — full deep cache hierarchy: private L1 (32 KB) + L2 (256 KB) per core, shared L3 (8 MB, 16 banks). Models Camp A's default.
2. **Host CPU with prefetcher** — same as above, plus a **stream prefetcher** (2-degree, 16 stream buffers, 64 entries) on the L2. Models Camp A trying harder.
3. **NDP** — a single, *read-only* L1 (32 KB), **no L2, no L3, no prefetcher**, sitting in the logic layer with HMC bandwidth/latency. Models Camp B.

Everything *else* (core type, instruction window, branch predictor, number of cores) is held identical across the three, **so that any performance/energy difference comes purely from data-movement handling** — caches vs. prefetch vs. NDP. That isolation is the entire reason for using a simulator.

**Why sweep core count 1→256?** Several reasons, all about exposing memory behavior:
- More cores → more pressure on shared memory; you can *watch* a program saturate the DRAM bandwidth as you add cores (a tell-tale of bandwidth-bound code).
- Reveals how much **memory-level parallelism (MLP)** a program has (can it keep many memory requests in flight?).
- Crucially, **aggregate cache size scales with cores**: when you go from 1→4 cores, the total private L1/L2 capacity also ×4. So at high core counts the caches are collectively much bigger. Watching how LFMR/MPKI *change* as caches grow is how DAMOV distinguishes "fundamentally uncacheable" from "just needs a bigger cache." (Prior works that picked large core counts to saturate high-bandwidth memory motivated this range.)
- 256 cores is high enough to saturate modern high-bandwidth memories; 1 core is the simplest baseline.

After Step 3 you have, for each function: its temporal locality (Step 2), and its AI, MPKI, LFMR and *how LFMR trends with core count* (Step 3). Feed those into a decision tree → one of **six classes**. That tree is Part 7.

---

# PART 6 — DAMOV-SIM: the simulator, in detail (with code)

This is the part most guides skip and you specifically asked for. We'll see *exactly* how the abstract method becomes running software, using the files in your repo.

## 6.1 The two engines: ZSim + Ramulator

DAMOV-SIM glues together two well-known open-source simulators:

- **ZSim** — simulates the **CPU side**: the cores (their pipelines), the cache hierarchy, the coherence protocol, and the prefetchers. ZSim is fast because it uses **Intel Pin** (a "dynamic binary instrumentation" tool) to actually *run* the program's real instructions and only *model* the timing, instead of interpreting everything from scratch.
- **Ramulator** — simulates the **memory side**: the DRAM device (here, an HMC), the memory controllers, timing rules, and the actual memory accesses.

ZSim hands every memory request that misses the caches to Ramulator, which tells it how long that request takes. Together they form a **cycle-accurate** model of the whole machine. The repo layout (from the README):

```
simulator/
 ├── src/                 # ZSim C++ source (cores, caches, locality, the Ramulator bridge…)
 ├── ramulator/           # Ramulator C++ source (DRAM/HMC models)
 ├── ramulator-configs/   # DRAM device configs (HMC-config.cfg, DDR4, HBM, …)
 ├── templates/           # system templates (host/pim × ooo/inorder/accelerator)
 ├── command_files/       # how to launch each benchmark (binary path + args)
 ├── scripts/             # generate configs, parse stats, build, setup
 └── network_*.mesh       # NoC topologies for the NUCA / multi-core experiments
```

## 6.2 Marking *what* to simulate: the ROI/offload hooks

Simulating a giant program from start to finish is wasteful — you only care about the hot function. So you **instrument** the source: you insert special marker calls around the region you care about. These live in `simulator/misc/hooks/zsim_hooks.h`:

```cpp
#define ZSIM_MAGIC_OP_ROI_BEGIN   (1025)
#define ZSIM_MAGIC_OP_ROI_END     (1026)

static inline void zsim_magic_op(uint64_t op) {
    __asm__ __volatile__("xchg %%rcx, %%rcx;" : : "c"(op));   // the "magic instruction"
}
static inline void zsim_roi_begin() { zsim_magic_op(ZSIM_MAGIC_OP_ROI_BEGIN); }
static inline void zsim_roi_end()   { zsim_magic_op(ZSIM_MAGIC_OP_ROI_END);   }
```

**How does this work?** `xchg %rcx, %rcx` exchanges a CPU register with *itself* — on real hardware it does nothing (a glorified NOP). But ZSim's Pin instrumentation *watches for this exact instruction* and reads the magic code that was loaded into `rcx`. So `1025` means "the region of interest starts here," `1026` means "it ends here." It's a way to send a signal from the running program to the simulator *without changing the program's behavior*. The README's workflow:

```cpp
#include "zsim_hooks.h"
foo(){
    zsim_roi_begin();           // start of region of interest (must be in serial code)
    zsim_PIM_function_begin();  // start of the hotspot to "offload" / measure
    ...                         // <-- this is the code that runs on the host or NDP core
    zsim_PIM_function_end();    // end of the hotspot
    zsim_roi_end();             // end of region of interest
}
```

So the program runs normally (fast-forwarded) until it hits the ROI, then ZSim switches to detailed cycle-accurate modeling for the marked hotspot — the part that, in a real NDP machine, you'd *offload* to the in-memory cores.

**Bounding the run.** The templates set `maxOffloadInstrs = 1000000000` (one billion). In `src/zsim.cpp`, each core counts how many instructions it executed *inside* the offload region (`offload_instrs`), and the simulation stops once the total reaches the cap:

```cpp
if (zinfo->maxOffloadInstrs) {
    uint64_t totalOffloadInstrs = 0;
    for (...) totalOffloadInstrs += zinfo->cores[i]->getOffloadInstrs();
    if (totalOffloadInstrs >= zinfo->maxOffloadInstrs) { /* stop */ }
}
```

This keeps each simulation finite and comparable (you measure the same billion instructions of the hotspot everywhere).

## 6.3 The three core models (OOO, Timing/in-order, Accelerator)

The README lists three CPU core types ZSim can simulate. DAMOV uses all three to show its conclusions don't depend on the core:

- **`OOO` (out-of-order).** A modern aggressive core: it can execute instructions out of program order and keep many memory requests in flight, so it can *hide* some memory latency (do useful work while waiting). DAMOV's default.
- **`Timing` (in-order, 1-issue).** A simple core that executes strictly in order. When it hits a memory stall, it just *waits*. This is the kind of small core you'd realistically fit in an HMC logic layer (limited area/power).
- **`Accelerator` (dataflow model).** Not a general CPU — a model of a custom hardware accelerator. Per the README it "issues at every clock cycle all independent arithmetic instructions in the dataflow graph of a given basic block."

The accelerator's timing model is beautifully simple and worth seeing (`src/accelerator_core.cpp`):

```cpp
void AcceleratorCore::bblAndRecord(Address bblAddr, BblInfo* bblInfo) {
    instrs   += bblInfo->instrs;     // count instructions
    curCycle += bblInfo->depth;      // but advance time by the DEPENDENCY-GRAPH DEPTH, not the instr count
    ...
}
```

Read that again: a normal core would advance the clock by roughly the *number* of instructions; the accelerator advances it by the **depth** of the basic block's dataflow graph — i.e., the *critical path*. If 100 independent additions can all run in parallel in one cycle, the accelerator takes ~1 "step," not 100. That captures "all independent ops issue simultaneously." This model underlies the accelerator case study (Part 9).

**Why OOO *and* in-order?** Because an OOO host can hide memory latency, an OOO host vs. OOO NDP comparison *understates* NDP's benefit, while an in-order host vs. in-order NDP *overstates* it. By doing both, the paper shows the *six classes are the same either way* — the classification is about the program's data movement, not the core (paper §3.5.2). Quantitatively, NDP's average speedup with in-order cores is ~11% higher than with OOO cores, exactly because the OOO host was hiding latency.

## 6.4 The three system configurations, as template files

This is where the paper's three configs (§5, Step 3) become concrete. Configs are written in a `key = value` format (libconfig). The repo ships *templates* with placeholders like `NUMBER_CORES` that a script fills in. Let's compare the three OOO templates side by side; the differences are the whole story.

**(1) Host CPU — `templates/template_host_ooo.cfg`** (full hierarchy):

```
cores:  type="OOO", cores=NUMBER_CORES
caches:
  l1d = 32768 B,  8-way, latency 4
  l1i = 32768 B,  4-way, latency 3
  l2  = 262144 B, 8-way, latency 7,  children="l1i|l1d"
  l3  = 8388608 B (8 MB), 16 banks, 16-way, latency 27, type="Timing", children="l2"
mem:  type="Ramulator", ramulatorConfig="ramulator-configs/HMC-config.cfg"
sim:  pimMode = false
```

There it is, exactly matching paper Table 1: L1 32 KB / L2 256 KB / L3 8 MB. The `children` lines wire the hierarchy: L1i and L1d feed L2; L2 feeds L3; L3 feeds memory.

**(2) Host CPU with prefetcher — `templates/template_host_prefetch_ooo.cfg`.** Identical to (1) but inserts a prefetcher between L1 and L2:

```
l2prefetcher = { isPrefetcher = true; entries = 16; prefetchers = NUMBER_CORES; children = "l1d"; }
l2 = { ... children = "l1i|l2prefetcher"; }     // L2 now sees the prefetcher
```

`entries = 16` is the paper's "16 stream buffers." This is the *only* difference from config (1) — clean isolation of "what does prefetching alone change?"

**(3) NDP / PIM — `templates/template_pim_ooo.cfg`** (the deep hierarchy is gone):

```
cores:  type="OOO", cores=NUMBER_CORES
caches:
  l1d = 32768 B, 8-way, latency 4
  l1i = 32768 B, 4-way, latency 3
  l3  = { ... bypass = true; children = "l1i|l1d"; }   // ← NO L2; L3 is set to BYPASS (acts like it's not there)
mem:  type="Ramulator", ramulatorConfig="ramulator-configs/HMC-config.cfg"
sim:  pimMode = true                                    // ← the magic flag
```

Two changes capture "NDP has only a single cache level":
- **No L2 cache at all**, and the **L3 has `bypass = true`** — meaning requests pass straight through it to memory (it's there in name only so the wiring is valid). Net effect: only the L1 acts as a real cache, exactly as the paper says NDP has "only a private read-only L1."
- **`pimMode = true`** — flips the memory model into "I'm sitting next to the DRAM" mode (next subsection).

> So the *entire* architectural difference between "Host" and "NDP" in DAMOV is: **NDP deletes L2/L3 and turns on `pimMode`.** Everything you read about the six classes is, mechanically, a study of *what happens when you delete the deep cache hierarchy and move next to memory* — for different kinds of programs.

There are parallel templates for in-order (`*_inorder.cfg`), accelerator (`*_accelerator.cfg`), and a **NUCA** family (`template_host_nuca*.cfg`) used for the cache-size sweep — more on NUCA in §8.1.

## 6.5 What `pimMode` actually does (the latency/bandwidth/NoC mechanics)

`pimMode = true` is the switch that turns a "host" memory model into an "NDP" one. In `simulator/src/ramulator_mem_ctrl.cpp`:

```cpp
cpu_tick = int(1000000.0 / _cpuFreq);
mem_tick = wrapper->get_tCK() * 1000;
if (pim_mode) cpu_tick = mem_tick;   // CPU now ticks in the memory's clock domain
...
if (pim_mode) cpuFreq = memFreq;     // and runs at the memory's frequency
```

Plain English: in host mode the CPU and DRAM live in *different clock domains*, and crossing between them (the off-chip link) costs time. In `pimMode`, the compute and the memory share a clock domain — there *is* no off-chip crossing, because the compute is *inside* the memory. That's the source of NDP's lower latency.

Down in Ramulator's HMC controller (`ramulator/HMC_Controller.h`), the bandwidth difference shows up in the burst sizing, and the response path skips the off-chip link:

```cpp
if (!pim_mode_enabled) req.burst_count = channel->spec->burst_count;  // off-chip link bursts
else                   req.burst_count = 2;  // TSV is 32 B wide, a request is 64 B → just 2 bursts internally
...
if (pim_mode_enabled) {           // PIM: deliver the result locally, no off-chip response packet
    req.callback(req); pending.pop_front();
} else {                          // host: build a response packet that must traverse the off-chip link
    Packet packet = form_response_packet(req); ...
}
```

And — neatly — `pimMode` is also where the **inter-vault network** is modeled (this powers Case Study 1, Part 9). In `ramulator/HMC_Memory.h`:

```cpp
if (pim_mode_enabled) {
    // Model NoC traffic: 32 vaults arranged as a 6x6 mesh; cost = Manhattan distance (hops)
    int vault_destination_x = req.addr_vec[Vault]/6, vault_destination_y = req.addr_vec[Vault]%6;
    int vault_origin_x = req.coreid/6, vault_origin_y = req.coreid%6;
    int hops = abs(dst_x - org_x) + abs(dst_y - org_y);
    if (!network_overhead) hops = 0;            // can be toggled off for an "ideal" network
    if (req.type == READ) hops = hops * 6;      // a read pays per-hop latency
}
```

So when an NDP core in one vault needs data living in a *different* vault, the model charges it for the hops across the on-chip mesh. Set `network_overhead = false` and you get an idealized zero-latency network — the paper compares the two to measure the network's cost (Figure 20–21).

## 6.6 The memory device: HMC config

`ramulator-configs/HMC-config.cfg` describes the DRAM device itself:

```
standard = HMC
org = HMC_4GB
maxblock = HMC_256B
source_mode_host_links = 4
payload_flits = 16
pim_mode = 0
cpu_tick = 8
mem_tick = 3
expected_limit_insts = 200000000
warmup_insts = 100000000
```

Key bits: it's an HMC stack with a 256-byte row block, 4 host links, and the simulation does a **warmup of 100 M instructions** then measures up to **200 M** (so the caches are "warm" — already populated — before measurements count, avoiding cold-start bias). 

> **Honest nitty-gritty:** the paper's Table 1 lists the HMC as **8 GB**; this shipped config says `org = HMC_4GB`. Minor configuration differences like this between a paper's description and its released artifact are common; if you reproduce results, trust the config file you actually run. The `200000000` here also matches the `max_instructions = 200000000` cap inside `locality.cpp`, so the locality analysis and the timing simulation look at comparable amounts of the program.

## 6.7 From stats to metrics: `get_stats_per_app.py`

After a simulation finishes, ZSim writes a stats file (`*.zsim.out`) full of counters. The parser `simulator/scripts/get_stats_per_app.py` reads it and computes the metrics we defined in Part 4. The essential lines:

```python
cycles = max(cycles_list)                          # total time = the slowest core's cycle count
ipc    = float(instructions) / float(cycles)       # Instructions Per Cycle (higher = faster)

l3_mpki = l3_misses / float(instructions/1000.0)   # §4.5  (LLC MPKI)
lfmr    = float(l3_misses / float(l1_misses))      # §4.6  (the new metric)
```

Two things to internalize:
- **`cycles = max` across cores**, because the cores run in parallel — the program isn't done until the slowest core finishes. **Performance = IPC**, and "speedup of NDP over host" is just `IPC_pim / IPC_host`.
- It also computes per-level miss rates. The miss/hit counts are pulled out of the stats by matching ZSim's labels like `GETS misses`, `GETX I->M misses` (these are coherence-protocol request types — `GETS` = "get shared/read," `GETX` = "get exclusive/write").

The README shows the worked example for STREAM Add on 4 OOO cores:
- Host: IPC **2.22**, L3 MPKI **23.4**, LFMR **≈1.0** (almost every L1 miss hits DRAM — caches useless).
- NDP: IPC **3.52**, L3 MPKI **0** (no L3 to miss), LFMR **0**.
- Speedup = 3.52 / 2.22 = **1.58×**.

That LFMR ≈ 1.0 on the host is the fingerprint of a **Class 1a (DRAM bandwidth-bound)** function — STREAM is the textbook bandwidth hog — and the 1.58× NDP win confirms it. You just watched the whole method work end to end on one program.

## 6.8 The full workflow, automated

How do you go from "16 benchmark suites" to "thousands of simulations"? Three scripts:

**(a) `command_files/*`** — one file per suite listing every function to run, as `benchmark,application,function,command`. E.g. `command_files/stream_cf`:

```
stream,Add,Add,PIM_ROOT/STREAM/stream_add THREADS
stream,Copy,Copy,PIM_ROOT/STREAM/stream_copy THREADS
...
```

`PIM_ROOT` (the workloads folder) and `THREADS` (the core count) are placeholders.

**(b) `scripts/generate_config_files.py`** — reads a command file and *stamps out* concrete `.cfg` files from the templates, for **every combination** of {core count} × {core type} × {config}. The combinatorial loop is right there:

```python
number_of_cores = [1, 4, 16, 64, 256]
...
# for each application, generate configs for:
create_host_configs_no_prefetch(... "inorder");  create_host_configs_prefetch(... "inorder");  create_pim_configs(... "inorder")
create_host_configs_no_prefetch(... "ooo");      create_host_configs_prefetch(... "ooo");      create_pim_configs(... "ooo")
create_host_configs_no_prefetch(... "accelerator"); create_host_configs_prefetch(... "accelerator"); create_pim_configs(... "accelerator")
```

So one application → 5 core counts × 3 core types × 3 configs = **45 config files**, each writing its stats to a tidy path under `zsim_stats/`. This is how the "sweep core count, three configs" of Step 3 is realized in bulk.

**(c) Run + parse:**
```
./build/opt/zsim config_files/host_ooo/no_prefetch/stream/4/Add_Add.cfg
python scripts/get_stats_per_app.py zsim_stats/host_ooo/no_prefetch/4/stream_Add_Add.zsim.out
```

That's the entire machine: instrument → generate configs → run ZSim+Ramulator → parse stats → compute IPC/MPKI/LFMR → classify.

---

# PART 7 — The six bottleneck classes (the central result)

Everything so far was setup. Here's the payoff: a **decision tree** (paper Figure 26) that takes a function's metrics and outputs *why* it's memory-bound and *whether NDP helps*. I'll give the tree, then each class with its full story.

## 7.0 The decision tree

```
                         ┌── high MPKI ──────────────────────────────► 1a  DRAM Bandwidth-bound
            ┌─ high LFMR ─┤
            │             └── low MPKI ───────────────────────────────► 1b  DRAM Latency-bound
LOW temporal┤
 locality   └─ LFMR DECREASES with core count, low MPKI ──────────────► 1c  L1/L2 Cache Capacity
            
            ┌─ LFMR INCREASES with core count, low MPKI ──────────────► 2a  L3 Cache Contention
HIGH        │
 temporal ──┤             ┌── low AI ─────────────────────────────────► 2b  L1 Cache Capacity
 locality   └─ low LFMR ──┤
                          └── high AI ────────────────────────────────► 2c  Compute-bound
```

First branch: **temporal locality** (Step 2). Then **LFMR** (and *how it trends with core count*). Then **MPKI** or **AI** as the final discriminator.

**Why spatial locality isn't in the tree:** the L1 cache exists in *both* host and NDP, and it already captures most spatial locality (sequential streaming). So spatial locality doesn't separate the classes — it's measured, but it informs *sub*-decisions (like "will a prefetcher help?"), not the class label.

**Why some metric combinations never appear** (the authors note these are physically impossible, paper §3.3):
- *High MPKI with low LFMR* is impossible: low LFMR means L2/L3 catch most L1 misses, so the L3 can't be missing a lot (high MPKI).
- *High temporal locality with both high LFMR and high MPKI* is impossible: lots of reuse means the caches will catch the repeats, so you can't simultaneously have most misses reach DRAM at a high rate.
- *Low temporal locality with low LFMR* is impossible: with little reuse, there's nothing for the caches to capture, so LFMR can't be low.

That's why there are six classes, not 2×2×2×2 = 16.

---

Now each class. For every one I give: **the metric fingerprint**, **the plain-English bottleneck**, **what happens as you add cores**, **energy story**, **the NDP verdict and why**, and **example functions**.

## Class 1a — DRAM Bandwidth-bound  *(paper §3.3.1)*

- **Fingerprint:** low temporal locality, **low AI**, **high LFMR**, **high MPKI**.
- **Bottleneck:** the function reads enormous amounts of data with almost no reuse, so it floods the memory system. It needs more bytes/second than the off-chip pins can deliver. It is starved by **DRAM bandwidth**.
- **At scale:** on the host, performance rises with cores until the DRAM bandwidth saturates (e.g., HSJNPO gains 27.5× from 1→64 cores but only +27% from 64→256 — the pins are maxed). On NDP, performance *keeps climbing* because the internal TSV bandwidth is much higher. NDP outperforms host by up to ~4.8× at 256 cores.
- **Prefetcher?** Useless or harmful (low spatial locality → wrong guesses; the host-with-prefetcher is actually ~40% *slower* here).
- **Energy:** NDP wins big — it skips the L2/L3 lookups and the off-chip link entirely.
- **NDP verdict:** ✅ **The classic NDP win.** These are exactly the apps prior NDP work targeted.
- **Examples:** STREAM (Add/Copy/Scale/Triad), graph kernels (Ligra PageRank, Radii), hash-join probe, Darknet Yolo's gemm. Two sub-flavors: *regular* access (STREAM, Yolo — great for SIMD accelerators) and *irregular/pointer-chasing* access (graphs, hash joins — need MLP-extracting techniques).

## Class 1b — DRAM Latency-bound  *(paper §3.3.2)*

- **Fingerprint:** low temporal locality, low AI, **high LFMR**, **low MPKI**.
- **Bottleneck:** like 1a, the caches are useless (high LFMR), **but** the function makes memory requests *infrequently* (low MPKI), so it doesn't saturate bandwidth. Instead, each individual DRAM request sits on the **critical path** — the core stalls waiting for that one long-latency request before it can continue. It is starved by **DRAM latency**, not bandwidth. (Average DRAM bandwidth used is a mere 0.5 GB/s; LFMR is ≥ 0.94.)
- **At scale:** both host and NDP scale well, but NDP is *always* a bit faster at the same core count (~1.12–1.13× on average). Why? Because NDP cuts the **Average Memory Access Time (AMAT)** — it doesn't waste time looking up L2/L3 (which almost never hit anyway) or crossing the off-chip link.
- **Prefetcher?** Doesn't help — requests are too infrequent for the prefetcher to learn a pattern.
- **Energy:** NDP saves a lot (up to 69%, ~39% avg) by skipping the pointless L2/L3 lookups.
- **NDP verdict:** ✅ NDP helps, primarily via lower latency and energy. (Other latency tricks — cache bypassing, low-latency DRAM, better scheduling — would also help.)
- **Examples:** Chai Histogram, PolyBench LU decomposition (`PLYalu`), Phoenix String Match / Linear Regression, SPLASH-2 Ocean `slave2`.

## Class 1c — L1/L2 Cache Capacity-bound  *(paper §3.3.3)*

- **Fingerprint:** low temporal locality, low AI, low MPKI, and the tell-tale: **LFMR *decreases* as core count rises**.
- **Bottleneck:** at *low* core counts the private L1/L2 caches are too small to hold the working set (high LFMR → behaves like latency-bound 1b). But remember aggregate L1/L2 capacity grows with core count. At *high* core counts the working set finally fits → LFMR plummets (e.g., DRKRes LFMR 0.5 at 1 core → 0.09 at 256) → the caches start working → the host catches up and passes NDP.
- **At scale:** NDP wins at *low* core counts (small caches), host wins at *high* core counts (big aggregate caches). The crossover is the signature.
- **Energy:** mixed — depends on how much the L2/L3 are catching at a given core count.
- **NDP verdict:** ⚠️ **It depends on core count.** NDP is a smart choice if your area budget is tight (NDP matches the host *without* needing big L2/L3 caches), but a big-cache host wins at high core counts.
- **Examples:** Darknet Resnet's gemm (`DRKRes`), PARSEC Fluidanimate (`PRSFlu`), Chai Padding.

## Class 2a — L3 Cache Contention-bound  *(paper §3.3.4)*

- **Fingerprint:** **high** temporal locality, low AI, low MPKI, and: **LFMR *increases* as core count rises**.
- **Bottleneck:** these functions reuse data (high temporal locality), so at low core counts the shared L3 serves them beautifully (LFMR ~0.03–0.44; almost nothing reaches DRAM). But as you add cores, they all **fight over the single shared L3** — they evict each other's data. This **contention** wrecks the L3 (LFMR shoots up to ~0.97 at 256 cores), converting a cache-friendly program into a latency-bound one. At 256 cores the memory-controller queues even overflow, forcing requests to be re-issued.
- **At scale:** at low cores NDP is *worse* (it threw away the L3 these apps love — PLYGramSch is 67% slower on NDP at 1 core). At high cores NDP is *much better* (2.2×–3.85×), because the host's L3 has become useless due to contention while NDP's huge internal bandwidth absorbs the flood.
- **Energy:** host wins at low cores; NDP wins at high cores (host burns L3+link energy on all the contention-induced off-chip traffic).
- **NDP verdict:** ⚠️→✅ at scale. NDP is a *cheaper* way to relieve L3 contention than building an ever-bigger shared L3.
- **Examples:** PolyBench Gram-Schmidt (`PLYGramSch`), SPLASH-2 FFT (Reverse/Transpose), Ligra graph kernels' edgeMapSparse on USA road graph.

## Class 2b — L1 Cache Capacity-bound  *(paper §3.3.5)*

- **Fingerprint:** high temporal locality, **low AI**, **low/medium LFMR**, low MPKI.
- **Bottleneck:** the working set is bottlenecked by the small **L1**, but the L2/L3 partially catch the spillover. The interesting result: the latency the host pays for "L3 + DRAM" is about the same as the latency NDP pays for "DRAM," so the two come out **essentially equal** (within 1%) at every core count.
- **At scale:** host and NDP track each other almost exactly. NDP neither helps nor hurts performance (sometimes a small energy saving, e.g., 12% for PLYgemver).
- **NDP verdict:** ➖ **Performance-neutral.** But there's a *systems* angle: since NDP matches the host *without* L2/L3, you can use NDP to **delete the L2/L3 SRAM** (saving chip area and static power) with no performance penalty.
- **Examples:** PolyBench gemver (`PLYgemver`) and 2D convolution, SPLASH-2 LU (`SPLLucb`) and Radix, Chai Bezier Surface.

## Class 2c — Compute-bound  *(paper §3.3.6)*

- **Fingerprint:** high temporal locality, **low LFMR**, low MPKI, **high AI**.
- **Bottleneck:** there *isn't* a serious memory bottleneck. These functions do lots of math per byte (AI up to 44), reuse their data, and the caches serve them perfectly. They are limited by *compute*, and they make excellent use of the deep cache hierarchy and the prefetcher (high spatial locality → very accurate prefetch).
- **At scale:** the host is **always faster** (44–54% on average), because NDP *removed* the L2/L3 hierarchy these apps depend on.
- **Energy:** host usually wins (it benefits from caches); NDP can occasionally save energy only for functions with medium LFMR at high core counts.
- **NDP verdict:** ❌ **Do not send to NDP** — it makes them slower and usually more energy-hungry.
- **Examples:** HPCG kernels (SpMV, SymGS, Prolongation, Restriction), most PolyBench dense linear algebra (3mm, gemm, doitgen, symm), Rodinia Needleman-Wunsch (`RODNw`) and BFS, PARSEC Ferret.

## 7.1 The summary numbers

Averaged across all 144 functions and all core counts, NDP speedup over host (OOO / in-order cores) by class (paper §3.5.2):

| Class | Bottleneck | NDP speedup (OOO / in-order) | Verdict |
|---|---|---|---|
| 1a | DRAM bandwidth | **1.59 / 1.77** (up to 4.8×) | ✅ NDP win |
| 1b | DRAM latency | **1.22 / 1.15** (up to 3.4×) | ✅ NDP win |
| 1c | L1/L2 capacity | 0.96 / 0.95 (up to 2.3× at low cores) | ⚠️ depends on cores |
| 2a | L3 contention | 1.04 / 1.22 (up to 3.8× at high cores) | ⚠️ NDP win at scale |
| 2b | L1 capacity | 0.94 / 1.01 (≈ neutral) | ➖ neutral; saves SRAM |
| 2c | Compute-bound | **0.56 / 0.76** (NDP slowdown) | ❌ keep on CPU |

The single most important takeaway: **"memory-bound" is not one thing.** Six different root causes hide under that label, and NDP is a brilliant fix for some (1a, 1b), a conditional fix for others (1c, 2a), neutral for one (2b), and harmful for one (2c). The roofline/MPKI shortcuts couldn't see this; LFMR-plus-the-others can.

---

# PART 8 — Robustness: did they check their own work? (yes)

A good methodology paper stress-tests its claims. DAMOV does four checks.

## 8.1 Does L3 cache size change the conclusions? (the NUCA sweep, §3.4)

In Step 3, only the *private* L1/L2 grew with core count; the shared L3 was a fixed 8 MB. Fair question: what if you let the L3 grow too? They re-ran everything with a **NUCA** (Non-Uniform Cache Architecture) L3 that scales at **2 MB/core** (so 512 MB at 256 cores!), with cores and L3 banks connected by a 2-D mesh **Network-on-Chip** (3 cycles/hop, router 63 pJ, link 71 pJ per traversal). This is exactly the `template_host_nuca*.cfg` family in the repo, which adds `networkType="mesh"`, a `networkFile`, and `size = LLC_SIZE`.

Result: the six classes behave **exactly as their definitions predict**. Class 1a still can't escape the bandwidth wall even with 512 MB of L3; Class 1b/1c don't benefit from extra L3; Class 2a (the contention class) *does* benefit from a bigger L3 (makes sense — more room, less fighting) but NDP still wins when the NoC itself gets congested; Class 2c loves the bigger cache. And NDP still gives large energy savings even versus a 512 MB cache — which would be absurdly expensive to build. **Conclusion: the classification is robust to L3 size.**

## 8.2 Does the method generalize? (validation on held-out functions, §3.5)

The six classes were defined using 44 functions. To prove they aren't just curve-fit to those 44, the authors:
1. Computed numeric **thresholds** for each metric from the 44 (temporal locality **0.48**, LFMR **0.56**, MPKI **11.0**, AI **8.5**), plus the LFMR-vs-core-count trend.
2. Applied those thresholds *blind* to the **other 100** memory-bound functions that were *not* used to build the classes.

Result: **97% of the 100 held-out functions classified correctly** (a function counts as correct only if it both matches the threshold fingerprint *and* shows the expected host-vs-NDP scaling). Only 3 graph functions misclassified, and for an understandable reason (their MPKI sat just under the 1a threshold). **Conclusion: the method works on never-seen functions** — which is the whole point of a diagnostic tool.

## 8.3 Does it depend on the core type? (in-order vs OOO, §3.5.2)

They reran with both in-order and out-of-order cores. The architecture-dependent metrics (LFMR, MPKI, AI) come out **the same regardless of core type**, and the six classes show the same trends. NDP's average speedup is ~11% higher with in-order cores (because an OOO host hides some latency, shrinking the apparent NDP benefit). **Conclusion: the classification is a property of the program's data movement, not the core microarchitecture.**

## 8.4 Does an independent algorithm agree? (hierarchical clustering, §4.1)

They fed the same metrics into a totally different unsupervised algorithm — **hierarchical clustering** (which builds a tree/dendrogram by repeatedly merging the most-similar functions). The dendrogram **independently reproduces the six classes**: high-temporal-locality functions on one side, low on the other, with clean sub-groups matching 1a–2c. It even re-discovers the regular-vs-irregular split *within* Class 1a. **Conclusion: the structure DAMOV imposes is really in the data, not an artifact of the authors' hand-drawn tree.**

## 8.5 Honest limitations (§3.6)

The authors list three, and you should remember them when citing DAMOV:
1. **NDP design space:** they used the *same* core type/count for host and NDP to isolate data-movement effects; they did *not* model real area/thermal limits on how many NDP cores you can fit. So this is a *characterization*, not a finished NDP design.
2. **Function-level (not whole-app) analysis:** they analyze one function at a time. This is right for NDP (you offload *parts* of a program) and catches per-phase behavior, but it **ignores data movement *between* functions** — if you offload function A to NDP and B stays on the host, and they constantly pass data, that cost isn't counted.
3. **Overestimating NDP potential:** real NDP has overheads they didn't model — keeping host and NDP caches coherent, synchronizing NDP cores, virtual-memory support, deciding what to offload. So real-world NDP gains would be somewhat lower than the upper bounds reported.

These aren't fatal — they're the honest boundary of a deliberately-controlled study, and each is flagged as future work.

---

# PART 9 — What the suite is good for: the four case studies (§5)

DAMOV isn't just a paper; it's a *tool*. The case studies are demos of research questions the suite can answer. Each maps to code you've now seen.

**Case Study 1 — Inter-vault communication / NoC (§5.1).** Map functions onto NDP cores in a 6×6 mesh of HMC vaults and measure the overhead of cross-vault traffic — this is exactly the `pim_mode` NoC-hops code in `HMC_Memory.h` (§6.5). Finding: ~40% of memory requests travel 3–4 hops, and <5% stay local; network overhead is 5–26% depending on the function. Lesson: NDP needs smart **data placement** (put data near the core that uses it) and **better NoC designs**.

**Case Study 2 — NDP accelerators (§5.2).** Using the Aladdin accelerator model (and the simulator's dataflow `Accelerator` core, §6.3), they tailor a custom accelerator and compare it placed *near memory* vs. *compute-side*. Picking one function from each of Classes 1a, 1b, 2c: the NDP accelerator gives **1.9×** for the bandwidth-bound 1a function (DRKYolo), **1.25×** for the latency-bound 1b function (PLYalu), and **no benefit** for the compute-bound 2c function (PLY3mm) — *exactly* as the class labels predict. Lesson: the classification predicts accelerator behavior too, not just general-purpose cores.

**Case Study 3 — Core model under area/power limits (§5.3).** An HMC vault's logic layer has a tiny budget (4.4 mm², 312 mW). Under that budget you can fit either ~6 out-of-order cores *or* ~128 in-order cores. Result: **128 in-order NDP cores beat 6 OOO NDP cores by ~4× on average** — more, simpler cores win in the logic layer. But the speedup doesn't scale linearly with core count (21× the cores, only ~2× the performance for some functions), because in-order cores can't hide latency on their own. Lesson: future NDP wants the *throughput* of many simple cores plus *some* latency-hiding — a research direction DAMOV can drive.

**Case Study 4 — Fine-grained offloading (§5.4).** Instead of offloading whole functions, offload just the **hottest basic block** (a tiny straight-line code region). Finding: **1–10% of basic blocks cause up to 95.3% of all LLC misses** (and for 65% of workloads, a *single* basic block causes 90–100% of misses!). Offloading just that one block gives up to **1.25×** (vs 1.5× for the whole function) — so most of the benefit with far simpler NDP hardware. Lesson: you can build *much simpler* NDP units if you offload tiny "delinquent" code regions instead of whole functions.

---

# PART 10 — Hands-on: how you would actually run this

Let me walk you through the concrete steps, tying together every file we've discussed. (You don't need to execute these now — this is the mental model of the pipeline.)

**Step 0 — Build the simulator.**
```
cd simulator
sudo sh ./scripts/setup.sh      # installs ZSim deps: pin, scons, libconfig, libhdf5, libelfg0
sh ./scripts/compile.sh         # compiles ZSim+Ramulator → produces ./build/opt/zsim
cd ../
```
(Caveat from the README: Intel Pin 2.14 — which ZSim relies on — breaks on Ubuntu ≥ 20.04. The authors ship a **Docker container** `gfojunior/damov` with everything pre-built to sidestep this. If you ever try to run it, use Docker.)

**Step 1 — Get the workloads.**
```
sh get_workloads.sh             # downloads ~6 GB of pre-built binaries+inputs into ./workloads/
```
Under the hood this uses the bundled `megatools` to pull a tarball from MEGA and unpacks it (you saw `get_workloads.sh`).

**Step 2 — Compile a benchmark** (each suite has a helper):
```
cd workloads/STREAM/
python compile.py               # compiles the apps, unpacks datasets, sets expected filenames
cd ../../
```

**Step 3 — Generate the config files** for a suite from its command file:
```
cd simulator
python scripts/generate_config_files.py command_files/stream_cf
```
This stamps out the 45-per-app `.cfg` files under `config_files/` (and makes the `zsim_stats/` output dirs), as we traced in §6.8.

**Step 4 — Run a single simulation** (host, OOO, 4 cores, STREAM Add):
```
./build/opt/zsim config_files/host_ooo/no_prefetch/stream/4/Add_Add.cfg
```
ZSim fast-forwards to the ROI (the magic-op hook), then cycle-accurately simulates the offload region, feeding cache misses to Ramulator's HMC model. The stats land in `zsim_stats/host_ooo/no_prefetch/4/stream_Add_Add.zsim.out`.

**Step 5 — Run the NDP version** (same app, `pim_ooo`):
```
./build/opt/zsim config_files/pim_ooo/stream/4/Add_Add.cfg
```

**Step 6 — Parse and compute metrics:**
```
python scripts/get_stats_per_app.py zsim_stats/host_ooo/no_prefetch/4/stream_Add_Add.zsim.out
python scripts/get_stats_per_app.py zsim_stats/pim_ooo/4/stream_Add_Add.zsim.out
```
You get IPC, MPKI, LFMR for each. Speedup = IPC(pim)/IPC(host). For STREAM Add this is ~1.58×, LFMR≈1 on host → **Class 1a, NDP win** — the method confirms itself.

**To run a *new* program of your own:** (1) `#include "zsim_hooks.h"` and wrap your hotspot with `zsim_roi_begin()/…_end()` (and the PIM-function markers); (2) make a config (copy a template, set cores/caches/`pimMode`); (3) run ZSim. That's the "instrumenting new applications" recipe in the README §(3).

---

# PART 11 — One-page cheat sheet (keep this)

**The problem.** Data movement (CPU↔DRAM) dominates time and energy in modern apps (the "memory wall"). Two fixes: compute-centric (caches, prefetchers) vs memory-centric (NDP/PIM — put compute in the 3D-memory logic layer; ~3.7× bandwidth via STREAM). Question: *which fix helps which program, and why?*

**Why old shortcuts fail.** Roofline (AI) and MPKI each see only one facet; they can't tell apart programs that NDP helps, hurts, or doesn't touch.

**The five metrics.**
- *Temporal locality* (independent): do I reuse data? → caches can help.
- *Spatial locality* (independent): do I use nearby data? → prefetch can help. (Not a class definer.)
- *AI* (dependent): ops per cache line → high = compute-bound.
- *MPKI* (dependent): DRAM trips / 1000 instr → high (>10) = bandwidth pressure.
- *LFMR* (dependent, **new**): L3 misses ÷ L1 misses → ~1 means deep caches are useless (NDP-friendly); ~0 means caches do the job (CPU-friendly).

**The method (3 steps).** (1) VTune top-down → keep functions with MemBound>30% & ≥3% cycles. (2) Measure locality → split into low/high temporal. (3) Simulate on {Host / Host+prefetch / NDP} × cores{1,4,16,64,256}; collect AI/MPKI/LFMR + LFMR trend → classify.

**The six classes & NDP verdict.**

| Class | Fingerprint | Root cause | NDP? |
|---|---|---|---|
| 1a | lowTL, lowAI, highLFMR, **highMPKI** | DRAM **bandwidth** | ✅ big win |
| 1b | lowTL, lowAI, highLFMR, **lowMPKI** | DRAM **latency** | ✅ win (latency/energy) |
| 1c | lowTL, lowMPKI, **LFMR↓ with cores** | **L1/L2 capacity** | ⚠️ wins at low cores |
| 2a | **highTL**, lowMPKI, **LFMR↑ with cores** | **L3 contention** | ⚠️ wins at high cores |
| 2b | highTL, lowAI, low/med LFMR, lowMPKI | **L1 capacity** | ➖ neutral; saves SRAM |
| 2c | highTL, lowLFMR, lowMPKI, **highAI** | **compute** | ❌ slower on NDP |

**The simulator (DAMOV-SIM).** ZSim (cores+caches+coherence+prefetch, via Pin) + Ramulator (HMC DRAM). Configs = templates with `NUMBER_CORES` filled in. **Host vs NDP difference = delete L2/L3 + set `pimMode=true`** (CPU joins memory's clock domain, skips off-chip link, optionally models 6×6-vault NoC). Metrics from `get_stats_per_app.py`: `MPKI=L3miss/(instr/1000)`, `LFMR=L3miss/L1miss`, `IPC=instr/max(cycles)`, speedup=IPC_pim/IPC_host.

**Validated:** 97% accuracy on 100 held-out functions; robust to L3 size (NUCA), to core type (in-order vs OOO), and confirmed by independent hierarchical clustering.

**The artifact.** 144 functions, 74 apps, 16 suites, in 6 labeled classes — the first open benchmark suite for studying data-movement bottlenecks, plus the open simulator. Use it to study NDP data placement, accelerators, core choice, and fine-grained offloading.

---

*That's the whole study, from the physics of why memory is slow, through the five metrics and the three-step method, into the six bottleneck classes, and all the way down to the C++ and Python in the repository that turns each idea into a number. If you internalize the LFMR intuition (§4.6) and the six-class tree (§7.0), you understand DAMOV.*
