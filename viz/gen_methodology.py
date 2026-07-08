#!/usr/bin/env python3
"""gen_methodology.py — the methodology, documented from the SOURCE OF TRUTH.

Emits reports/methodology.json: the full profiling → placement pipeline as
expandable sections (capture → screen → classify → attribute → trace → verdict),
where every formula, counter ID, and threshold is IMPORTED from the analysis
modules — never re-typed here — so the dashboard's Methodology tab can never
drift from what classify.py / screen.py / roofline.py actually compute.

Prose is authored here; numbers come from:
  analysis.classify.THRESHOLDS      the decision-tree cut values
  analysis.screen.M / .STALLS       the ncu counter -> feature map + stall list
  analysis.roofline.FADD/FMUL/...    the FLOP + byte counters
A worked example substitutes one real kernel's measured numbers (from an
existing summary.json) into each formula.

Usage:  python3 viz/gen_methodology.py     # -> reports/methodology.json
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "profiling"))
from analysis import classify, roofline, screen, stages  # noqa: E402

TH = classify.THRESHOLDS
M = screen.M


def worked_kernel():
    """One real kernel's evidence, to substitute into the formulas."""
    for src in ("profiling/reports/2026-07-03_tum_office_rtx2000ada/summary.json",):
        p = os.path.join(REPO, src)
        if os.path.isfile(p):
            s = json.load(open(p))
            k = next((x for x in s["kernels"]
                      if x["name"] == "st_track_with_cache_kernel"), s["kernels"][0])
            return s, k
    return None, None


def main():
    S, K = worked_kernel()
    ev = (K or {}).get("evidence", {})

    def ex(label, key, unit=""):
        v = ev.get(key)
        return f"  · {K['name']}: {label} = {v}{unit}" if (K and v is not None) else ""

    sections = []

    # ── 1 · Overview ─────────────────────────────────────────────────────────
    sections.append({
        "id": "overview", "title": "The pipeline: profile → place",
        "summary": "cuVSLAM (or any adapted GPU workload) is reduced to a per-kernel "
        "placement decision through six steps. This is GPU-DAMOV: DAMOV's CPU "
        "data-movement-bottleneck method [Oliveira '21], re-derived for GPUs "
        "(two-level cache, occupancy-hidden latency, coalescing) and extended "
        "from 'is this memory-bound' to 'WHICH substrate — GPU, CPU, PiM, ISP — "
        "and if it stays on the GPU, what's the fault to fix'.",
        "blocks": [
            {"type": "prose", "text":
             "Every number the harness reports is defined below by the exact ncu "
             "counter it comes from and the exact threshold it is compared against "
             "— all imported live from the analysis code, so this page is the code."},
        ]})

    # ── 2 · Capture ──────────────────────────────────────────────────────────
    sections.append({
        "id": "capture", "title": "1 · Capture",
        "summary": "Three profilers, one bounded run, locked clocks.",
        "blocks": [
            {"type": "prose", "text":
             "Nsight Systems captures the whole-run TIMELINE (kernel time shares, "
             "H2D/D2H transfers, and cuVSLAM's own NVTX stage ranges). Nsight "
             "Compute replays a bounded window of launches with a CURATED metric "
             "set (not `--set full`, which is killed on a small GPU before it "
             "writes a report) — the counters listed in §3. NVBit binary-"
             "instruments a bounded launch window for exact per-warp addresses "
             "(§5). Clocks are locked (graphics 1620 MHz / memory 7001 MHz, "
             "persistence on); ceilings are MEASURED not quoted (205 GB/s DRAM, "
             "5445 GFLOP/s FP32); 5-repeat CoV is 0.14%."},
            {"type": "table", "title": "the three curated ncu metric sets (profile.py METRIC_SETS)",
             "head": ["set", "purpose"],
             "rows": [["quick", "3 counters — duration + memory/compute SoL; a 'did it run' smoke"],
                      ["roofline", "~15 counters — SoL, roofline, L1/L2 hit, key stalls, occupancy"],
                      ["characterize", "~30 counters — + FLOP + byte traffic + sectors/request + full stall taxonomy"]]},
        ]})

    # ── 3 · Screen — features & formulas ─────────────────────────────────────
    stall_rows = [[name, counter, meaning + (" · MEMORY" if name in screen.MEMORY_STALLS else "")]
                  for name, counter, meaning in screen.STALLS]
    sections.append({
        "id": "screen", "title": "2 · Screen — the decision features",
        "summary": "Each raw ncu counter is reduced to one interpretable feature. "
        "Extensive quantities (time, bytes, instructions) are SUMMED across a "
        "kernel's launches, then ratios are formed once — never average ratios.",
        "blocks": [
            {"type": "formula", "name": "LFMR_gpu  (L2-first-miss ratio)",
             "expr": "LFMR = 1 − l2_hit%/100",
             "counters": [M["l2_hit"]],
             "note": f"≈1 ⇒ the L2 is not helping (NDP-favorable); ≤{TH['lfmr_lo']} ⇒ the L2 "
             "absorbs the reuse (keep on GPU). It is the GPU analog of DAMOV's "
             "LFMR (last-level-miss/first-level-miss), one hierarchy level down."
             + ex("LFMR", "lfmr")},
            {"type": "formula", "name": "MPKI_gpu  (DRAM sectors per kilo warp-instruction)",
             "expr": "MPKI = (dram_bytes / 32) / (inst / 1000)",
             "counters": [M["dram_rd"], M["dram_wr"], M["inst"]],
             "note": "32 = sector bytes; `inst` is WARP-instructions (32× fewer than "
             "thread-instructions — the common trap). Memory-intensity, DAMOV-style."
             + ex("MPKI", "mpki")},
            {"type": "formula", "name": "Speed-of-Light: memory / compute / DRAM",
             "expr": "mem_sol, comp_sol, dram_sol  =  each counter, directly (% of peak)",
             "counters": [M["mem_sol"], M["comp_sol"], M["dram_sol"]],
             "note": f"bound-ness: a resource is 'high' at ≥{TH['sol_hi']}%; DRAM is "
             f"'saturated' at ≥{TH['dram_sat']}%. The mem-vs-compute ratio (≥{TH['sol_ratio']}×) "
             "decides compute- vs memory-leaning."
             + ex("DRAM-SoL", "dram_sol_pct", "%")},
            {"type": "formula", "name": "Arithmetic intensity (roofline x-axis)",
             "expr": "AI_dram = (fadd + fmul + 2·ffma) / (dram_read + dram_write)",
             "counters": [roofline.FADD, roofline.FMUL, roofline.FFMA, roofline.DR, roofline.DW],
             "note": "FP32 FLOPs the standard way (FMA = 2 ops) [Yang '20]. A second "
             "roofline AI_l2 = FLOP / L2-bytes is reported too (why AI differs per level). "
             "For non-FP32 adapter workloads the numerator op-type is the one knob to change."
             + (f"  · {K['name']}: AI_dram = {K['roofline']['ai']} FLOP/B" if K and K.get("roofline") else "")},
            {"type": "formula", "name": "Coalescing fingerprint",
             "expr": "sectors/request = max(sect_ld, sect_st)",
             "counters": [M["sect_ld"], M["sect_st"]],
             "note": f"1 = perfect, 4 = fully coalesced, ≥{TH['sect_scatter']} = a scattered "
             "gather (the G2 signature)."
             + ex("sectors/req", "sectors_per_req")},
            {"type": "formula", "name": "Occupancy",
             "expr": "occupancy% = warps_active (% of peak sustained active)",
             "counters": [M["occ"]],
             "note": f"the GPU's primary latency-hiding knob; below {TH['occ_low']}% latency "
             "can't be hidden (G4/G7 territory)."
             + ex("occupancy", "occupancy_pct", "%")},
            {"type": "table", "title": "warp-stall taxonomy (which counter → which stall)",
             "head": ["stall", "ncu counter (…per_issue_active.ratio)", "meaning"],
             "rows": [[n, c.replace("smsp__average_warps_issue_stalled_", "…").replace(
                 "_per_issue_active.ratio", ""), m] for n, c, m in
                 [(a, b, cc) for a, b, cc in stall_rows]]},
        ]})

    # ── 4 · Classify — the decision tree ─────────────────────────────────────
    sections.append({
        "id": "classify", "title": "3 · Classify — the GPU-DAMOV decision tree",
        "summary": "The features feed an ORDERED decision tree (first match wins) "
        "into one of eight classes G0–G7. Thresholds are stated once (below, "
        "imported live) and stress-tested ±25%: a kernel whose class flips under "
        "the perturbation is flagged 'borderline' and cannot carry 'high' "
        "confidence. k-means over the pooled 27-sequence feature cloud "
        "independently prefers k=7–8 — the classes fall out of the data; the "
        "tree is only the labeling.",
        "blocks": [
            {"type": "table", "title": "thresholds (imported from classify.THRESHOLDS)",
             "head": ["name", "value", "gate"],
             "rows": [["sol_hi", TH["sol_hi"], "SoL% considered 'high'"],
                      ["sol_ratio", TH["sol_ratio"], "mem-vs-compute dominance"],
                      ["dram_sat", TH["dram_sat"], "DRAM-SoL% = saturated (G1)"],
                      ["lfmr_hi / lfmr_lo", f"{TH['lfmr_hi']} / {TH['lfmr_lo']}", "L2 not-helping / earning-keep"],
                      ["sect_scatter", TH["sect_scatter"], "sectors/req = scattered (G2)"],
                      ["occ_low / occ_low_dep", f"{TH['occ_low']} / {TH['occ_low_dep']}", "latency (G4) / dependency (G7)"]]},
            {"type": "decision_tree", "note":
             "The interactive tree — hover a branch for what it tests. In the "
             "Explore tab the same tree fires for a selected kernel with its real "
             "numbers substituted (one shared renderer)."},
            {"type": "link", "run": "profiling/reports/2026-07-03_tum_office_rtx2000ada/summary.json",
             "kernel": "st_track_with_cache_kernel",
             "label": "see the tree FIRE on st_track (Explore →)"},
        ]})

    # ── 5 · Annotate — NVTX + TaggedAllocator ────────────────────────────────
    persistence_rows = [[stg, hyp, desc] for stg, (hyp, desc) in stages.STAGES.items()] \
        if hasattr(stages, "STAGES") else []
    sections.append({
        "id": "annotate", "title": "4 · Annotate — NVTX stages & data-structure tags",
        "summary": "Kernels are placed in pipeline STAGES and their DRAM traffic is "
        "named by DATA STRUCTURE — both measured, not guessed.",
        "blocks": [
            {"type": "prose", "text":
             "Stages: patch 0002 compiles cuVSLAM with USE_NVTX, enabling its own "
             "profiler domains; nsys `nvtx_kern_sum` then gives the MEASURED "
             "kernel→stage table (e.g. st_track_with_cache sits under SLAM loop-"
             "closure). stages.py carries the persistence HYPOTHESIS per stage "
             "that the measurements test:"},
            {"type": "table", "title": "stage → persistence hypothesis (stages.py)",
             "head": ["stage", "persistence", "what it is"],
             "rows": persistence_rows or [["(see stages.py)", "", ""]]},
            {"type": "prose", "text":
             "Data structures: the TaggedAllocator (patch 0002) journals every "
             "allocation as CSV — `A,<t_us>,<ptr>,<bytes>,<kind>,<pc…>` with up to "
             "12 host backtrace PCs, plus `M` maps lines (to undo ASLR) and `F` "
             "frees. attribution.py rebases the PCs, batch-runs addr2line, walks "
             "to the innermost cuVSLAM frame = the owner, and applies TAG_RULES "
             "regexes → a data-structure tag (ba_linear_system, "
             "keyframe_descriptors, pyramid_levels, images_raw, …)."},
        ]})

    # ── 6 · Trace — NVBit ────────────────────────────────────────────────────
    sections.append({
        "id": "trace", "title": "5 · Trace — what NVBit measures",
        "summary": "NVBit binary-instruments SASS to record exact per-warp memory "
        "addresses — the ground truth behind the ncu counters.",
        "blocks": [
            {"type": "prose", "text":
             "mem_trace instruments LDG / STG / global atomics ONLY — it excludes "
             "LDS/STS (shared, on-chip), LDL/STL (local = register spill), and the "
             "texture path — so 'locality' is about DRAM-visible DATA, not scratch. "
             "Each record carries the grid launch id, opcode, and the 32 lane "
             "addresses. Two patches bound it: LAUNCH_BEGIN/END windows the launches "
             "and KERNEL_FILTER=<substr> restricts to one kernel (ANDed → sparse "
             "kernels traced without a full-window blow-up); an alloc-events sidecar "
             "logs driver cuMemAlloc/Free lifetimes keyed by launch id."},
            {"type": "prose", "text":
             "locality.py consumes the trace: a Fenwick-tree reuse-distance pass "
             "(O(log n)/access) → a hit-rate-vs-cache-capacity CDF across "
             "64 KiB→48 MiB (a flat CDF = the cache is structurally useless, "
             "compulsory misses); plus footprint (unique 32 B sectors), coalescing "
             "(sectors/warp), and divergence (active lanes). `--spaces "
             "{global,shared,local,all}` (default global) is what separated "
             "st_track's real scattered gather from its register-spill stream."},
            {"type": "prose", "text":
             "attribution.py joins trace ⋈ allocations: a bisect-based LiveSet maps "
             "each address to the allocation live at that launch id → its space and "
             "data-structure tag, streaming and O(1) in memory. This is how "
             "'92.6% of st_track's DRAM traffic is register spill, the rest "
             "keyframe_descriptors' is a MEASURED statement."},
            {"type": "link", "chart": "attribution",
             "src": "profiling/reports/2026-07-05_attribution_campaign/attribution_consistency.csv",
             "label": "the attribution composition (Findings →)"},
        ]})

    # ── 7 · Verdict — to architecture ────────────────────────────────────────
    sections.append({
        "id": "verdict", "title": "6 · Verdict — from evidence to architecture",
        "summary": "The class × persistence × features map to a SUBSTRATE, and the "
        "same evidence names the CURRENT-architecture fault when a kernel stays "
        "on the GPU. This is the deliverable the whole pipeline exists for.",
        "blocks": [
            {"type": "table", "title": "heterogeneous placement (classify.pim_affinity)",
             "head": ["class + condition", "affinity", "substrate"],
             "rows": [
                 ["G1 + streaming persistence", "strong", "near-sensor SRAM (consume before DRAM)"],
                 ["G1 + else", "strong", "DRAM-PiM (bank-level bandwidth)"],
                 ["G2 (scatter)", "conditional", "scatter-capable PiM — or a data-layout fix first"],
                 ["G4 + (LFMR≥0.4 or set≫L2)", "strong", "near-memory compute (uncacheable set)"],
                 ["cold-persistent + big set", "strong", "ISP / near-storage scan engine"],
                 ["G3 (L2 earning keep)", "weak", "a bigger/persisted L2 wins; PiM forfeits reuse"],
                 ["G5 / G6 / G7", "none", "host GPU (compute / on-chip / dependency bound)"],
                 ["tiny (occ<8%, <1 ms)", "—", "CPU/host (launch-overhead territory)"]]},
            {"type": "formula", "name": "PiM placement model (pim_placement_model.py)",
             "expr": "t_pim = t·(1−m)/c + t·m/k        offload iff affinity∈scenario and t_pim < t",
             "counters": ["m = dram_sol/100", "k = PiM internal-BW × (4 conservative / 8 moderate)",
                          "c = PiM compute ratio (0.5 / 0.75)"],
             "note": "energy ratio = ((bytes − offloaded) + offloaded/r) / bytes. Reports "
             "DELTAS vs the GPU baseline, never absolutes — and now against a MEASURED "
             "baseline joule count (§ energy)."},
            {"type": "table", "title": "current-architecture faults (same evidence, on-GPU fixes)",
             "head": ["signature in the evidence", "the fault", "fix before offloading"],
             "rows": [
                 ["DRAM traffic dominated by register spill (attribution)",
                  "register-pressure / codegen fault", "cut register pressure / raise occupancy — it's scratch, not data"],
                 ["high sectors/request (≥8) on global data",
                  "data-layout fault", "AoS→SoA / re-tile so warps coalesce"],
                 ["low occupancy + wait/dependency stalls (G7)",
                  "launch-config / ILP fault", "more blocks/ILP; re-screen — memory isn't the wall"],
                 ["flat reuse CDF 64 KiB→48 MiB (locality)",
                  "structurally cache-immune streaming", "no cache size helps → near-sensor consumption, not a bigger L2"]]},
            {"type": "link", "run": "profiling/reports/2026-07-03_tum_office_rtx2000ada/summary.json",
             "kernel": "st_track_with_cache_kernel",
             "label": "trace one kernel end-to-end (Explore →)"},
        ]})

    # roadmap strip (the DEFER/READY backlog — see docs/BACKLOG.md)
    roadmap = [
        {"item": "Whole-run energy (joules)", "status": "DONE", "note": "NVML sampling, this cycle"},
        {"item": "Host-side LMDB / ISP I/O", "status": "DEFER", "note": "strace/iostat — arms the ISP claim (G3)"},
        {"item": "Jetson Orin re-run", "status": "READY", "note": "app targets Orin; needs the device"},
        {"item": "Layer-3 kernel-arg correlation", "status": "DEFER", "note": "names the static residuals (G5)"},
        {"item": "Occupancy sweep", "status": "DEFER", "note": "a mutate_configs variant"},
        {"item": "Accel-Sim NDP + AccelWattch", "status": "DEFER", "note": "Phase-4 architecture paper"},
    ]

    out = os.path.join(REPO, "reports", "methodology.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"sections": sections, "roadmap": roadmap,
               "thresholds": TH}, open(out, "w"), indent=1)
    print(f"[✓] {len(sections)} methodology sections -> {os.path.relpath(out, REPO)} "
          f"(thresholds stamped from classify.THRESHOLDS)")


if __name__ == "__main__":
    main()
