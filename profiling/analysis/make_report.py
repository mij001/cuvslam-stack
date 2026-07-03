#!/usr/bin/env python3
"""make_report.py — build the cuVSLAM memory-characterization report.

Orchestrates the analysis modules over one or more results dirs and writes a
self-contained report directory (markdown + SVG figures + CSVs) suitable for
committing. This is the Slice-2 deliverable: DAG, DAMOV Step-1 screen, roofline,
stall breakdown, bandwidth breakdown — plus, when a [slam] capture is supplied,
the loop-closure delta (kernels that appear only when the SLAM layer runs).

Everything is stdlib + the sibling modules; runs headless anywhere, even with
no GPU (it only reads derived CSVs).

Usage:
  python -m analysis.make_report --hw profiling/hw/mx450_sm75.toml \
      --nsys <nsys_run_dir> --ncu <ncu_run_dir> \
      [--nsys-slam <dir>] [--ncu-slam <dir>] \
      [--out profiling/reports/<name>]

With no --nsys/--ncu it picks the newest matching runs under profiling/results/.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import bandwidth, build_dag, classify, common, roofline, screen, stages  # noqa: E402

PROF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(PROF_ROOT, "results")


def provenance_block(run_dir):
    meta = common.load_metadata(run_dir)
    gpu = meta.get("gpu", {})
    lines = [f"- **run:** `{os.path.basename(run_dir)}`",
             f"  - GPU {gpu.get('name', '?')} · driver {gpu.get('driver_version', '?')}"
             f" · clocks {gpu.get('graphics_clock', '?')}/{gpu.get('memory_clock', '?')}",
             f"  - config `{meta.get('config', '?')}` · frames {meta.get('frame_override') or 'as-config'}"
             f" · cuvslam {meta.get('cuvslam_version', '?')}"]
    for tool in ("nsys_version", "ncu_version"):
        if meta.get(tool):
            lines.append(f"  - {tool.split('_')[0]} {meta[tool]}")
    if meta.get("ncu_config"):
        nc = meta["ncu_config"]
        lines.append(f"  - ncu window: launch-skip {nc.get('launch_skip')} · "
                     f"launch-count {nc.get('launch_count')} · metrics `{nc.get('metrics')}`")
    return "\n".join(lines)


def rel_fig(out_dir, path):
    return os.path.relpath(path, out_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hw", required=True)
    ap.add_argument("--nsys", default=None)
    ap.add_argument("--ncu", default=None)
    ap.add_argument("--nsys-slam", default=None)
    ap.add_argument("--ncu-slam", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="cuVSLAM per-kernel memory characterization")
    args = ap.parse_args(argv)

    nsys_dir = args.nsys or common.newest_run(RESULTS_ROOT, "nsys")
    ncu_dir = args.ncu or common.newest_run(RESULTS_ROOT, "ncu")
    if not nsys_dir or not ncu_dir:
        raise SystemExit("need at least one nsys and one ncu results dir "
                         f"(searched {RESULTS_ROOT})")
    hw = common.load_hw(args.hw)
    hw_name = os.path.splitext(os.path.basename(args.hw))[0]
    out_dir = args.out or os.path.join(
        PROF_ROOT, "reports", f"{datetime.now().strftime('%Y-%m-%d')}_{hw_name}")
    figs = os.path.join(out_dir, "figures")
    data = os.path.join(out_dir, "data")
    os.makedirs(figs, exist_ok=True)
    os.makedirs(data, exist_ok=True)

    # ── run the modules, collecting outputs into the report dir ─────────────
    dag = build_dag.build(nsys_dir)
    dag_files = build_dag.emit(dag, figs)
    scr_rows = screen.aggregate(common.load_ncu_csv(common.find_derived(ncu_dir, "ncu_metrics.csv")))
    scr_files = screen.emit(scr_rows, figs)
    roof_rows = roofline.aggregate(common.load_ncu_csv(common.find_derived(ncu_dir, "ncu_metrics.csv")))
    roof_files, roof_warn = roofline.emit(roof_rows, hw, figs)
    bw_rows = bandwidth.aggregate(common.load_ncu_csv(common.find_derived(ncu_dir, "ncu_metrics.csv")))
    bw_files = bandwidth.emit(bw_rows, hw, figs, nsys_dir=nsys_dir)

    slam_dag = slam_scr_rows = None
    slam_only = []
    if args.nsys_slam:
        slam_dag = build_dag.build(args.nsys_slam)
        base_kernels = {k["kernel"] for k in dag["per_kernel"]}
        slam_only = [k for k in slam_dag["per_kernel"] if k["kernel"] not in base_kernels]
    if args.ncu_slam:
        slam_scr_rows = screen.aggregate(
            common.load_ncu_csv(common.find_derived(args.ncu_slam, "ncu_metrics.csv")))
        screen.emit(slam_scr_rows, os.path.join(figs, "slam"))

    # move CSVs to data/, keep SVGs in figures/
    for f in list(dag_files) + list(scr_files) + list(roof_files) + list(bw_files):
        if f.endswith(".csv"):
            shutil.move(f, os.path.join(data, os.path.basename(f)))

    # ── the report ───────────────────────────────────────────────────────────
    dev = hw.get("device", {})
    memc = hw.get("memory", {})
    R = []
    R.append(f"# {args.title}\n")
    R.append(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by "
             f"`analysis/make_report.py` — headless, stdlib-only.*\n")
    R.append("## 1. Provenance\n")
    R.append(f"**Hardware descriptor:** `{os.path.relpath(args.hw, PROF_ROOT)}` — "
             f"{dev.get('name')} ({dev.get('arch')}, sm_{str(dev.get('compute_cap','')).replace('.','')}, "
             f"{dev.get('sms')} SMs, L2 {int(memc.get('l2_bytes',0))//1024} KiB, "
             f"DRAM {memc.get('dram_gbps_theoretical')} GB/s theoretical"
             f"{', ECC' if memc.get('ecc') else ', no ECC'}). "
             f"Role: **{dev.get('role','?')}**.\n")
    for d in filter(None, [nsys_dir, ncu_dir, args.nsys_slam, args.ncu_slam]):
        R.append(provenance_block(d) + "\n")
    if dev.get("role") == "prototype":
        R.append("> ⚠ **Prototype-hardware caveat:** this GPU cannot lock clocks; "
                 "numbers here support methodology and relative comparisons, not "
                 "publishable absolutes. Re-run on the production descriptor for "
                 "report-grade data.\n")

    R.append("## 2. Pipeline decomposition (kernel→stage DAG)\n")
    kpf = dag["kernels_per_frame"]
    R.append(f"Workload: {dag['frames'] or '?'} frames, {dag['kernel_instances']} kernel "
             f"launches ({f'{kpf:.1f}/frame' if kpf else 'n/a'}), "
             f"{dag['unique_kernels']} unique kernels, "
             f"total GPU time {dag['total_gpu_ms']:.1f} ms.\n")
    R.append(f"![stage share](figures/fig_stage_share.svg)\n")
    st_rows = []
    for s in stages.ORDER:
        agg = dag["per_stage"][s]
        if agg["instances"] == 0:
            continue
        st_rows.append([s, stages.persistence_of(s), stages.describe(s),
                        round(100.0 * agg["ns"] / (dag["total_gpu_ms"] * 1e6), 1),
                        agg["instances"], len(agg["kernels"])])
    R.append(common.md_table(
        ["stage", "persistence hypothesis", "what it is", "GPU time %", "launches", "kernels"],
        st_rows) + "\n")

    R.append("## 3. DAMOV Step-1 screen — which kernels are memory-bound\n")
    R.append("Rule (GPU adaptation): *memory-bound* if Memory-SoL ≥ 40% and ≥ 1.5× "
             "Compute-SoL; *memory-latency* if both SoLs are low but the dominant "
             "warp stall is a memory stall. Time-weighted across launches.\n")
    R.append(f"![screen](figures/fig_screen.svg)\n")
    scr_table = [[r["kernel"], r["stage"], r["verdict"],
                  r["mem_sol"], r["comp_sol"], r["l1_hit"], r["l2_hit"],
                  r["sect_ld"]] for r in scr_rows[:16]]
    R.append(common.md_table(
        ["kernel", "stage", "verdict", "MemSoL%", "CompSoL%", "L1 hit%", "L2 hit%",
         "sectors/req (ld)"], scr_table) + "\n")
    if os.path.isfile(os.path.join(figs, "fig_stalls.svg")):
        R.append("### Warp-stall breakdown\n")
        R.append(f"![stalls](figures/fig_stalls.svg)\n")

    R.append("## 4. Roofline placement\n")
    if roof_warn:
        R.append(f"> ⚠ {roof_warn}\n")
    else:
        R.append(f"![roofline](figures/fig_roofline.svg)\n")
        roof_table = [[r["kernel"], r["stage"],
                       r["ai_dram"], r["gflops"], r["dram_gbps"]]
                      for r in roof_rows[:14] if r["ai_dram"] == r["ai_dram"]]
        R.append(common.md_table(
            ["kernel", "stage", "AI (FLOP/DRAM-byte)", "GFLOP/s", "DRAM GB/s"],
            roof_table) + "\n")

    R.append("## 5. DRAM traffic by stage\n")
    R.append(f"![stage bytes](figures/fig_stage_bytes.svg)\n")
    if os.path.isfile(os.path.join(figs, "fig_kernel_gbps.svg")):
        R.append(f"![kernel gbps](figures/fig_kernel_gbps.svg)\n")

    R.append("## 6. Loop-closure (SLAM layer) delta\n")
    if slam_dag:
        R.append(f"SLAM capture: {slam_dag['frames'] or 'full-sequence'} frames, "
                 f"{slam_dag['unique_kernels']} unique kernels "
                 f"(baseline had {dag['unique_kernels']}).\n")
        if slam_only:
            R.append("Kernels present **only** with `[slam]` enabled — the "
                     "cold-persistent candidates:\n")
            R.append(common.md_table(
                ["kernel", "stage", "persistence", "GPU time %", "launches"],
                [[k["kernel"], k["stage"], k["persistence"],
                  round(k["pct_gpu_time"], 2), k["instances"]] for k in slam_only]) + "\n")
            if slam_scr_rows:
                slam_by = {r["kernel"]: r for r in slam_scr_rows}
                bw_slam = bandwidth.aggregate(common.load_ncu_csv(
                    common.find_derived(args.ncu_slam, "ncu_metrics.csv")))
                bpl = {r["kernel"]: r["bytes_per_launch"] for r in bw_slam}
                rows_ = []
                for k in slam_only:
                    r = slam_by.get(k["kernel"])
                    if not r:
                        continue
                    rows_.append([r["kernel"], r["verdict"], r["mem_sol"], r["comp_sol"],
                                  r["l2_hit"], r["occ"], r["sect_ld"],
                                  f"{bpl.get(r['kernel'], float('nan'))/1e6:.1f}"])
                if rows_:
                    R.append("Their per-kernel memory profile (ncu, `characterize` set):\n")
                    R.append(common.md_table(
                        ["kernel", "verdict", "MemSoL%", "CompSoL%", "L2 hit%",
                         "occupancy%", "sectors/req (ld)", "DRAM MB/launch"], rows_) + "\n")
        else:
            R.append("> No SLAM-only kernels appeared in this window — either no "
                     "loop closure fired, or the SLAM layer ran on CPU. Check the "
                     "nsys timeline and widen the frame window / choose a "
                     "revisit-heavy sequence.\n")
    else:
        R.append("> Not captured in this report — run a `[slam]`-enabled config "
                 "(e.g. configs/euroc_v101_slam_profile.toml) under nsys and pass "
                 "--nsys-slam.\n")

    R.append("## 7. GPU-DAMOV classification — PiM/ISP candidates\n")
    R.append("Bottleneck classes per the GPU-adapted DAMOV taxonomy "
             "(`suggestions_and_summuries/Adapting_DAMOV_to_GPU.md` §6; "
             "[Oliveira21] for the CPU original). This is the NCU-counter "
             "**first-cut** classification — single-point LFMR_gpu (= 1 − L2 hit), "
             "MPKI, DRAM-SoL, coalescing, occupancy, stall taxonomy. The gated "
             "Slice-3 trace/simulation track refines it (LFMR-vs-#SM trend, "
             "divergence, true reuse distance) but is not required to produce it.\n")
    cls_paths = [d for d in [ncu_dir, args.ncu_slam] if d]
    cls_rows = classify.run(cls_paths, hw)
    for f in classify.emit(cls_rows, hw, figs):
        if f.endswith(".csv"):
            shutil.move(f, os.path.join(data, os.path.basename(f)))
    R.append("![classification](figures/fig_classification.svg)\n")
    R.append("**Synthesis — stage → dominant class → PiM/ISP affinity** "
             "(time-weighted within stage):\n")
    R.append(common.md_table(["stage", "persistence", "dominant class", "share",
                              "PiM affinity", "substrate"],
                             classify.synthesis(cls_rows)) + "\n")
    top_cls = [[r["kernel"], r["class"], r["confidence"], r["stability"],
                r["pim"], r["substrate"], r["why"]] for r in cls_rows[:14]]
    R.append("Per-kernel placement (top by profiled time; full table in "
             "`data/classification.csv`). *Stability* = the class survives all "
             "decision thresholds perturbed ±25%:\n")
    R.append(common.md_table(["kernel", "class", "conf", "stability", "PiM",
                              "substrate", "rationale"], top_cls) + "\n")
    n_borderline = sum(1 for r in cls_rows if r["stability"] != "stable")
    R.append(f"Threshold sensitivity: {len(cls_rows) - n_borderline}/{len(cls_rows)} "
             f"kernels keep their class under ±25% threshold perturbation; "
             f"{n_borderline} are borderline (flagged above and in the CSV).\n")

    R.append("## 8. Persistence-class evidence so far\n")
    verd = {r["kernel"]: r for r in scr_rows}
    ev_rows = []
    for s in stages.ORDER:
        agg = dag["per_stage"][s]
        if agg["instances"] == 0:
            continue
        ks = [verd[k] for k in agg["kernels"] if k in verd]
        n_mem = sum(1 for k in ks if k["verdict"] in ("memory-bound", "memory-latency"))
        ev_rows.append([s, stages.persistence_of(s),
                        f"{n_mem}/{len(ks)} profiled kernels memory-bound" if ks
                        else "not in ncu window"])
    R.append(common.md_table(["stage", "hypothesis", "evidence in this capture"], ev_rows) + "\n")
    R.append("Methodology caveats: ncu flushes caches between replay passes, so hit "
             "rates are cold-start (steady-state needs the gated Accel-Sim track); "
             "SoL/stall/traffic counters are robust. Simulated numbers, when they "
             "arrive, are reported as deltas, not absolutes.\n")

    report = os.path.join(out_dir, "report.md")
    open(report, "w").write("\n".join(R))
    print(f"[✓] {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
