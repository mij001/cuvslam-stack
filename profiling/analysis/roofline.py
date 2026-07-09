#!/usr/bin/env python3
"""roofline.py — per-kernel roofline placement from ncu counters + hw ceilings.

FLOPs are counted the standard way (Yang20 / NERSC): FP32 adds + muls + 2×FMA
from the SASS thread-instruction counters. Arithmetic intensity is computed at
the DRAM level (FLOP per DRAM byte); L1/L2 traffic intensities are reported in
the CSV when the byte counters are present. Ceilings come from the --hw
descriptor (theoretical until ERT-measured values are filled in — the figure
says which).

Needs a capture made with profile.py's 'characterize' metric set (the FLOP and
byte counters); on captures made with the older 'roofline' set it degrades to
reporting what is missing instead of failing.

Emits:  roofline.csv, fig_roofline.svg

Usage:  python -m analysis.roofline <ncu_results_dir> --hw profiling/hw/<gpu>.toml [--out DIR]
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common, stages, svgfig  # noqa: E402

TIME = "gpu__time_duration.sum"
FADD = "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"
FMUL = "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"
FFMA = "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
DR, DW = "dram__bytes_read.sum", "dram__bytes_write.sum"
L1B, L2B = "l1tex__t_bytes.sum", "lts__t_bytes.sum"

# Op-type FLOP numerators — the one knob for retargeting the roofline to
# NON-FP32 workloads (the "profile any GPU codebase" goal): a DNN in fp16, a
# solver in fp64, an integer-heavy kernel. Each entry is (counters, weights);
# FMA counts as 2. `auto` picks the op-type with the most measured FLOPs, so a
# workload's AI numerator is correct with zero config (cuVSLAM → fp32; an fp16
# matmul → fp16). Tensor-core FLOPs need dedicated ops-path counters (arch-
# specific) and are intentionally left out rather than approximated wrongly.
OPTYPE_FLOPS = {
    "fp32": ([FADD, FMUL, FFMA], [1.0, 1.0, 2.0]),
    "fp16": (["smsp__sass_thread_inst_executed_op_hadd_pred_on.sum",
              "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum",
              "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum"], [1.0, 1.0, 2.0]),
    "fp64": (["smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
              "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",
              "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"], [1.0, 1.0, 2.0]),
    "int":  (["smsp__sass_thread_inst_executed_op_integer_pred_on.sum"], [1.0]),
}


def aggregate(launches, optype="auto"):
    """optype: which FLOP numerator drives AI ('auto' = the op-type with the
    most measured FLOPs, so the roofline is correct per workload with no config;
    or fp32/fp16/fp64/int to force one)."""
    by = {}
    for lk in launches:
        by.setdefault(lk.kernel, []).append(lk)
    rows = []
    for kernel, lks in by.items():
        def tot(metric):
            vals = [l.m(metric) for l in lks if l.m(metric) == l.m(metric)]
            return sum(vals) if vals else float("nan")

        # FLOPs per op-type from whatever counters are present
        flops_by = {}
        for ot, (counters, weights) in OPTYPE_FLOPS.items():
            vals = [(tot(c), w) for c, w in zip(counters, weights)]
            present = [v * w for v, w in vals if v == v]
            if present:
                flops_by[ot] = sum(present)
        chosen = (optype if optype != "auto" and optype in flops_by
                  else (max(flops_by, key=flops_by.get) if flops_by else None))
        flops = flops_by.get(chosen, float("nan"))

        t = tot(TIME)
        dram = tot(DR) + tot(DW)
        row = {
            "kernel": kernel, "stage": stages.stage_of(kernel),
            "launches": len(lks), "time_s": t, "flops": flops,
            "ai_optype": chosen or "",
            "dram_bytes": dram, "l1_bytes": tot(L1B), "l2_bytes": tot(L2B),
            "ai_dram": flops / dram if flops == flops and dram and dram == dram else float("nan"),
            "ai_l2": flops / tot(L2B) if flops == flops and tot(L2B) == tot(L2B) and tot(L2B) else float("nan"),
            "gflops": flops / t / 1e9 if flops == flops and t and t == t else float("nan"),
            "dram_gbps": dram / t / 1e9 if dram == dram and t and t == t else float("nan"),
        }
        for ot in OPTYPE_FLOPS:                       # keep per-op-type FLOPs
            row[f"flops_{ot}"] = flops_by.get(ot, float("nan"))
        rows.append(row)
    rows.sort(key=lambda r: -(r["time_s"] if r["time_s"] == r["time_s"] else 0))
    return rows


def emit(rows, hw, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    p = os.path.join(out_dir, "roofline.csv")
    common.write_csv(p, ["kernel", "stage", "launches", "time_ms", "ai_optype", "flops",
                         "dram_bytes", "l1_bytes", "l2_bytes", "ai_dram",
                         "ai_l2", "gflops", "dram_gbps"],
                     [[r["kernel"], r["stage"], r["launches"],
                       round(r["time_s"] * 1e3, 4) if r["time_s"] == r["time_s"] else "",
                       r.get("ai_optype", ""),
                       *[f"{r[k]:.6g}" if r[k] == r[k] else "" for k in
                         ("flops", "dram_bytes", "l1_bytes", "l2_bytes", "ai_dram",
                          "ai_l2", "gflops", "dram_gbps")]] for r in rows])
    written.append(p)

    pts = [r for r in rows if r["ai_dram"] == r["ai_dram"] and r["gflops"] == r["gflops"]
           and r["ai_dram"] > 0 and r["gflops"] > 0]
    if not pts:
        return written, "no FLOP/byte counters in this capture — re-profile with --metrics characterize"

    dev = hw.get("device", {})
    comp = hw.get("compute", {})
    memc = hw.get("memory", {})
    bw_meas = float(memc.get("dram_gbps_measured") or 0.0)
    bw = bw_meas or float(memc.get("dram_gbps_theoretical") or 0.0)
    bw_kind = "measured" if bw_meas else "theoretical"
    fp_meas = float(comp.get("fp32_gflops_measured") or 0.0) / 1000.0
    tflops = fp_meas or float(comp.get("fp32_tflops_theoretical") or 0.0)
    fp_kind = "measured" if fp_meas else "theoretical"
    l2_gbps = float(memc.get("l2_gbps_estimate") or 0.0) or None
    total_t = sum(r["time_s"] for r in pts) or 1.0
    points = []
    for r in pts:
        radius = 4 + 10 * math.sqrt(r["time_s"] / total_t)
        points.append((r["kernel"], r["ai_dram"], r["gflops"],
                       svgfig.stage_color(r["stage"], stages.ORDER), radius))
    p = os.path.join(out_dir, "fig_roofline.svg")
    svgfig.roofline(p, f'Roofline — {dev.get("name", "GPU")} '
                       f'(DRAM {bw:g} GB/s {bw_kind}, FP32 {tflops:g} TFLOP/s '
                       f'{fp_kind}; size ∝ GPU-time share)',
                    points, dram_gbps=bw, fp32_tflops=tflops, l2_gbps=l2_gbps)
    written.append(p)
    return written, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--hw", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ai-optype", default="auto",
                    choices=["auto", *OPTYPE_FLOPS.keys()],
                    help="FLOP numerator for AI (default auto = dominant op-type)")
    args = ap.parse_args(argv)
    csv_path = common.find_derived(args.run_dir, "ncu_metrics.csv")
    if not csv_path:
        raise SystemExit(f"no ncu_metrics.csv under {args.run_dir}/derived")
    rows = aggregate(common.load_ncu_csv(csv_path), optype=args.ai_optype)
    out = args.out or os.path.join(args.run_dir, "derived")
    written, warn = emit(rows, common.load_hw(args.hw), out)
    for p in written:
        print(f"[✓] {p}")
    if warn:
        print(f"[!] {warn}")


if __name__ == "__main__":
    main()
