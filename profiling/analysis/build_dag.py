#!/usr/bin/env python3
"""build_dag.py — turn an nsys kernel summary into the kernel→stage DAG table.

Consumes a results dir from `profile.py --profiler nsys` (its
derived/kern_sum_cuda_gpu_kern_sum.csv), maps every kernel to a canonical
V-SLAM stage (stages.py), and emits:

  dag.csv               per-kernel: stage, persistence class, time share
  dag_stages.csv        per-stage rollup: time share, instances, kernel list
  fig_stage_share.svg   time-weighted stage share bar chart

Also estimates kernels-per-frame (total kernel instances ÷ frames), which is
what --auto-window in profile.py uses to scope a steady-state ncu capture.

Usage:  python -m analysis.build_dag <nsys_results_dir> [--out DIR]
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common, stages, svgfig  # noqa: E402


def frames_of_run(run_dir: str) -> int | None:
    """Frame count of the profiled run: metadata frame_override, else used_config."""
    meta = common.load_metadata(run_dir)
    fo = meta.get("frame_override") or {}
    if fo.get("max_frames"):
        return int(fo["max_frames"])
    cfg = os.path.join(run_dir, "derived", "used_config.toml")
    if os.path.isfile(cfg):
        m = re.search(r"(?m)^\s*max_frames\s*=\s*(\d+)", open(cfg).read())
        if m and int(m.group(1)) > 0:
            return int(m.group(1))
    return None


def build(run_dir: str) -> dict:
    ks_path = common.find_derived(run_dir, "kern_sum.csv") or \
        common.find_derived(run_dir, "cuda_gpu_kern_sum.csv")
    if not ks_path:
        raise SystemExit(f"no kernel-summary CSV under {run_dir}/derived — "
                         "was this an nsys run post-processed by profile.py?")
    ks = common.load_kern_sum(ks_path)
    total_ns = sum(k.total_ns for k in ks) or 1.0
    frames = frames_of_run(run_dir)

    per_kernel = []
    per_stage: dict[str, dict] = {s: {"ns": 0.0, "instances": 0, "kernels": set()}
                                  for s in stages.ORDER}
    for k in ks:
        st = stages.stage_of(k.kernel)
        per_kernel.append({
            "kernel": k.kernel, "stage": st,
            "persistence": stages.persistence_of(st),
            "pct_gpu_time": 100.0 * k.total_ns / total_ns,
            "total_ms": k.total_ns / 1e6, "instances": k.instances,
            "avg_us": k.avg_ns / 1e3,
        })
        agg = per_stage[st]
        agg["ns"] += k.total_ns
        agg["instances"] += k.instances
        agg["kernels"].add(k.kernel)

    n_instances = sum(k.instances for k in ks)
    return {
        "run_dir": run_dir,
        "frames": frames,
        "total_gpu_ms": total_ns / 1e6,
        "kernel_instances": n_instances,
        "kernels_per_frame": (n_instances / frames) if frames else None,
        "unique_kernels": len(ks),
        "per_kernel": sorted(per_kernel, key=lambda r: -r["pct_gpu_time"]),
        "per_stage": per_stage,
    }


def emit(dag: dict, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []

    rows = [[r["kernel"], r["stage"], r["persistence"],
             round(r["pct_gpu_time"], 2), round(r["total_ms"], 3),
             r["instances"], round(r["avg_us"], 1)] for r in dag["per_kernel"]]
    p = os.path.join(out_dir, "dag.csv")
    common.write_csv(p, ["kernel", "stage", "persistence", "pct_gpu_time",
                         "total_ms", "instances", "avg_us"], rows)
    written.append(p)

    st_rows = []
    for s in stages.ORDER:
        agg = dag["per_stage"][s]
        if agg["instances"] == 0:
            continue
        st_rows.append([s, stages.persistence_of(s),
                        round(100.0 * agg["ns"] / (dag["total_gpu_ms"] * 1e6), 2),
                        round(agg["ns"] / 1e6, 3), agg["instances"],
                        " ".join(sorted(agg["kernels"]))])
    p = os.path.join(out_dir, "dag_stages.csv")
    common.write_csv(p, ["stage", "persistence", "pct_gpu_time", "total_ms",
                         "instances", "kernels"], st_rows)
    written.append(p)

    labels = [r[0] for r in st_rows]
    values = [r[2] for r in st_rows]
    colors = [svgfig.stage_color(s, stages.ORDER) for s in labels]
    ann = [f"{v:.1f}%  ({r[1]})" for v, r in zip(values, st_rows)]
    p = os.path.join(out_dir, "fig_stage_share.svg")
    svgfig.hbar(p, "GPU time share by pipeline stage", labels, values,
                colors=colors, annotations=ann)
    written.append(p)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="output dir (default <run_dir>/derived)")
    args = ap.parse_args(argv)
    dag = build(args.run_dir)
    out = args.out or os.path.join(args.run_dir, "derived")
    for p in emit(dag, out):
        print(f"[✓] {p}")
    kpf = dag["kernels_per_frame"]
    print(f"kernels/frame ≈ {kpf:.1f}" if kpf else "kernels/frame: unknown (no frame count)")


if __name__ == "__main__":
    main()
