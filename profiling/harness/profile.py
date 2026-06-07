#!/usr/bin/env python3
"""profile.py — one entrypoint to profile cuVSLAM under nsys or ncu.

The workload-under-test is the stack's own TOML runner (`run.py <config>`), so any
dataset the runner supports is profilable. This wrapper:

  * reads a hardware descriptor (profiling/hw/*.toml) for provenance + clock policy,
  * launches `run.py <config>` under the chosen profiler with a TARGETED metric set
    (not `ncu --set full`, which is killed on small GPUs before writing a report),
  * writes a versioned results dir with a mandatory metadata.json + raw/ + derived/.

Stdlib only (no tomllib dependency — works on py3.9+). Config/hw fields it needs are
read with light regex; run.py does the real TOML parsing inside the venv.

Examples
--------
  # Nsight Systems timeline (for the DAG)
  python profiling/harness/profile.py --config profiling/configs/kitti06_profile.toml \
         --profiler nsys --hw profiling/hw/mx450_sm75.toml

  # Nsight Compute, targeted roofline/SoL/stall metrics
  python profiling/harness/profile.py --config profiling/configs/kitti06_profile.toml \
         --profiler ncu --hw profiling/hw/mx450_sm75.toml \
         --metrics roofline --launch-skip 100 --launch-count 12
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_VENV_PY = os.path.join(REPO_ROOT, "cuvslam_venv", "bin", "python")
RESULTS_ROOT = os.path.join(REPO_ROOT, "profiling", "results")

# ── Targeted Nsight Compute metric sets ──────────────────────────────────────
# Curated from the Cao23 "gpudb" taxonomy: just the counters that drive the
# roofline / Speed-of-Light / stall / memory numbers. Validated against ncu 2026.2.
# Far fewer replay passes than `--set full` -> actually finishes on a 2 GB GPU.
METRIC_SETS = {
    # ~15 metrics, ~8 passes: throughput+roofline, SoL, L1/L2 hit, key stalls, occupancy
    "roofline": [
        "gpu__time_duration.sum",
        "sm__cycles_elapsed.avg.per_second",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",                 # Compute SoL %
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", # Memory SoL %
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",               # DRAM BW %
        "lts__t_sector_hit_rate.pct",                                       # L2 hit rate
        "l1tex__t_sector_hit_rate.pct",                                     # L1 hit rate
        "lts__t_requests_srcunit_tex_op_read.sum",                          # L2 read requests
        "smsp__inst_executed.sum",
        "sm__warps_active.avg.pct_of_peak_sustained_active",                # achieved occupancy
        "launch__registers_per_thread",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",  # Mem-LD stall
        "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",       # throttle stall
    ],
    # 3 metrics, ~1-2 passes: a fast "did it work" smoke set
    "quick": [
        "gpu__time_duration.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    ],
}


def sh(cmd, **kw):
    """Run, stream output, return CompletedProcess (never raises on nonzero)."""
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    return subprocess.run(cmd, **kw)


def grep1(path, key):
    """First `key = value` (TOML-ish) value in a file, or None. Strips quotes."""
    try:
        with open(path) as fh:
            for line in fh:
                m = re.match(rf"\s*{re.escape(key)}\s*=\s*(.+?)\s*(#.*)?$", line)
                if m:
                    return m.group(1).strip().strip('"').strip("'")
    except OSError:
        pass
    return None


def seq_name_from_config(config):
    root = grep1(config, "root") or ""
    base = os.path.basename(root.rstrip("/")) or "seq"
    # KITTI roots end in .../sequences/06 -> use parentdir+base for clarity
    parent = os.path.basename(os.path.dirname(root.rstrip("/")))
    if parent == "sequences":
        return f"kitti{base}"
    return re.sub(r"[^A-Za-z0-9]+", "", base) or "seq"


def apply_frame_override(config_text, start, count):
    """Return config text with [run].start_index/max_frames set to (start, count)."""
    def set_or_inject(text, key, value):
        pat = re.compile(rf"(?m)^(\s*){re.escape(key)}\s*=.*$")
        if pat.search(text):
            return pat.sub(rf"\g<1>{key} = {value}", text, count=1)
        # inject right after the [run] header
        return re.sub(r"(?m)^(\[run\]\s*)$", rf"\1\n{key} = {value}", text, count=1)
    text = set_or_inject(config_text, "start_index", start)
    text = set_or_inject(text, "max_frames", count)
    return text


def nvidia_smi_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,clocks.current.graphics,clocks.current.memory",
             "--format=csv,noheader"], text=True).strip().splitlines()[0]
        name, drv, gclk, mclk = [x.strip() for x in out.split(",")]
        return {"name": name, "driver_version": drv,
                "graphics_clock": gclk, "memory_clock": mclk}
    except Exception:
        return {"name": "unknown"}


def tool_version(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).splitlines()[0].strip()
    except Exception:
        return "unknown"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="stack runner TOML (the workload)")
    p.add_argument("--profiler", required=True, choices=["nsys", "ncu"])
    p.add_argument("--hw", required=True, help="profiling/hw/*.toml descriptor")
    p.add_argument("--tag", default="default")
    p.add_argument("--venv-python", default=DEFAULT_VENV_PY,
                   help="python that can import cuvslam (default: stack cuvslam_venv)")
    p.add_argument("--frames", default=None, metavar="START:COUNT",
                   help="override [run].start_index/max_frames for the profiled run")
    # ncu knobs
    p.add_argument("--metrics", default="roofline",
                   help="named set (roofline|quick) or a literal comma metric list, or 'full'")
    p.add_argument("--launch-skip", type=int, default=100)
    p.add_argument("--launch-count", type=int, default=12)
    # nsys knobs
    p.add_argument("--nsys-traces", default="cuda,nvtx,osrt")
    p.add_argument("--nsys-sample", default="cpu")
    p.add_argument("--timeout", type=int, default=2400, help="seconds before aborting the profiler")
    args = p.parse_args(argv)

    config = os.path.abspath(args.config)
    hw = os.path.abspath(args.hw)
    for path, what in [(config, "config"), (hw, "hw descriptor")]:
        if not os.path.isfile(path):
            p.error(f"{what} not found: {path}")
    if not os.path.exists(args.venv_python):
        p.error(f"venv python not found: {args.venv_python} (run setup_env.sh)")

    hw_name = os.path.splitext(os.path.basename(hw))[0]
    seq = seq_name_from_config(config)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(RESULTS_ROOT, f"{ts}_{seq}_{args.profiler}_{hw_name}")
    raw, derived = os.path.join(run_dir, "raw"), os.path.join(run_dir, "derived")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(derived, exist_ok=True)

    # Resolve the workload config (apply --frames override into a recorded copy).
    used_config = os.path.join(derived, "used_config.toml")
    text = open(config).read()
    frame_range = None
    if args.frames:
        start, count = (int(x) for x in args.frames.split(":"))
        text = apply_frame_override(text, start, count)
        frame_range = {"start_index": start, "max_frames": count}
    open(used_config, "w").write(text)

    # The workload command: run the stack runner from REPO_ROOT (so imports resolve).
    workload = [args.venv_python, os.path.join(REPO_ROOT, "run.py"), used_config]

    # ── metadata.json (provenance) ───────────────────────────────────────────
    gpu = nvidia_smi_gpu()
    cuvslam_ver = subprocess.run(
        [args.venv_python, "-c", "import cuvslam;print(getattr(cuvslam,'__version__','?'))"],
        capture_output=True, text=True).stdout.strip() or "unknown"
    meta = {
        "profiler": {"nsys": "nsight_systems", "ncu": "nsight_compute"}[args.profiler],
        "date": datetime.now(timezone.utc).astimezone().isoformat(),
        "timestamp": ts, "hostname": os.uname().nodename, "tag": args.tag,
        "hw_descriptor": os.path.relpath(hw, REPO_ROOT), "hw_name": hw_name,
        "gpu": gpu,
        "cuda_version": tool_version(["nvcc", "--version"]).split("release")[-1].strip(" ,") or "unknown",
        "python_version": subprocess.run([args.venv_python, "--version"], capture_output=True, text=True).stdout.strip().replace("Python ", ""),
        "cuvslam_version": cuvslam_ver,
        "config": os.path.relpath(config, REPO_ROOT),
        "frame_override": frame_range,
        "workload_cmd": " ".join(os.path.relpath(c, REPO_ROOT) if os.path.exists(c) else c for c in workload),
    }

    rc = 1
    if args.profiler == "nsys":
        meta["nsys_version"] = tool_version(["nsys", "--version"])
        meta["nsys_config"] = {"traces": args.nsys_traces, "sample": args.nsys_sample}
        out = os.path.join(raw, "profile")
        json.dump(meta, open(os.path.join(run_dir, "metadata.json"), "w"), indent=2)
        cmd = ["nsys", "profile", f"--trace={args.nsys_traces}", f"--sample={args.nsys_sample}",
               "--output", out, "--force-overwrite=true", *workload]
        rc = run_profiler(cmd, args.timeout, cwd=REPO_ROOT).returncode
        rep = out + ".nsys-rep"
        if os.path.isfile(rep):
            print(f"[✓] {rep} ({human(os.path.getsize(rep))})")
            # textual + CSV kernel summary for the DAG (--force-export refreshes the
            # SQLite sidecar so a re-run never trips the "export older than input" error)
            with open(os.path.join(derived, "nsys_stats.txt"), "w") as fh:
                sh(["nsys", "stats", "--force-export=true", rep], stdout=fh, stderr=subprocess.STDOUT)
            sh(["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
                "--force-export=true", "--output", os.path.join(derived, "kern_sum"), rep])
        else:
            print("[✗] no .nsys-rep produced", file=sys.stderr)

    else:  # ncu
        metrics = resolve_metrics(args.metrics)
        meta["ncu_version"] = tool_version(["ncu", "--version"])
        meta["ncu_config"] = {"metrics": args.metrics, "n_metrics": len(metrics) if metrics else "full",
                              "launch_skip": args.launch_skip, "launch_count": args.launch_count}
        out = os.path.join(raw, "kernels")
        json.dump(meta, open(os.path.join(run_dir, "metadata.json"), "w"), indent=2)
        cmd = ["ncu", "--target-processes", "all",
               "--launch-skip", str(args.launch_skip), "--launch-count", str(args.launch_count),
               "--clock-control", "none",            # don't let ncu fight the (unlockable) laptop clocks
               "-o", out, "--force-overwrite"]
        if metrics is None:                          # literal "full"
            cmd += ["--set", "full"]
        else:
            cmd += ["--metrics", ",".join(metrics)]
        cmd += workload
        rc = run_profiler(cmd, args.timeout, cwd=REPO_ROOT).returncode
        rep = out + ".ncu-rep"
        if os.path.isfile(rep):
            print(f"[✓] {rep} ({human(os.path.getsize(rep))})")
            # Export a parseable per-kernel CSV for analysis/.
            with open(os.path.join(derived, "ncu_metrics.csv"), "w") as fh:
                sh(["ncu", "--import", rep, "--csv", "--page", "raw"], stdout=fh, stderr=subprocess.DEVNULL)
        else:
            print("[✗] no .ncu-rep produced — see console above", file=sys.stderr)

    print(f"\nResults: {os.path.relpath(run_dir, REPO_ROOT)}")
    return 0 if (rc == 0 or os.path.isfile(rep)) else rc


def resolve_metrics(spec):
    if spec == "full":
        return None
    if spec in METRIC_SETS:
        return METRIC_SETS[spec]
    if "," in spec or "." in spec:        # literal metric list
        return [m.strip() for m in spec.split(",") if m.strip()]
    raise SystemExit(f"unknown metric set '{spec}' (have: {', '.join(METRIC_SETS)}, or 'full', or a literal list)")


def run_profiler(cmd, timeout, cwd):
    print(f"  $ {' '.join(cmd)}", flush=True)
    try:
        return subprocess.run(cmd, cwd=cwd, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[✗] profiler exceeded {timeout}s — aborted", file=sys.stderr)
        class _R:  # noqa
            returncode = 124
        return _R()


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.0f}TB"


if __name__ == "__main__":
    raise SystemExit(main())
