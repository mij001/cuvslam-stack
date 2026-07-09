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
  python profiling/harness/profile.py --config configs/profiling/kitti06_profile.toml \
         --profiler nsys --hw profiling/hw/mx450_sm75.toml

  # Nsight Compute, targeted roofline/SoL/stall metrics
  python profiling/harness/profile.py --config configs/profiling/kitti06_profile.toml \
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

# Profiler binaries — overridable so a driver-matched Nsight (e.g. the CUDA-12.9
# ncu 2025.2 needed on a downgraded 575 driver) can be used without touching PATH.
NCU = os.environ.get("NCU_BIN", "ncu")
NSYS = os.environ.get("NSYS_BIN", "nsys")
# NVBit mem_trace shared object (built in the podman wheel-builder; needs driver ≤575)
NVBIT_TOOL = os.environ.get("NVBIT_TOOL", os.path.join(
    REPO_ROOT, "external_repos", "nvbit_release_x86_64", "tools", "mem_trace", "mem_trace.so"))

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

# "characterize" = roofline + FLOP counters (arithmetic intensity, Yang20 method)
#                + L1/L2 byte traffic (hierarchical roofline)
#                + sectors/request (coalescing fingerprint; 4=coalesced, 32=scattered)
#                + the fuller warp-stall taxonomy (Cao23 buckets).
# ~30 metrics, ~12 passes — still targeted; use for the Slice-2 report captures.
METRIC_SETS["characterize"] = METRIC_SETS["roofline"] + [
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    # op-type FLOPs for non-FP32 workloads (the roofline retargets automatically;
    # analysis/roofline.py OPTYPE_FLOPS) — fp16 + integer are the common cases
    "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_integer_pred_on.sum",
    "l1tex__t_bytes.sum",
    "lts__t_bytes.sum",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
]


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
    root = grep1(config, "root") or grep1(config, "path") or ""
    base = os.path.basename(root.rstrip("/")) or "seq"
    # KITTI roots end in .../sequences/06 -> use parentdir+base for clarity
    parent = os.path.basename(os.path.dirname(root.rstrip("/")))
    if parent == "sequences":
        return f"kitti{base}"
    base = re.sub(r"^rgbd_dataset_", "", base)          # TUM verbosity
    return re.sub(r"[^A-Za-z0-9]+", "", base)[:24] or "seq"


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


DATASET_VARS = {
    # env var -> default location; configs reference ${VAR}/... so the same
    # file runs on any host — per-machine locations come from the environment.
    "CUVSLAM_DATASETS": os.path.expanduser("~/Projects/cuvslam_datasets"),
    "CUVSLAM_DATA2": "/mnt/data",   # the workstation's read-only dataset volume
}


def expand_dataset_root(config_text):
    """Expand the ${CUVSLAM_*} dataset-root variables in the config copy."""
    for var, default in DATASET_VARS.items():
        config_text = config_text.replace("${" + var + "}",
                                          os.environ.get(var, default))
    return config_text


def derive_launch_window(nsys_run_dir, warm_frames, profile_launches):
    """Steady-state ncu window from a prior nsys run of the same workload.

    ncu bounds by kernel-launch index, not frame. kernels/frame comes from the
    nsys kernel summary (total instances ÷ frames of that run); the returned
    launch_skip lands the ncu window right after `warm_frames` tracked frames.
    """
    sys.path.insert(0, os.path.join(REPO_ROOT, "profiling"))
    from analysis import build_dag  # stdlib-only
    dag = build_dag.build(nsys_run_dir)
    kpf = dag["kernels_per_frame"]
    if not kpf:
        raise SystemExit(f"--auto-window: cannot derive kernels/frame from {nsys_run_dir} "
                         "(no frame count in its used_config/metadata)")
    skip = int(round(kpf * warm_frames))
    print(f"[auto-window] {nsys_run_dir}: {kpf:.1f} kernels/frame × {warm_frames} warm frames "
          f"→ --launch-skip {skip}, --launch-count {profile_launches}")
    return skip, profile_launches


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
    p.add_argument("--profiler", required=True, choices=["nsys", "ncu", "nvbit"])
    p.add_argument("--adapter", default="auto",
                   help="workload adapter: cuvslam (run.py <config>, the default) or "
                        "command (any argv from the config's [workload] table); auto "
                        "picks command when [workload] is present. See profiling/adapters.py")
    p.add_argument("--hw", required=True, help="profiling/hw/*.toml descriptor")
    p.add_argument("--tag", default="default")
    p.add_argument("--venv-python", default=DEFAULT_VENV_PY,
                   help="python that can import cuvslam (default: stack cuvslam_venv)")
    p.add_argument("--frames", default=None, metavar="START:COUNT",
                   help="override [run].start_index/max_frames for the profiled run")
    # ncu knobs
    p.add_argument("--metrics", default="roofline",
                   help="named set (roofline|characterize|quick), literal comma list, or 'full'")
    p.add_argument("--launch-skip", type=int, default=100)
    p.add_argument("--launch-count", type=int, default=12)
    p.add_argument("--auto-window", default=None, metavar="NSYS_DIR:WARM_FRAMES:LAUNCHES",
                   help="derive --launch-skip/--launch-count for a steady-state ncu "
                        "window from a prior nsys results dir, e.g. "
                        "results/<nsys_run>:200:250 (overrides the two flags above)")
    p.add_argument("--kernel-filter", default=None, metavar="REGEX",
                   help="ncu: profile only kernels whose name matches this regex "
                        "(e.g. 'st_.*cache' for the SLAM keyframe-cache kernels)")
    p.add_argument("--cache-control", default="all", choices=["all", "none"],
                   help="ncu: 'all' flushes caches between replay passes (cold-start "
                        "hit rates, default); 'none' keeps them warm (upper bound). "
                        "Capture BOTH to bracket the true steady-state hit rates.")
    # nsys knobs
    p.add_argument("--nsys-traces", default="cuda,nvtx,osrt")
    p.add_argument("--nsys-sample", default="cpu")
    p.add_argument("--gpu-warmup", type=float, default=0.0, metavar="SECONDS",
                   help="pre-capture clock warm-up (env/gpu_warmup.py) — use ~8 on "
                        "hosts that cannot lock clocks so every run starts at the "
                        "same sustained clock state (see METHODOLOGY.md §4.2)")
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

    # Resolve the workload config (dataset root + --frames override, recorded copy).
    used_config = os.path.join(derived, "used_config.toml")
    text = expand_dataset_root(open(config).read())
    frame_range = None
    if args.frames:
        start, count = (int(x) for x in args.frames.split(":"))
        text = apply_frame_override(text, start, count)
        frame_range = {"start_index": start, "max_frames": count}
    open(used_config, "w").write(text)

    # Steady-state window: derive the ncu launch bounds from a prior nsys run.
    if args.auto_window:
        try:
            nsys_dir, warm, launches = args.auto_window.rsplit(":", 2)
            args.launch_skip, args.launch_count = derive_launch_window(
                os.path.abspath(nsys_dir), int(warm), int(launches))
        except ValueError:
            p.error("--auto-window expects NSYS_DIR:WARM_FRAMES:LAUNCHES")

    # The workload command comes from the adapter: cuvslam = the stack runner
    # (default, byte-compatible); command = any argv from [workload] — profile
    # ANY GPU codebase (torch, CUDA benchmarks, image pipelines, ...).
    sys.path.insert(0, os.path.join(REPO_ROOT, "profiling"))
    from adapters import pick_adapter  # noqa: E402
    adapter = pick_adapter(used_config, REPO_ROOT, args.venv_python, args.adapter)
    workload = adapter.argv()
    workload_env = adapter.env()
    if workload_env:
        os.environ.update(workload_env)   # profiler children inherit it

    # ── metadata.json (provenance) ───────────────────────────────────────────
    gpu = nvidia_smi_gpu()
    cuvslam_ver = subprocess.run(
        [args.venv_python, "-c", "import cuvslam;print(getattr(cuvslam,'__version__','?'))"],
        capture_output=True, text=True).stdout.strip() or "unknown"
    meta = {
        "profiler": {"nsys": "nsight_systems", "ncu": "nsight_compute",
                     "nvbit": "nvbit_memtrace"}[args.profiler],
        "date": datetime.now(timezone.utc).astimezone().isoformat(),
        "timestamp": ts, "hostname": os.uname().nodename, "tag": args.tag,
        "hw_descriptor": os.path.relpath(hw, REPO_ROOT), "hw_name": hw_name,
        "gpu": gpu,
        "cuda_version": tool_version(["nvcc", "--version"]).split("release")[-1].strip(" ,") or "unknown",
        "python_version": subprocess.run([args.venv_python, "--version"], capture_output=True, text=True).stdout.strip().replace("Python ", ""),
        "cuvslam_version": cuvslam_ver,
        "config": os.path.relpath(config, REPO_ROOT),
        "frame_override": frame_range,
        "adapter": adapter.name,
        "workload_cmd": " ".join(os.path.relpath(c, REPO_ROOT) if os.path.exists(c) else c for c in workload),
    }

    if args.gpu_warmup > 0:
        sh([sys.executable, os.path.join(REPO_ROOT, "profiling", "env", "gpu_warmup.py"),
            "--seconds", str(args.gpu_warmup)])
        meta["gpu_warmup_s"] = args.gpu_warmup
        meta["gpu_clocks_post_warmup"] = nvidia_smi_gpu()

    rc = 1
    if args.profiler == "nsys":
        meta["nsys_version"] = tool_version([NSYS, "--version"])
        meta["nsys_config"] = {"traces": args.nsys_traces, "sample": args.nsys_sample}
        out = os.path.join(raw, "profile")
        json.dump(meta, open(os.path.join(run_dir, "metadata.json"), "w"), indent=2)
        cmd = [NSYS, "profile", f"--trace={args.nsys_traces}", f"--sample={args.nsys_sample}",
               "--output", out, "--force-overwrite=true", *workload]
        rc = run_profiler(cmd, args.timeout, cwd=REPO_ROOT).returncode
        rep = out + ".nsys-rep"
        if os.path.isfile(rep):
            print(f"[✓] {rep} ({human(os.path.getsize(rep))})")
            # textual + CSV kernel summary for the DAG (--force-export refreshes the
            # SQLite sidecar so a re-run never trips the "export older than input" error)
            with open(os.path.join(derived, "nsys_stats.txt"), "w") as fh:
                sh([NSYS, "stats", "--force-export=true", rep], stdout=fh, stderr=subprocess.STDOUT)
            sh([NSYS, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
                "--force-export=true", "--output", os.path.join(derived, "kern_sum"), rep])
        else:
            print("[✗] no .nsys-rep produced", file=sys.stderr)

    elif args.profiler == "nvbit":
        # Binary-instrumentation memory trace (32B-sector level) via NVBit
        # mem_trace, bounded to a launch window (LAUNCH_BEGIN/END — the same
        # windowing idea as ncu's --launch-skip/count) so traces stay disk-
        # bounded; outside the window kernels run native, so the app completes
        # a FULL run and any [eval] block still produces a comparable eval.txt.
        if not os.path.isfile(NVBIT_TOOL):
            p.error(f"NVBit tool not found: {NVBIT_TOOL} (build it in the podman "
                    "wheel-builder or set NVBIT_TOOL)")
        begin, end = args.launch_skip, args.launch_skip + args.launch_count
        meta["nvbit_config"] = {"tool": os.path.relpath(NVBIT_TOOL, REPO_ROOT),
                                "launch_begin": begin, "launch_end": end,
                                "kernel_filter": args.kernel_filter}
        json.dump(meta, open(os.path.join(run_dir, "metadata.json"), "w"), indent=2)
        env = dict(os.environ)
        env["CUDA_INJECTION64_PATH"] = NVBIT_TOOL
        env["LAUNCH_BEGIN"], env["LAUNCH_END"] = str(begin), str(end)
        if args.kernel_filter:
            env["KERNEL_FILTER"] = args.kernel_filter
        # mem_trace disassembles with nvdisasm at runtime — make sure it's on PATH
        if os.path.isdir("/opt/cuda/bin"):
            env["PATH"] = "/opt/cuda/bin" + os.pathsep + env.get("PATH", "")
        rep = os.path.join(raw, "mem_trace.txt.gz")
        print(f"  $ LAUNCH_BEGIN={begin} LAUNCH_END={end} CUDA_INJECTION64_PATH="
              f"{os.path.relpath(NVBIT_TOOL, REPO_ROOT)} {' '.join(workload)}", flush=True)
        import gzip
        try:
            with gzip.open(rep, "wt") as gz:
                proc = subprocess.Popen(workload, cwd=REPO_ROOT, env=env,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True)
                for line in proc.stdout:
                    gz.write(line)
                proc.wait(timeout=args.timeout)
                rc = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f"[✗] nvbit run exceeded {args.timeout}s — aborted", file=sys.stderr)
            rc = 124
        if os.path.isfile(rep) and os.path.getsize(rep) > 0:
            print(f"[✓] {rep} ({human(os.path.getsize(rep))})")
        else:
            print("[✗] no mem_trace output produced", file=sys.stderr)

    else:  # ncu
        metrics = resolve_metrics(args.metrics)
        meta["ncu_version"] = tool_version([NCU, "--version"])
        meta["ncu_config"] = {"metrics": args.metrics, "n_metrics": len(metrics) if metrics else "full",
                              "launch_skip": args.launch_skip, "launch_count": args.launch_count,
                              "auto_window": args.auto_window, "kernel_filter": args.kernel_filter,
                              "cache_control": args.cache_control}
        out = os.path.join(raw, "kernels")
        json.dump(meta, open(os.path.join(run_dir, "metadata.json"), "w"), indent=2)
        cmd = [NCU, "--target-processes", "all",
               "--launch-skip", str(args.launch_skip), "--launch-count", str(args.launch_count),
               "--clock-control", "none",            # don't let ncu fight the (unlockable) laptop clocks
               "--cache-control", args.cache_control,
               "-o", out, "--force-overwrite"]
        if args.kernel_filter:
            cmd += ["--kernel-name", f"regex:{args.kernel_filter}"]
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
                sh([NCU, "--import", rep, "--csv", "--page", "raw"], stdout=fh, stderr=subprocess.DEVNULL)
        else:
            print("[✗] no .ncu-rep produced — see console above", file=sys.stderr)

    # whole-run telemetry (measured during run_profiler) → into metadata.json
    if LAST_ENERGY is not None or LAST_HOST_IO is not None:
        if LAST_ENERGY is not None:
            meta["energy"] = LAST_ENERGY
        if LAST_HOST_IO is not None:
            meta["host_io"] = LAST_HOST_IO
        json.dump(meta, open(os.path.join(run_dir, "metadata.json"), "w"), indent=2)
        if LAST_ENERGY and LAST_ENERGY.get("available"):
            print(f"[energy] {LAST_ENERGY['joules']} J  "
                  f"(mean {LAST_ENERGY['mean_w']} W, peak {LAST_ENERGY['peak_w']} W)")
        if LAST_HOST_IO and LAST_HOST_IO.get("available"):
            print(f"[host-io] storage read {LAST_HOST_IO['storage_read_mb']} MB, "
                  f"mmap page-in {LAST_HOST_IO['mmap_pagein_mb']} MB, "
                  f"peak host RSS {LAST_HOST_IO['peak_host_rss_mb']} MB")

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


LAST_ENERGY = None   # {joules, mean_w, peak_w, ...} — whole-run NVML sampling
LAST_HOST_IO = None  # {storage_read_mb, mmap_pagein_mb, peak_host_rss_mb, ...}


def run_profiler(cmd, timeout, cwd):
    """Run the profiled workload while sampling, over its lifetime: GPU board
    power (whole-run joules) and host-side I/O + memory of the process tree.
    Both are best-effort and recorded next to the run's metadata."""
    global LAST_ENERGY, LAST_HOST_IO
    print(f"  $ {' '.join(cmd)}", flush=True)
    sys.path.insert(0, REPO_ROOT)
    try:
        from profiling.env.energy import EnergySampler
        esampler = EnergySampler()
    except Exception:  # noqa: BLE001 — telemetry is best-effort, never blocks a run
        esampler = None
    try:
        from profiling.env.host_io import HostIOSampler
        hsampler = HostIOSampler()
    except Exception:  # noqa: BLE001
        hsampler = None

    import contextlib
    with (esampler or contextlib.nullcontext()), (hsampler or contextlib.nullcontext()):
        proc = subprocess.Popen(cmd, cwd=cwd)
        if hsampler is not None:
            hsampler.set_root(proc.pid)          # track this process tree
        try:
            proc.wait(timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print(f"[✗] profiler exceeded {timeout}s — aborted", file=sys.stderr)
            rc = 124
    if esampler is not None:
        LAST_ENERGY = esampler.result()
    if hsampler is not None:
        LAST_HOST_IO = hsampler.result()

    class _R:  # noqa — minimal CompletedProcess stand-in
        returncode = rc
    return _R()


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.0f}TB"


if __name__ == "__main__":
    raise SystemExit(main())
