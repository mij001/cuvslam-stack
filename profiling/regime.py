#!/usr/bin/env python3
"""regime.py — the cohesive profiling pipeline: one command, every technique.

The stack's profiling techniques were runnable separately (nsys timeline, ncu
metric sets, NVBit traces, NVTX attribution, then per-technique analyses).
This driver composes them into ONE regime per workload config, so a capture is
never missing the piece another analysis needs:

    ┌ capture ─────────────────────────────────────────────────────┐
    │ 1 nsys timeline            (whole run; kernel/stage DAG)      │
    │ 2 auto-window              (steady-state launch window from 1)│
    │ 3 ncu characterize         (roofline+SoL+stalls+coalescing)   │
    │ 4 nvbit mem_trace          (32B-sector trace, same window)    │
    └──────────────────────────────────────────────────────────────┘
    ┌ analyze ─────────────────────────────────────────────────────┐
    │ 5 build_dag   nsys  → kernel→stage DAG (dag.csv, stages)      │
    │ 6 screen      ncu   → DAMOV Step-1 memory-bound screen        │
    │ 7 roofline    ncu   → per-kernel roofline placement           │
    │ 8 classify    ncu   → DAMOV-GPU classes + PiM/ISP candidacy   │
    │ 9 locality    nvbit → footprints, reuse-distance CDFs, overlap│
    └──────────────────────────────────────────────────────────────┘
    manifest.json ties every produced dir together; derived/ collects the
    analysis CSVs. analysis/substrate.py then reads one-or-many regime dirs
    to rank GPU/CPU/PiM/ISP substrate candidates and their dynamics.

Usage:
  python3 profiling/regime.py --config configs/base/kitti06_stereo_slam.toml \
      --hw profiling/hw/dellworkstation_sm89.toml
  # partial re-runs / skips:
  ... --steps nsys,window,ncu,analyze          # no nvbit trace
  ... --steps analyze --manifest <regime_dir>  # re-analyze an existing capture
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
RESULTS_ROOT = os.path.join(HERE, "results")
PROFILE_PY = os.path.join(HERE, "harness", "profile.py")
sys.path.insert(0, HERE)

DEFAULT_STEPS = "nsys,window,ncu,nvbit,analyze"

# On a downgraded 575 driver the system ncu (2026.2/2025.3, CUDA-13) rejects the
# driver; profile.py honours $NCU_BIN, so default it to the CUDA-12.9 ncu 2025.2
# when present and unset (matches scripts/validation_regime.sh). Harmless
# elsewhere — profile.py falls back to plain "ncu" if the path does not exist.
_NCU_129 = os.path.expanduser("~/ncu2025/nsight_compute-linux-x86_64-2025.2.1.3-archive/ncu")
if "NCU_BIN" not in os.environ and os.path.isfile(_NCU_129):
    os.environ["NCU_BIN"] = _NCU_129


def sh(cmd, **kw):
    print(f"[regime] $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd], **kw)


def capture(cmd):
    """Run profile.py, return the results dir it reports (or None)."""
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    sys.stdout.write(r.stdout[-2000:])
    if r.stderr:
        sys.stderr.write(r.stderr[-1000:])
    m = re.search(r"(?m)^Results: (.+)$", r.stdout)
    return os.path.join(REPO_ROOT, m.group(1)) if m else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="workload TOML (a base or mutated config)")
    ap.add_argument("--hw", required=True, help="profiling/hw/*.toml")
    ap.add_argument("--steps", default=DEFAULT_STEPS,
                    help=f"comma list of {DEFAULT_STEPS}")
    ap.add_argument("--manifest", default=None,
                    help="existing regime dir (resume / re-analyze)")
    ap.add_argument("--warm-frames", type=int, default=200,
                    help="frames considered warm-up when deriving the steady window")
    ap.add_argument("--window-launches", type=int, default=250,
                    help="launches profiled inside the steady-state window")
    ap.add_argument("--metrics", default="characterize",
                    help="ncu metric set for the characterize step")
    ap.add_argument("--python", default=os.path.join(REPO_ROOT, "cuvslam_venv", "bin", "python"))
    ap.add_argument("--timeout", type=int, default=3000)
    args = ap.parse_args(argv)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]

    config = os.path.abspath(args.config)
    hw = os.path.abspath(args.hw)
    hw_name = os.path.splitext(os.path.basename(hw))[0]
    tag = os.path.splitext(os.path.basename(config))[0]

    # regime dir + manifest (append/resume-able)
    if args.manifest:
        rdir = os.path.abspath(args.manifest)
        man = json.load(open(os.path.join(rdir, "manifest.json")))
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        rdir = os.path.join(RESULTS_ROOT, f"{ts}_{tag}_regime_{hw_name}")
        os.makedirs(os.path.join(rdir, "derived"), exist_ok=True)
        man = {"config": os.path.relpath(config, REPO_ROOT), "hw": os.path.relpath(hw, REPO_ROOT),
               "tag": tag, "created": ts, "captures": {}, "analyses": {}}
    manifest_path = os.path.join(rdir, "manifest.json")

    def save():
        json.dump(man, open(manifest_path, "w"), indent=2)

    prof = [args.python, PROFILE_PY, "--config", config, "--hw", hw,
            "--gpu-warmup", "0", "--timeout", str(args.timeout)]

    # ── capture ──────────────────────────────────────────────────────────────
    if "nsys" in steps:
        d = capture(prof + ["--profiler", "nsys"])
        if not d:
            sys.exit("[regime] nsys capture failed")
        man["captures"]["nsys"] = os.path.relpath(d, REPO_ROOT); save()

    nsys_dir = man["captures"].get("nsys") and os.path.join(REPO_ROOT, man["captures"]["nsys"])
    skip = count = None
    if "window" in steps:
        if not nsys_dir:
            sys.exit("[regime] window step needs a prior nsys capture")
        from harness.profile import derive_launch_window  # reuse, one implementation
        try:
            skip, count = derive_launch_window(nsys_dir, args.warm_frames, args.window_launches)
            how = "derived"
        except SystemExit:
            # unbounded run (max_frames = 0) — kernels/frame underivable; fall
            # back to the launch window proven accuracy-neutral in the
            # profiler-neutrality report (skip past warm-up, bounded count)
            skip, count = 3000, args.window_launches
            how = "fallback"
            print(f"[regime] auto-window underivable (unbounded run) — "
                  f"falling back to launch-skip={skip}, count={count}")
        man["window"] = {"launch_skip": skip, "launch_count": count,
                         "warm_frames": args.warm_frames, "how": how}; save()
    elif man.get("window"):
        skip, count = man["window"]["launch_skip"], man["window"]["launch_count"]

    if "ncu" in steps:
        w = ["--launch-skip", str(skip), "--launch-count", str(count)] if skip is not None else []
        d = capture(prof + ["--profiler", "ncu", "--metrics", args.metrics] + w)
        if d:
            man["captures"]["ncu"] = os.path.relpath(d, REPO_ROOT); save()

    if "nvbit" in steps:
        w = ["--launch-skip", str(skip if skip is not None else 1000),
             "--launch-count", str(count if count is not None else 200)]
        d = capture(prof + ["--profiler", "nvbit"] + w)
        if d:
            man["captures"]["nvbit"] = os.path.relpath(d, REPO_ROOT); save()

    # ── analyze ──────────────────────────────────────────────────────────────
    if "analyze" in steps:
        derived = os.path.join(rdir, "derived")
        os.makedirs(derived, exist_ok=True)
        py = [args.python]
        ncu_dir = man["captures"].get("ncu") and os.path.join(REPO_ROOT, man["captures"]["ncu"])
        nvb_dir = man["captures"].get("nvbit") and os.path.join(REPO_ROOT, man["captures"]["nvbit"])

        if nsys_dir:
            sh(py + [os.path.join(HERE, "analysis", "build_dag.py"), nsys_dir, "--out", derived])
            man["analyses"]["dag"] = "derived/dag.csv"
        if ncu_dir:
            sh(py + [os.path.join(HERE, "analysis", "screen.py"), ncu_dir, "--out", derived])
            sh(py + [os.path.join(HERE, "analysis", "roofline.py"), ncu_dir, "--hw", hw, "--out", derived])
            sh(py + [os.path.join(HERE, "analysis", "classify.py"), ncu_dir, "--hw", hw, "--out", derived])
            man["analyses"].update({"screen": "derived/screen.csv",
                                    "roofline": "derived/roofline.csv",
                                    "classification": "derived/classification.csv"})
        if nvb_dir:
            traces = glob.glob(os.path.join(nvb_dir, "raw", "mem_trace.*"))
            if traces:
                with open(os.path.join(derived, "locality.txt"), "w") as fh:
                    sh(py + [os.path.join(HERE, "analysis", "locality.py"), traces[0]],
                       stdout=fh, stderr=subprocess.STDOUT)
                man["analyses"]["locality"] = "derived/locality.txt"
        save()

    print(f"\n[regime] manifest: {os.path.relpath(manifest_path, REPO_ROOT)}")
    print(json.dumps(man, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
