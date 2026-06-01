#!/usr/bin/env python3
"""Run (or validate) every TOML config in a directory and summarize the results.

Each config runs in its own subprocess, so a missing dataset, absent camera, or
a crash in one config never stops the others. A summary table is printed at the
end with per-config status, frame count, and ATE where evaluation was enabled.

    python run_all.py                      # run every configs/*.toml
    python run_all.py --check              # validate only (no cuvslam needed)
    python run_all.py --configs configs    # explicit directory or glob
    python run_all.py --only euroc_v1_eval,kitti_stereo
    python run_all.py --skip realsense_stereo,webcam_mono
    python run_all.py --timeout 600        # per-config seconds (0 = no limit)

Exit code is 0 only if every selected config succeeded (or validated, in
--check mode).
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
import re
import shlex
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_PY = os.path.join(HERE, "run.py")

_DONE_RE = re.compile(r"\[runner\] done:\s*(\{.*\})")


def discover(spec: str) -> list:
    """Return a sorted list of .toml paths from a directory or a glob."""
    if os.path.isdir(spec):
        return sorted(glob.glob(os.path.join(spec, "*.toml")))
    return sorted(glob.glob(spec))


def _name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _parse_summary(stdout: str) -> dict:
    """Pull the `[runner] done: {...}` dict out of stdout, if present."""
    last = None
    for m in _DONE_RE.finditer(stdout):
        last = m.group(1)
    if not last:
        return {}
    try:
        return ast.literal_eval(last)
    except (ValueError, SyntaxError):
        return {}


def _first_error(text: str) -> str:
    """Best-effort one-line reason from stderr/stdout for a failed run."""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(("error:", "source error:", "[eval] evaluation failed")):
            return s
    # else last non-empty traceback-ish line
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()[:160]
    return "(no output)"


def run_one(config: str, python: str, check: bool, timeout: float, log_dir: str,
            stream: bool = True) -> dict:
    cmd = [python, RUN_PY, config] + (["--check"] if check else [])
    # Always show the exact command being executed.
    print("    $ " + " ".join(shlex.quote(c) for c in cmd), flush=True)

    start = time.time()
    # Merge stderr into stdout and read line-by-line so the runner's own output
    # and any Python tracebacks are shown in full (nothing is hidden/captured-away).
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    timed_out = {"v": False}
    timer = None
    if timeout > 0:
        def _kill():
            timed_out["v"] = True
            proc.kill()
        timer = threading.Timer(timeout, _kill)
        timer.start()

    buf = []
    try:
        for line in proc.stdout:
            buf.append(line)
            if stream:
                sys.stdout.write("      | " + line)
                sys.stdout.flush()
        proc.wait()
    finally:
        if timer is not None:
            timer.cancel()
    elapsed = time.time() - start
    out = "".join(buf)

    if timed_out["v"]:
        status, rc = "TIMEOUT", None
    else:
        status = "OK" if proc.returncode == 0 else "FAIL"
        rc = proc.returncode

    # On failure, if we were NOT streaming, dump the full output so error
    # traces are never hidden.
    if status != "OK" and not stream:
        sys.stdout.write(out if out.endswith("\n") else out + "\n")
        sys.stdout.flush()

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, _name(config) + ".log"), "w") as handle:
            handle.write(out)

    result = {
        "config": config,
        "name": _name(config),
        "status": status,
        "returncode": rc,
        "elapsed": elapsed,
        "summary": _parse_summary(out) if status == "OK" else {},
    }
    if status != "OK" and not check:
        result["reason"] = _first_error(out)
    elif status == "OK" and check:
        result["reason"] = "valid"
    elif status != "OK":
        result["reason"] = _first_error(out)
    return result


def print_table(results: list, check: bool) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY" + ("  (--check: validation only)" if check else ""))
    print("=" * 78)
    if check:
        print(f"{'config':28} {'status':8} {'time(s)':>8}  note")
        for r in results:
            note = r.get("reason", "")
            print(f"{r['name']:28} {r['status']:8} {r['elapsed']:8.1f}  {note}")
    else:
        print(f"{'config':28} {'status':8} {'frames':>7} {'ATE(m)':>9} "
              f"{'RTE%':>7} {'time(s)':>8}  note")
        for r in results:
            s = r["summary"]
            frames = s.get("frames_tracked", "")
            ate = s.get("ate_rmse_m", "")
            rte = s.get("avg_rte_pct", "")
            ate_s = f"{ate:.4f}" if isinstance(ate, (int, float)) else ""
            rte_s = f"{rte:.2f}" if isinstance(rte, (int, float)) else ""
            note = "" if r["status"] == "OK" else r.get("reason", "")
            print(f"{r['name']:28} {r['status']:8} {str(frames):>7} {ate_s:>9} "
                  f"{rte_s:>7} {r['elapsed']:8.1f}  {note}")
    n_ok = sum(1 for r in results if r["status"] == "OK")
    print("-" * 78)
    print(f"{n_ok}/{len(results)} succeeded")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", default=os.path.join(HERE, "configs"),
                   help="directory or glob of .toml configs (default: ./configs)")
    p.add_argument("--check", action="store_true",
                   help="validate configs only (no cuvslam, no tracking)")
    p.add_argument("--only", default="",
                   help="comma-separated config names to include")
    p.add_argument("--skip", default="",
                   help="comma-separated config names to exclude")
    p.add_argument("--timeout", type=float, default=0.0,
                   help="per-config timeout in seconds (0 = no limit)")
    p.add_argument("--python", default=sys.executable,
                   help="interpreter used to run each config (default: this one)")
    p.add_argument("--log-dir", default=os.path.join(HERE, "out", "run_all"),
                   help="where to write per-config logs ('' to disable)")
    p.add_argument("--quiet", action="store_true",
                   help="don't stream each config's output live (still shown in full on failure)")
    args = p.parse_args(argv)

    configs = discover(args.configs)
    if not configs:
        print(f"error: no configs found at {args.configs!r}", file=sys.stderr)
        return 2

    only = {x for x in args.only.split(",") if x}
    skip = {x for x in args.skip.split(",") if x}
    if only:
        configs = [c for c in configs if _name(c) in only]
    if skip:
        configs = [c for c in configs if _name(c) not in skip]
    if not configs:
        print("error: selection left no configs to run", file=sys.stderr)
        return 2

    mode = "Validating" if args.check else "Running"
    print(f"{mode} {len(configs)} config(s) with {args.python}")
    results = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {_name(cfg)} ...", flush=True)
        r = run_one(cfg, args.python, args.check, args.timeout, args.log_dir,
                    stream=not args.quiet)
        tag = r["status"] + (f" — {r['reason']}" if r["status"] != "OK" and r.get("reason") else "")
        print(f"    -> {tag} ({r['elapsed']:.1f}s)")
        results.append(r)

    print_table(results, args.check)
    if args.log_dir:
        print(f"per-config logs: {args.log_dir}")
    return 0 if all(r["status"] == "OK" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
