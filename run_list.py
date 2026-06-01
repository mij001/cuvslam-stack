#!/usr/bin/env python3
"""Run (or validate) every cuVSLAM config listed in a text file.

The list file has one TOML path per line; blank lines and ``#`` comments are
ignored. Relative paths are resolved against the list file's directory first,
then the current directory. Each config runs in its own subprocess (reusing
``run_all.py``), and a summary table is printed at the end.

    python run_list.py runlist.txt                      # run each listed config
    python run_list.py runlist.txt --check              # validate only (no cuvslam)
    python run_list.py runlist.txt --python ./cuvslam_venv/bin/python
    python run_list.py runlist.txt --timeout 600

By default each config is launched with ./cuvslam_venv/bin/python if that venv
exists (created by setup_env.sh), otherwise with the interpreter running this
script. Exit code is 0 only if every listed config succeeded.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_all  # noqa: E402  (reuse run_one / print_table / _name)


def read_list(path: str) -> list:
    """Parse the list file into resolved config paths."""
    base = os.path.dirname(os.path.abspath(path))
    configs = []
    with open(path) as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if os.path.isabs(line):
                configs.append(line)
                continue
            candidates = [os.path.join(os.getcwd(), line), os.path.join(base, line)]
            configs.append(next((c for c in candidates if os.path.exists(c)), line))
    return configs


def default_python() -> str:
    venv_py = os.path.join(HERE, "cuvslam_venv", "bin", "python")
    return venv_py if os.path.exists(venv_py) else sys.executable


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("list_file", help="text file with one TOML config path per line")
    p.add_argument("--check", action="store_true",
                   help="validate only (no cuvslam, no tracking)")
    p.add_argument("--python", default="", help="interpreter used to run each config")
    p.add_argument("--timeout", type=float, default=0.0,
                   help="per-config timeout in seconds (0 = no limit)")
    p.add_argument("--log-dir", default=os.path.join(HERE, "out", "run_list"),
                   help="where to write per-config logs ('' to disable)")
    p.add_argument("--quiet", action="store_true",
                   help="don't stream each config's output live (still shown in full on failure)")
    args = p.parse_args(argv)

    if not os.path.exists(args.list_file):
        print(f"error: list file not found: {args.list_file}", file=sys.stderr)
        return 2
    configs = read_list(args.list_file)
    if not configs:
        print(f"error: no configs listed in {args.list_file}", file=sys.stderr)
        return 2

    python = args.python or default_python()
    mode = "Validating" if args.check else "Running"
    print(f"{mode} {len(configs)} config(s) from {args.list_file}")
    print(f"interpreter: {python}")

    results = []
    for i, cfg in enumerate(configs, 1):
        name = run_all._name(cfg)
        print(f"\n[{i}/{len(configs)}] {name} ...", flush=True)
        if not os.path.exists(cfg):
            print(f"    -> FAIL — config file not found: {cfg}")
            results.append({"config": cfg, "name": name, "status": "FAIL",
                            "returncode": None, "elapsed": 0.0, "summary": {},
                            "reason": "config file not found"})
            continue
        r = run_all.run_one(cfg, python, args.check, args.timeout, args.log_dir,
                            stream=not args.quiet)
        tag = r["status"] + (f" — {r['reason']}" if r["status"] != "OK" and r.get("reason") else "")
        print(f"    -> {tag} ({r['elapsed']:.1f}s)")
        results.append(r)

    run_all.print_table(results, args.check)
    if args.log_dir:
        print(f"per-config logs: {args.log_dir}")
    return 0 if all(r["status"] == "OK" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
