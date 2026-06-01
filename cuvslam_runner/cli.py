"""Command-line entry point: ``python -m cuvslam_runner <config.toml>``."""

from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_config
from .sources import available_types


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="cuvslam_runner",
        description="Run cuVSLAM from a single TOML configuration file.",
    )
    parser.add_argument("config", help="path to the TOML configuration file")
    parser.add_argument(
        "--check", action="store_true",
        help="parse and validate the config (and enumerate frames) without importing cuvslam",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.check:
        return _check(config)

    from .runner import run
    run(config)
    return 0


def _check(config) -> int:
    """Dry run: validate config + source wiring without touching cuvslam."""
    from .sources import build_source

    print(f"input.type      = {config.input['type']}  (available: {available_types()})")
    print(f"odometry_mode   = {config.odometry.odometry_mode}")
    print(f"multicam_mode   = {config.odometry.multicam_mode}")
    print(f"slam.enabled    = {config.slam.enabled}")
    print(f"output.trajectory = {config.output.trajectory or '(none)'}")
    if config.eval.enabled:
        print(f"eval.ground_truth = {config.eval.ground_truth} "
              f"(format={config.eval.gt_format}, align={config.eval.align})")

    try:
        source = build_source(config.input)
    except Exception as exc:  # noqa: BLE001 - report any wiring error to the user
        print(f"source error: {exc}", file=sys.stderr)
        return 1

    rig_spec = config.rig or source.build_rig_spec()
    n_rig = len(rig_spec.cameras) if rig_spec else 0
    print(f"source cameras  = {source.num_cameras}")
    print(f"rig cameras     = {n_rig} ({'explicit TOML' if config.rig else 'from source'})")
    if rig_spec and rig_spec.imu is not None:
        print("rig imu         = yes")
    try:
        print(f"frame count     = {len(source)}")
    except TypeError:
        print("frame count     = (live / unbounded)")
    print("OK: configuration is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
