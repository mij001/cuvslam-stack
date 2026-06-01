"""cuvslam_runner: drive cuVSLAM entirely from a single TOML file.

Public API::

    from cuvslam_runner import load_config, run
    run(load_config("configs/kitti_stereo.toml"))
"""

from __future__ import annotations

from .config import ConfigError, load_config

__all__ = ["load_config", "ConfigError", "run"]


def run(config):
    """Run a configuration (lazy import so config parsing never needs cuvslam)."""
    from .runner import run as _run

    return _run(config)
