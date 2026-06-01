#!/usr/bin/env python3
"""Thin launcher so the system can be run from its own directory.

    python run.py configs/kitti_stereo.toml
    python run.py configs/kitti_stereo.toml --check
"""

import os
import sys

# Make the package importable when launched as a loose script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cuvslam_runner.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
