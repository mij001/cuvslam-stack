"""TUM-format trajectory writer.

Each line is ``timestamp tx ty tz qx qy qz qw`` — the format consumed by the
TUM/EuRoC evaluation tools and by this project's own SLAM localization example.
"""

from __future__ import annotations

import os
from typing import List, Tuple

_UNIT_DIV = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0}


class TrajectoryWriter:
    def __init__(self, path: str, timestamp_unit: str = "s"):
        self.path = path
        self.div = _UNIT_DIV[timestamp_unit]
        self._rows: List[Tuple[float, list, list]] = []

    def add(self, timestamp_ns: int, translation, rotation) -> None:
        self._rows.append((timestamp_ns / self.div, list(translation), list(rotation)))

    def __len__(self) -> int:
        return len(self._rows)

    def save(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        with open(self.path, "w") as handle:
            for ts, t, q in self._rows:
                handle.write(
                    f"{ts:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
                )
