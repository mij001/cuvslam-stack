"""Generic IMU CSV reader shared by file-based sources.

Lets any recorded dataset supply IMU samples for ``odometry_mode = "Inertial"``
without a dataset-specific loader. Configure it with an ``[input.imu]`` table::

    [input.imu]
    path = "imu0/data.csv"
    format = "euroc"            # euroc | generic
    # generic only:
    columns = ["timestamp", "gx", "gy", "gz", "ax", "ay", "az"]
    timestamp_unit = "ns"       # s | ms | us | ns
    delimiter = ","
    skip_header = true
    angular_unit = "rad"        # rad | deg   (gyro)

EuRoC format is the canonical ``timestamp[ns], wx, wy, wz, ax, ay, az`` layout
with a one-line header.
"""

from __future__ import annotations

import csv as _csv
import math
import os
from typing import List

from .base import ImuEvent

_UNIT_TO_NS = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}

# Recognised generic column names -> internal role.
_GYRO = {"gx", "gy", "gz", "wx", "wy", "wz"}
_ACCEL = {"ax", "ay", "az"}


def parse_imu_table(root: str, table: dict) -> List[ImuEvent]:
    allowed = {
        "path", "format", "columns", "timestamp_unit", "delimiter",
        "skip_header", "angular_unit",
    }
    unknown = set(table) - allowed
    if unknown:
        raise ValueError(f"[input.imu]: unknown key(s) {sorted(unknown)}")
    if "path" not in table:
        raise ValueError("[input.imu] requires a 'path' to the IMU CSV")

    path = table["path"]
    if root and not os.path.isabs(path):
        path = os.path.join(root, path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"IMU CSV not found: {path}")

    fmt = str(table.get("format", "euroc"))
    if fmt == "euroc":
        columns = ["timestamp", "gx", "gy", "gz", "ax", "ay", "az"]
        ts_unit = "ns"
        delimiter = ","
        skip_header = True
        angular_unit = "rad"
    elif fmt == "generic":
        columns = table.get("columns")
        if not columns:
            raise ValueError("[input.imu] format='generic' requires 'columns'")
        ts_unit = str(table.get("timestamp_unit", "ns"))
        delimiter = str(table.get("delimiter", ","))
        skip_header = bool(table.get("skip_header", True))
        angular_unit = str(table.get("angular_unit", "rad"))
    else:
        raise ValueError("[input.imu].format must be 'euroc' or 'generic'")

    if ts_unit not in _UNIT_TO_NS:
        raise ValueError(f"[input.imu].timestamp_unit must be one of {list(_UNIT_TO_NS)}")
    if angular_unit not in ("rad", "deg"):
        raise ValueError("[input.imu].angular_unit must be 'rad' or 'deg'")

    idx = {name: i for i, name in enumerate(columns)}
    if "timestamp" not in idx:
        raise ValueError("[input.imu].columns must include 'timestamp'")
    gyro_keys = [k for k in ("gx", "gy", "gz", "wx", "wy", "wz") if k in idx]
    accel_keys = [k for k in ("ax", "ay", "az") if k in idx]
    if len(gyro_keys) != 3 or len(accel_keys) != 3:
        raise ValueError(
            "[input.imu].columns must include 3 gyro (gx/gy/gz or wx/wy/wz) "
            "and 3 accel (ax/ay/az) columns"
        )
    ts_scale = _UNIT_TO_NS[ts_unit]
    gyro_scale = math.pi / 180.0 if angular_unit == "deg" else 1.0

    events: List[ImuEvent] = []
    with open(path, newline="") as handle:
        reader = _csv.reader(handle, delimiter=delimiter)
        rows = iter(reader)
        if skip_header:
            next(rows, None)
        for row in rows:
            if not row or len(row) < len(columns):
                continue
            ts = int(round(float(row[idx["timestamp"]]) * ts_scale))
            gyro = [float(row[idx[k]]) * gyro_scale for k in gyro_keys]
            accel = [float(row[idx[k]]) for k in accel_keys]
            events.append(ImuEvent(ts, accel, gyro))

    events.sort(key=lambda e: e.timestamp_ns)
    return events
