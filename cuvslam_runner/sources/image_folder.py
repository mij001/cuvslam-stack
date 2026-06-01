"""Generic multi-camera image-folder source.

This is the workhorse loader. It can drive any dataset that stores frames as
ordered image files in one folder per camera — KITTI, TartanGround, Oxford
RobotCar (with bayer demosaicing), or any arbitrary collection of your own.

Per-camera streams are aligned by their sorted file order; optional depth and
mask streams are aligned the same way. Timestamps can come from a file, from
the image filenames, from a fixed FPS, or simply be the frame index.

Example ``[input]`` table::

    [input]
    type = "image_folder"
    root = "dataset/sequences/06"          # optional path prefix for all globs

      [[input.cameras]]
      images = "image_0/*.png"

      [[input.cameras]]
      images = "image_1/*.png"

      [input.timestamps]
      mode = "file"                        # file | filename | fps | index
      path = "times.txt"
      unit = "s"                           # s | ms | us | ns

For inertial datasets, add an ``[input.imu]`` table (see csv_imu.py); IMU
samples are merged with the frames in ascending-timestamp order so the source
can drive ``odometry_mode = "Inertial"``.
"""

from __future__ import annotations

import glob
import os
from typing import Iterator, List, Optional

from ..images import empty_image, load_depth, load_image
from .base import FrameEvent, FrameSource, ImuEvent
from .csv_imu import parse_imu_table


def _sorted_glob(pattern: str) -> List[str]:
    return sorted(glob.glob(pattern))


def _resolve(root: str, pattern: str) -> str:
    return pattern if os.path.isabs(pattern) or not root else os.path.join(root, pattern)


class _CameraStream:
    def __init__(self, root: str, table: dict, idx: int):
        allowed = {"images", "depth", "mask", "bgr", "bayer"}
        unknown = set(table) - allowed
        if unknown:
            raise ValueError(f"input.cameras[{idx}]: unknown key(s) {sorted(unknown)}")
        if "images" not in table:
            raise ValueError(f"input.cameras[{idx}] must define an 'images' glob")
        self.images = _sorted_glob(_resolve(root, table["images"]))
        if not self.images:
            raise ValueError(
                f"input.cameras[{idx}]: 'images' glob matched no files: {table['images']!r}"
            )
        self.depth = _sorted_glob(_resolve(root, table["depth"])) if "depth" in table else None
        self.mask = _sorted_glob(_resolve(root, table["mask"])) if "mask" in table else None
        self.bgr = bool(table.get("bgr", True))
        self.bayer = table.get("bayer")

    def __len__(self) -> int:
        return len(self.images)


def _read_timestamp_file(path: str) -> List[float]:
    values: List[float] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # take the first whitespace/comma separated token
            token = line.replace(",", " ").split()[0]
            values.append(float(token))
    return values


def _filename_timestamps(paths: List[str]) -> List[float]:
    out = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        digits = "".join(ch for ch in stem if (ch.isdigit() or ch == "."))
        if not digits:
            raise ValueError(f"Cannot parse a numeric timestamp from filename {p!r}")
        out.append(float(digits))
    return out


_UNIT_TO_NS = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}


class ImageFolderSource(FrameSource):
    def __init__(self, table: dict):
        allowed = {"type", "root", "cameras", "timestamps", "depth_scale_to_uint16", "imu"}
        unknown = set(table) - allowed
        if unknown:
            raise ValueError(f"[input] (image_folder): unknown key(s) {sorted(unknown)}")

        root = str(table.get("root", ""))
        cams_tbl = table.get("cameras")
        if not cams_tbl:
            raise ValueError("[input] (image_folder) requires at least one [[input.cameras]].")
        self.streams = [_CameraStream(root, c, i) for i, c in enumerate(cams_tbl)]
        self.num_cameras = len(self.streams)
        self._depth_scale = float(table.get("depth_scale_to_uint16", 1.0))

        self.frame_count = min(len(s) for s in self.streams)
        for i, s in enumerate(self.streams):
            if len(s) != self.frame_count:
                print(
                    f"[image_folder] warning: camera {i} has {len(s)} frames; "
                    f"truncating all cameras to {self.frame_count}"
                )

        self._timestamps_ns = self._build_timestamps(root, table.get("timestamps"))

        # Optional IMU stream (enables Inertial mode on arbitrary datasets).
        self._imu_events: List[ImuEvent] = []
        if "imu" in table:
            self._imu_events = parse_imu_table(root, table["imu"])
        self.has_imu = bool(self._imu_events)

    # ----- timestamps ----------------------------------------------------- #
    def _build_timestamps(self, root: str, table: Optional[dict]) -> List[int]:
        n = self.frame_count
        if table is None:
            return list(range(n))  # default: index-based

        allowed = {"mode", "path", "unit", "fps"}
        unknown = set(table) - allowed
        if unknown:
            raise ValueError(f"[input.timestamps]: unknown key(s) {sorted(unknown)}")
        mode = str(table.get("mode", "index"))
        unit = str(table.get("unit", "s"))
        if unit not in _UNIT_TO_NS:
            raise ValueError(f"[input.timestamps].unit must be one of {list(_UNIT_TO_NS)}")
        scale = _UNIT_TO_NS[unit]

        if mode == "index":
            return list(range(n))
        if mode == "fps":
            fps = float(table.get("fps", 30.0))
            if fps <= 0:
                raise ValueError("[input.timestamps].fps must be > 0")
            return [int(round(i * 1e9 / fps)) for i in range(n)]
        if mode == "file":
            if "path" not in table:
                raise ValueError("[input.timestamps] mode='file' requires 'path'")
            vals = _read_timestamp_file(_resolve(root, table["path"]))
            if len(vals) < n:
                raise ValueError(
                    f"timestamp file has {len(vals)} entries < {n} frames"
                )
            return [int(round(v * scale)) for v in vals[:n]]
        if mode == "filename":
            vals = _filename_timestamps(self.streams[0].images[:n])
            return [int(round(v * scale)) for v in vals]
        raise ValueError(f"[input.timestamps].mode must be index|fps|file|filename, got {mode!r}")

    # ----- iteration ------------------------------------------------------ #
    def __len__(self) -> int:
        return self.frame_count

    def _load_frame(self, frame: int) -> FrameEvent:
        any_depth = any(s.depth for s in self.streams)
        any_mask = any(s.mask for s in self.streams)

        images = [
            load_image(s.images[frame], bgr=s.bgr, bayer=s.bayer)
            for s in self.streams
        ]

        depths = None
        if any_depth:
            depths = []
            for s in self.streams:
                if s.depth and frame < len(s.depth):
                    depths.append(load_depth(s.depth[frame], scale_to_uint16=self._depth_scale))
                else:
                    depths.append(empty_image().astype("uint16"))

        masks = None
        if any_mask:
            masks = []
            for s in self.streams:
                if s.mask and frame < len(s.mask):
                    masks.append(load_image(s.mask[frame], bgr=False))
                else:
                    masks.append(empty_image())

        return FrameEvent(
            timestamp_ns=self._timestamps_ns[frame],
            images=images,
            depths=depths,
            masks=masks,
            meta={"frame": frame},
        )

    def __iter__(self) -> Iterator:
        if not self._imu_events:
            for frame in range(self.frame_count):
                yield self._load_frame(frame)
            return

        # Merge frame and IMU events by timestamp (IMU first on ties so it is
        # integrated up to the frame). Images are loaded lazily as frames yield.
        schedule = [(self._timestamps_ns[f], 1, f) for f in range(self.frame_count)]
        schedule += [(ev.timestamp_ns, 0, k) for k, ev in enumerate(self._imu_events)]
        schedule.sort(key=lambda item: (item[0], item[1]))
        for _ts, kind, payload in schedule:
            if kind == 0:
                yield self._imu_events[payload]
            else:
                yield self._load_frame(payload)
