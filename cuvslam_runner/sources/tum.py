"""TUM RGB-D dataset source (Mono-Depth / RGBD mode).

Associates ``rgb.txt`` and ``depth.txt`` by timestamp and yields one
:class:`FrameEvent` per matched pair (single camera + aligned depth). The rig
(one camera) is read from the example ``freiburg3_rig.yaml`` style file when
provided; otherwise define it in ``[rig]``.

    [input]
    type = "tum"
    path = "examples/tum/dataset/rgbd_dataset_freiburg3_long_office_household"
    rig_yaml = "freiburg3_rig.yaml"   # optional, relative to path
    max_time_diff = 0.02              # s, rgb/depth association window
    max_gap = 0.5                     # s, skip large temporal gaps
"""

from __future__ import annotations

import os
from typing import Iterator, List, Optional, Tuple

import yaml

from ..specs import CameraSpec, RigSpec
from ..images import load_depth, load_image
from .base import FrameEvent, FrameSource


def _read_assoc(path: str) -> List[Tuple[float, str]]:
    out: List[Tuple[float, str]] = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) == 2:
                out.append((float(parts[0]), parts[1]))
    return out


def _match(rgb: List[Tuple[float, str]], depth: List[Tuple[float, str]],
           max_time_diff: float) -> List[Tuple[float, str, str]]:
    i = j = 0
    pairs: List[Tuple[float, str, str]] = []
    while i < len(rgb) and j < len(depth):
        rt, rf = rgb[i]
        dt, df = depth[j]
        if abs(rt - dt) < max_time_diff:
            pairs.append((rt, rf, df))
            i += 1
            j += 1
        elif rt < dt:
            i += 1
        else:
            j += 1
    return pairs


class TumSource(FrameSource):
    def __init__(self, table: dict):
        allowed = {"type", "path", "rig_yaml", "max_time_diff", "max_gap"}
        unknown = set(table) - allowed
        if unknown:
            raise ValueError(f"[input] (tum): unknown key(s) {sorted(unknown)}")
        self.path = str(table["path"])
        if not os.path.isdir(self.path):
            raise ValueError(f"[input].path is not a directory: {self.path!r}")
        self.rig_yaml = table.get("rig_yaml")
        self.max_time_diff = float(table.get("max_time_diff", 0.02))
        self.max_gap = float(table.get("max_gap", 0.5))
        self.num_cameras = 1

        rgb = _read_assoc(os.path.join(self.path, "rgb.txt"))
        depth = _read_assoc(os.path.join(self.path, "depth.txt"))
        if not rgb or not depth:
            raise ValueError(f"Missing/empty rgb.txt or depth.txt in {self.path}")
        self._pairs = self._filter_gaps(_match(rgb, depth, self.max_time_diff))

    def _filter_gaps(self, pairs):
        out = []
        prev = None
        for rt, rf, df in pairs:
            if prev is not None and (rt - prev) > self.max_gap:
                pass  # skip across a large temporal gap
            else:
                out.append((rt, os.path.join(self.path, rf), os.path.join(self.path, df)))
            prev = rt
        return out

    def build_rig_spec(self) -> Optional[RigSpec]:
        if not self.rig_yaml:
            return None
        ypath = self.rig_yaml if os.path.isabs(self.rig_yaml) else os.path.join(self.path, self.rig_yaml)
        if not os.path.exists(ypath):
            raise FileNotFoundError(f"TUM rig yaml not found: {ypath}")
        with open(ypath) as handle:
            cfg = yaml.safe_load(handle)
        cam = CameraSpec()
        cam.size = [int(cfg["rgb_camera"]["image_width"]), int(cfg["rgb_camera"]["image_height"])]
        cam.principal = [float(v) for v in cfg["rgb_camera"]["principal_point"]]
        cam.focal = [float(v) for v in cfg["rgb_camera"]["focal_length"]]
        return RigSpec(cameras=[cam])

    @property
    def depth_scale(self) -> Optional[float]:
        """Convenience: depth scale from the rig yaml, if available."""
        if not self.rig_yaml:
            return None
        ypath = self.rig_yaml if os.path.isabs(self.rig_yaml) else os.path.join(self.path, self.rig_yaml)
        with open(ypath) as handle:
            cfg = yaml.safe_load(handle)
        return float(cfg["depth_camera"]["scale"])

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self) -> Iterator[FrameEvent]:
        for rt, rgb_path, depth_path in self._pairs:
            yield FrameEvent(
                timestamp_ns=int(rt * 1e9),
                images=[load_image(rgb_path, bgr=True)],
                depths=[load_depth(depth_path)],
            )
