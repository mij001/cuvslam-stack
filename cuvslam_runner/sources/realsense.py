"""Live Intel RealSense stereo source (best-effort).

Streams the two infrared cameras of a RealSense device and builds the rig from
the device's own intrinsics/extrinsics. Requires ``pyrealsense2`` and a
connected camera, so it cannot be exercised without hardware — it follows the
``examples/realsense/run_stereo.py`` pattern.

    [input]
    type = "realsense"
    width = 640
    height = 360
    fps = 30
    warmup_frames = 60
    disable_emitter = true
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from ..specs import CameraSpec, DistortionSpec, PoseSpec, RigSpec
from .base import FrameEvent, FrameSource


class RealsenseSource(FrameSource):
    def __init__(self, table: dict):
        allowed = {"type", "width", "height", "fps", "warmup_frames", "disable_emitter"}
        unknown = set(table) - allowed
        if unknown:
            raise ValueError(f"[input] (realsense): unknown key(s) {sorted(unknown)}")
        try:
            import pyrealsense2 as rs  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "type='realsense' needs pyrealsense2 installed and a connected camera."
            ) from exc
        self._rs = rs
        self.width = int(table.get("width", 640))
        self.height = int(table.get("height", 360))
        self.fps = int(table.get("fps", 30))
        self.warmup_frames = int(table.get("warmup_frames", 60))
        self.disable_emitter = bool(table.get("disable_emitter", True))
        self.num_cameras = 2
        self._pipeline = None
        self._cached_params = None

    def _configure(self):
        rs = self._rs
        config = rs.config()
        pipeline = rs.pipeline()
        config.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, self.fps)
        config.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, self.fps)
        return pipeline, config

    def build_rig_spec(self) -> RigSpec:
        rs = self._rs
        pipeline, config = self._configure()
        pipeline.start(config)
        frames = pipeline.wait_for_frames()
        left = frames[0].profile.as_video_stream_profile()
        right = frames[1].profile.as_video_stream_profile()
        li, ri = left.intrinsics, right.intrinsics
        extr = right.get_extrinsics_to(left)
        pipeline.stop()

        def cam(intr) -> CameraSpec:
            c = CameraSpec()
            c.size = [intr.width, intr.height]
            c.focal = [intr.fx, intr.fy]
            c.principal = [intr.ppx, intr.ppy]
            c.distortion = DistortionSpec("Pinhole", [])
            return c

        left_cam = cam(li)
        right_cam = cam(ri)
        right_cam.rig_from_camera = PoseSpec(
            rotation=[0.0, 0.0, 0.0, 1.0],
            translation=[float(v) for v in extr.translation],
        )
        return RigSpec(cameras=[left_cam, right_cam])

    def __iter__(self) -> Iterator[FrameEvent]:
        rs = self._rs
        pipeline, config = self._configure()
        profile = pipeline.start(config)
        if self.disable_emitter:
            sensor = profile.get_device().query_sensors()[0]
            if sensor.supports(rs.option.emitter_enabled):
                sensor.set_option(rs.option.emitter_enabled, 0)
        self._pipeline = pipeline
        frame_id = 0
        try:
            while True:
                frames = pipeline.wait_for_frames()
                lf = frames.get_infrared_frame(1)
                rf = frames.get_infrared_frame(2)
                if not lf or not rf:
                    continue
                frame_id += 1
                if frame_id <= self.warmup_frames:
                    continue
                ts = int(lf.timestamp * 1e6)  # ms -> ns
                images = [
                    np.ascontiguousarray(np.asanyarray(lf.get_data())),
                    np.ascontiguousarray(np.asanyarray(rf.get_data())),
                ]
                yield FrameEvent(timestamp_ns=ts, images=images, meta={"frame": frame_id})
        finally:
            self.close()

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None
