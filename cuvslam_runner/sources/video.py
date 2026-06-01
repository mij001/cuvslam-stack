"""Generic OpenCV video / stream source (recorded files OR realtime).

Reads frames with ``cv2.VideoCapture``, so a single source string covers:
  * a webcam / capture device     -> "0", "1", ...     (realtime)
  * an IP camera / network stream  -> "rtsp://...", "http://..."  (realtime)
  * a recorded video file          -> "path/to/clip.mp4"          (recorded)

Multiple ``[[input.cameras]]`` are grabbed in lockstep (one frame each per
step). A single physical stereo stream that packs both views into one frame can
be split with ``split = "sbs"`` (side-by-side) or ``split = "tb"`` (top/bottom),
producing two images from one capture.

Calibration is not available from arbitrary video, so an explicit ``[rig]`` is
required (define one [[rig.cameras]] per produced image).

    [input]
    type = "video"
      [[input.cameras]]
      source = "0"             # webcam, or file path, or rtsp:// url
      split = "sbs"            # none | sbs | tb   (default none)
      grayscale = false        # convert to mono8 (default false -> BGR passthrough)
      [input.timing]
      mode = "auto"            # auto | wallclock | fps | index
      fps = 30                 # used by mode=fps (and auto for files w/o metadata)
"""

from __future__ import annotations

import time
from typing import Iterator, List, Optional

import numpy as np

from .base import FrameEvent, FrameSource


def _looks_like_device(src: str) -> bool:
    return src.isdigit()


def _looks_like_stream(src: str) -> bool:
    return "://" in src  # rtsp://, http://, udp://, ...


class _Capture:
    def __init__(self, cv2, table: dict, idx: int):
        allowed = {"source", "split", "grayscale", "api"}
        unknown = set(table) - allowed
        if unknown:
            raise ValueError(f"input.cameras[{idx}]: unknown key(s) {sorted(unknown)}")
        if "source" not in table:
            raise ValueError(f"input.cameras[{idx}] requires a 'source'")
        self._cv2 = cv2
        self.raw = str(table["source"])
        self.split = str(table.get("split", "none"))
        if self.split not in ("none", "sbs", "tb"):
            raise ValueError("camera.split must be none|sbs|tb")
        self.grayscale = bool(table.get("grayscale", False))
        self.is_live = _looks_like_device(self.raw) or _looks_like_stream(self.raw)

        target = int(self.raw) if _looks_like_device(self.raw) else self.raw
        self.cap = cv2.VideoCapture(target)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video source {self.raw!r}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    @property
    def images_per_step(self) -> int:
        return 2 if self.split in ("sbs", "tb") else 1

    def read(self) -> Optional[List[np.ndarray]]:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        if self.grayscale:
            frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        if self.split == "sbs":
            w = frame.shape[1] // 2
            parts = [frame[:, :w], frame[:, w:2 * w]]
        elif self.split == "tb":
            h = frame.shape[0] // 2
            parts = [frame[:h, :], frame[h:2 * h, :]]
        else:
            parts = [frame]
        return [np.ascontiguousarray(p) for p in parts]

    def release(self) -> None:
        self.cap.release()


class VideoSource(FrameSource):
    def __init__(self, table: dict):
        allowed = {"type", "cameras", "timing"}
        unknown = set(table) - allowed
        if unknown:
            raise ValueError(f"[input] (video): unknown key(s) {sorted(unknown)}")
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "type='video' needs opencv-python (`pip install opencv-python`)."
            ) from exc
        self._cv2 = cv2

        cams = table.get("cameras")
        if not cams:
            raise ValueError("[input] (video) requires at least one [[input.cameras]].")
        self.captures = [_Capture(cv2, c, i) for i, c in enumerate(cams)]
        self.num_cameras = sum(c.images_per_step for c in self.captures)

        timing = table.get("timing", {})
        allowed_t = {"mode", "fps", "max_frames"}
        unknown_t = set(timing) - allowed_t
        if unknown_t:
            raise ValueError(f"[input.timing]: unknown key(s) {sorted(unknown_t)}")
        self.mode = str(timing.get("mode", "auto"))
        if self.mode not in ("auto", "wallclock", "fps", "index"):
            raise ValueError("[input.timing].mode must be auto|wallclock|fps|index")
        self.fps_override = float(timing.get("fps", 0.0))
        self.max_frames = int(timing.get("max_frames", 0))

        self._live = any(c.is_live for c in self.captures)
        # resolve "auto": wallclock for live sources, fps for files
        self._mode = self.mode
        if self._mode == "auto":
            self._mode = "wallclock" if self._live else "fps"
        self._fps = self.fps_override or max((c.fps for c in self.captures), default=0.0) or 30.0

    def __len__(self) -> int:
        if self._live:
            raise TypeError("live video source has no length")
        counts = [c.frame_count for c in self.captures if c.frame_count > 0]
        if not counts:
            raise TypeError("video frame count unavailable")
        n = min(counts)
        return min(n, self.max_frames) if self.max_frames else n

    def __iter__(self) -> Iterator[FrameEvent]:
        step = 0
        try:
            while True:
                per_cap = [c.read() for c in self.captures]
                if any(p is None for p in per_cap):
                    break  # end of file or stream interrupted
                images: List[np.ndarray] = []
                for parts in per_cap:
                    images.extend(parts)

                if self._mode == "wallclock":
                    ts = time.time_ns()
                elif self._mode == "index":
                    ts = step
                else:  # fps
                    ts = int(round(step * 1e9 / self._fps))

                yield FrameEvent(timestamp_ns=ts, images=images, meta={"frame": step})
                step += 1
                if self.max_frames and step >= self.max_frames:
                    break
        finally:
            self.close()

    def close(self) -> None:
        for c in self.captures:
            c.release()
