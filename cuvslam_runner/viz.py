"""Optional, generic Rerun visualization.

Kept deliberately minimal and dataset-agnostic: it logs the trajectory, the
primary camera image, and that camera's 2D observations. If ``rerun`` is not
installed, :func:`make_visualizer` returns a no-op object so the runner works
headless without extra branches.
"""

from __future__ import annotations

from typing import List, Optional


class _NullVisualizer:
    enabled = False

    def log_frame(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


class _RerunVisualizer:
    enabled = True

    def __init__(self, name: str = "cuvslam_runner"):
        import rerun as rr
        import rerun.blueprint as rrb

        self.rr = rr
        rr.init(name, spawn=True)
        rr.send_blueprint(rrb.Blueprint(
            rrb.TimePanel(state="collapsed"),
            rrb.Horizontal(
                column_shares=[0.5, 0.5],
                contents=[rrb.Spatial3DView(origin="world"),
                          rrb.Spatial2DView(origin="world/cam0")],
            ),
        ))
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        self._trajectory: List[list] = []

    @staticmethod
    def _color(identifier: int):
        return [(identifier * 17) % 256, (identifier * 31) % 256, (identifier * 47) % 256]

    def log_frame(self, frame_id, timestamp_ns, pose, image=None, observations=None):
        rr = self.rr
        rr.set_time_nanos("timestamp", int(timestamp_ns))
        if pose is not None:
            self._trajectory.append(list(pose.translation))
            rr.log("world/trajectory", rr.LineStrips3D([self._trajectory]))
            rr.log("world/cam0", rr.Transform3D(
                translation=pose.translation, quaternion=pose.rotation))
        if image is not None and image.size:
            rr.log("world/cam0/image", rr.Image(image).compress(jpeg_quality=80))
        if observations:
            uv = [[o.u, o.v] for o in observations]
            colors = [self._color(o.id) for o in observations]
            rr.log("world/cam0/observations", rr.Points2D(uv, radii=4, colors=colors))

    def close(self) -> None:
        pass


def make_visualizer(enabled: bool):
    if not enabled:
        return _NullVisualizer()
    try:
        return _RerunVisualizer()
    except Exception as exc:  # pragma: no cover - rerun missing/broken
        print(f"[viz] visualization disabled ({exc}); install rerun-sdk to enable.")
        return _NullVisualizer()
