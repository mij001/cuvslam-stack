"""Smoke tests that exercise config parsing + source wiring without cuvslam.

Run with:  python -m pytest tests/  (from the cuvslam_runner directory)
or simply: python tests/test_smoke.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cuvslam_runner.config import load_config
from cuvslam_runner.sources import build_source
from cuvslam_runner.sources.base import FrameEvent
from cuvslam_runner.images import load_image, load_depth


def _make_dataset(root: str, n: int = 4) -> None:
    for cam in ("image_0", "image_1"):
        os.makedirs(os.path.join(root, cam), exist_ok=True)
        for i in range(n):
            arr = (np.random.rand(8, 12) * 255).astype(np.uint8)
            Image.fromarray(arr, mode="L").save(os.path.join(root, cam, f"{i:06d}.png"))
    with open(os.path.join(root, "times.txt"), "w") as handle:
        for i in range(n):
            handle.write(f"{i * 0.1:.6f}\n")


def _write_config(path: str, root: str) -> None:
    with open(path, "w") as handle:
        handle.write(f"""
[run]
verbosity = 0

[input]
type = "image_folder"
root = "{root}"
  [[input.cameras]]
  images = "image_0/*.png"
  [[input.cameras]]
  images = "image_1/*.png"
  [input.timestamps]
  mode = "file"
  path = "times.txt"
  unit = "s"

[odometry]
odometry_mode = "Multicamera"
multicam_mode = "Performance"
rectified_stereo_camera = true

[[rig.cameras]]
size = [12, 8]
focal = [10.0, 10.0]
principal = [6.0, 4.0]

[[rig.cameras]]
size = [12, 8]
focal = [10.0, 10.0]
principal = [6.0, 4.0]
  [rig.cameras.rig_from_camera]
  translation = [0.5, 0.0, 0.0]

[output]
trajectory = "out/traj.txt"
""")


def test_config_and_source():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "seq")
        _make_dataset(root, n=4)
        cfg_path = os.path.join(tmp, "cfg.toml")
        _write_config(cfg_path, root)

        config = load_config(cfg_path)
        assert config.input["type"] == "image_folder"
        assert config.odometry.odometry_mode == "Multicamera"
        assert config.rig is not None and len(config.rig.cameras) == 2

        source = build_source(config.input)
        assert source.num_cameras == 2
        assert len(source) == 4

        events = list(source)
        assert len(events) == 4
        assert all(isinstance(e, FrameEvent) for e in events)
        # timestamps strictly increasing, in ns, from the 0.1s steps
        ts = [e.timestamp_ns for e in events]
        assert ts == sorted(ts) and ts[1] - ts[0] == 100_000_000
        # two images per frame, correct dtype/shape
        for e in events:
            assert len(e.images) == 2
            for img in e.images:
                assert img.dtype == np.uint8 and img.shape == (8, 12)
    print("test_config_and_source: OK")


def test_timestamp_modes():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "seq")
        _make_dataset(root, n=3)
        # fps mode via direct dict
        src = build_source({
            "type": "image_folder",
            "root": root,
            "cameras": [{"images": "image_0/*.png"}],
            "timestamps": {"mode": "fps", "fps": 10.0},
        })
        ts = [e.timestamp_ns for e in src]
        assert ts == [0, 100_000_000, 200_000_000]
    print("test_timestamp_modes: OK")


def test_unknown_key_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = os.path.join(tmp, "bad.toml")
        with open(cfg_path, "w") as handle:
            handle.write('[input]\ntype = "image_folder"\n\n[odometry]\nbogus_key = 1\n')
        try:
            load_config(cfg_path)
        except Exception as exc:  # noqa: BLE001
            assert "bogus_key" in str(exc)
            print("test_unknown_key_rejected: OK")
            return
        raise AssertionError("expected ConfigError for unknown key")


def test_generic_imu_merge():
    """image_folder + a generic IMU CSV yields interleaved, time-sorted events."""
    from cuvslam_runner.sources.base import FrameEvent, ImuEvent

    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "seq")
        _make_dataset(root, n=3)  # frames at index ts 0,1,2 (default index mode)
        imu_csv = os.path.join(root, "imu.csv")
        with open(imu_csv, "w") as handle:
            handle.write("t,gx,gy,gz,ax,ay,az\n")
            # two IMU samples straddling each frame timestamp
            for t in (0, 1, 1, 2):
                handle.write(f"{t},0.01,0.02,0.03,0.0,0.0,9.81\n")

        src = build_source({
            "type": "image_folder",
            "root": root,
            "cameras": [{"images": "image_0/*.png"}],
            "imu": {
                "path": "imu.csv",
                "format": "generic",
                "columns": ["timestamp", "gx", "gy", "gz", "ax", "ay", "az"],
                "timestamp_unit": "ns",
            },
        })
        assert src.has_imu
        events = list(src)
        kinds = [("imu" if isinstance(e, ImuEvent) else "frame") for e in events]
        ts = [e.timestamp_ns for e in events]
        assert ts == sorted(ts), ts                       # strictly non-decreasing
        assert kinds.count("imu") == 4 and kinds.count("frame") == 3
        # On a tie, IMU precedes the frame so it is integrated up to that frame.
        first_frame = kinds.index("frame")
        assert kinds[0] == "imu" and first_frame >= 1
        imu0 = next(e for e in events if isinstance(e, ImuEvent))
        assert imu0.angular_velocities == [0.01, 0.02, 0.03]
        assert imu0.linear_accelerations == [0.0, 0.0, 9.81]
    print("test_generic_imu_merge: OK")


def test_video_source_optional():
    """Video source plumbing (skipped if opencv is unavailable)."""
    try:
        import cv2
    except ModuleNotFoundError:
        print("test_video_source_optional: SKIPPED (no opencv)")
        return
    from cuvslam_runner.sources.base import FrameEvent

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "clip.avi")
        H, W, N, FPS = 32, 64, 8, 20
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"FFV1"), FPS, (W, H))
        for i in range(N):
            vw.write(np.full((H, W, 3), (i * 20) % 255, np.uint8))
        vw.release()

        src = build_source({
            "type": "video",
            "cameras": [{"source": path, "split": "sbs", "grayscale": True}],
            "timing": {"mode": "fps", "fps": FPS},
        })
        assert src.num_cameras == 2
        events = list(src)
        assert events and isinstance(events[0], FrameEvent)
        assert len(events[0].images) == 2
        assert events[0].images[0].shape == (H, W // 2)
        ts = [e.timestamp_ns for e in events]
        assert ts == sorted(ts) and ts[1] - ts[0] == int(1e9 / FPS)
    print("test_video_source_optional: OK")


def test_image_helpers():
    with tempfile.TemporaryDirectory() as tmp:
        # RGB -> BGR ordering
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[..., 0] = 10  # R
        rgb[..., 2] = 30  # B
        p = os.path.join(tmp, "c.png")
        Image.fromarray(rgb, "RGB").save(p)
        bgr = load_image(p, bgr=True)
        assert bgr.shape == (4, 4, 3) and bgr[0, 0, 0] == 30 and bgr[0, 0, 2] == 10

        # depth as uint16
        d = (np.arange(16).reshape(4, 4) * 100).astype(np.uint16)
        dp = os.path.join(tmp, "d.png")
        Image.fromarray(d).save(dp)
        depth = load_depth(dp)
        assert depth.dtype == np.uint16 and depth.shape == (4, 4)
    print("test_image_helpers: OK")


def test_eval_metrics():
    """ATE/RPE math: exact under rigid/similarity transforms, sane under drift."""
    from scipy.spatial.transform import Rotation as Rot

    from cuvslam_runner import eval as ev

    N = 200
    ts = np.arange(N) * int(1e9 / 20)
    gt = np.tile(np.eye(4), (N, 1, 1))
    for i in range(N):
        a = i * 0.02
        gt[i, :3, :3] = Rot.from_euler("y", a).as_matrix()
        gt[i, :3, 3] = [5 * np.sin(a), 0.0, 5 * (1 - np.cos(a))]
    gt_t = ev.Trajectory(ts, gt)

    # (1) estimate = a rigid transform of GT -> ATE ~ 0 after se3 alignment
    S = np.eye(4)
    S[:3, :3] = Rot.from_euler("xyz", [0.3, -0.2, 1.1]).as_matrix()
    S[:3, 3] = [3.0, -1.0, 2.0]
    est = np.einsum("ij,njk->nik", S, gt)
    r1 = ev.evaluate(ev.Trajectory(ts.copy(), est), gt_t, align="se3", rpe_distances=[1, 2, 4])
    assert r1.ate["rmse"] < 1e-6, r1.ate["rmse"]
    assert r1.rpe["avg_trans_pct"] < 1e-6, r1.rpe["avg_trans_pct"]

    # (2) global scale -> sim3 recovers it and ATE ~ 0
    est2 = est.copy()
    est2[:, :3, 3] *= 2.5
    r2 = ev.evaluate(ev.Trajectory(ts.copy(), est2), gt_t, align="sim3", rpe_distances=[1, 2, 4])
    assert abs(r2.scale - 1 / 2.5) < 1e-3 and r2.ate["rmse"] < 1e-6

    # (3) known monotonic drift -> positive, finite ATE/RPE
    est3 = gt.copy()
    est3[:, 0, 3] += np.linspace(0, 1.0, N)
    r3 = ev.evaluate(ev.Trajectory(ts.copy(), est3), gt_t, align="none", rpe_distances=[1, 2, 4])
    assert r3.ate["rmse"] > 0 and r3.rpe["avg_trans_pct"] > 0
    print("test_eval_metrics: OK")


if __name__ == "__main__":
    test_config_and_source()
    test_timestamp_modes()
    test_unknown_key_rejected()
    test_generic_imu_merge()
    test_video_source_optional()
    test_image_helpers()
    test_eval_metrics()
    print("\nAll smoke tests passed.")
