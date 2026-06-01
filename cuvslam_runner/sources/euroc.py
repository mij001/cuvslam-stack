"""EuRoC MAV (ASL) dataset source.

Reads the standard ``mav0`` layout (``cam0``/``cam1``/``imu0`` with ``sensor*.yaml``
and ``data.csv``) and builds a stereo (+ IMU) rig from the calibration files.
Use ``odometry_mode = "Inertial"`` to interleave IMU samples, or any of
``Multicamera`` / ``Mono`` for vision-only tracking.

    [input]
    type = "euroc"
    path = "examples/euroc/dataset/mav0"
    use_imu = true        # optional; defaults to true (set false for vision-only)
"""

from __future__ import annotations

import csv
import os
from typing import Iterator, List, Optional, Tuple

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from ..specs import CameraSpec, DistortionSpec, ImuSpec, PoseSpec, RigSpec
from ..images import load_image
from .base import FrameEvent, FrameSource, ImuEvent


def _load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"EuRoC sensor yaml not found: {path}")
    with open(path) as handle:
        return yaml.safe_load(handle)


def _transform_key(cfg: dict) -> str:
    for key in cfg:
        if key.startswith("T_"):
            return key
    raise ValueError("No T_* transform found in EuRoC sensor config")


def _matrix(cfg: dict) -> np.ndarray:
    return np.array(cfg[_transform_key(cfg)]["data"]).reshape(4, 4)


def _to_cam0(cam0_t: np.ndarray, sensor_t: np.ndarray) -> np.ndarray:
    return np.linalg.inv(cam0_t) @ sensor_t


def _pose_from_matrix(mat: np.ndarray) -> PoseSpec:
    quat = Rotation.from_matrix(mat[:3, :3]).as_quat()  # x, y, z, w
    return PoseSpec(rotation=[float(v) for v in quat],
                    translation=[float(v) for v in mat[:3, 3]])


class EurocSource(FrameSource):
    def __init__(self, table: dict):
        allowed = {"type", "path", "use_imu"}
        unknown = set(table) - allowed
        if unknown:
            raise ValueError(f"[input] (euroc): unknown key(s) {sorted(unknown)}")
        self.path = str(table["path"]) if "path" in table else None
        if not self.path or not os.path.isdir(self.path):
            raise ValueError(f"[input].path must point to an existing mav0 folder, got {self.path!r}")
        self.use_imu = bool(table.get("use_imu", True))
        self.num_cameras = 2
        self._cam0, self._cam1, self._imu, self._is_default = self._sensor_paths()

    def _sensor_paths(self) -> Tuple[str, str, str, bool]:
        recal = os.path.join(self.path, "cam0", "sensor_recalibrated.yaml")
        if os.path.exists(recal):
            return (
                recal,
                os.path.join(self.path, "cam1", "sensor_recalibrated.yaml"),
                os.path.join(self.path, "imu0", "sensor_recalibrated.yaml"),
                False,
            )
        return (
            os.path.join(self.path, "cam0", "sensor.yaml"),
            os.path.join(self.path, "cam1", "sensor.yaml"),
            os.path.join(self.path, "imu0", "sensor.yaml"),
            True,
        )

    # ----- rig ------------------------------------------------------------ #
    def _camera_spec(self, cfg: dict) -> CameraSpec:
        cam = CameraSpec()
        cam.focal = [float(v) for v in cfg["intrinsics"][0:2]]
        cam.principal = [float(v) for v in cfg["intrinsics"][2:4]]
        cam.size = [int(v) for v in cfg["resolution"]]
        coeffs = list(cfg["distortion_coefficients"])
        if self._is_default:
            # default EuRoC ships radtan k1,k2,p1,p2 -> Brown needs k1,k2,k3,p1,p2
            cam.distortion = DistortionSpec("Brown", [coeffs[0], coeffs[1], 0.0, coeffs[2], coeffs[3]])
        else:
            cam.distortion = DistortionSpec("Fisheye", [float(v) for v in coeffs])
        return cam

    def build_rig_spec(self) -> RigSpec:
        c0, c1, imu = _load_yaml(self._cam0), _load_yaml(self._cam1), _load_yaml(self._imu)
        m0, m1, mi = _matrix(c0), _matrix(c1), _matrix(imu)

        cam0 = self._camera_spec(c0)
        cam0.rig_from_camera = PoseSpec()  # cam0 is rig origin

        cam1 = self._camera_spec(c1)
        cam1_t = _to_cam0(m0, m1) if self._is_default else m1
        cam1.rig_from_camera = _pose_from_matrix(cam1_t)

        imu_spec = None
        if self.use_imu:
            imu_t = _to_cam0(m0, mi) if self._is_default else mi
            imu_spec = ImuSpec(
                rig_from_imu=_pose_from_matrix(imu_t),
                gyroscope_noise_density=float(imu["gyroscope_noise_density"]),
                accelerometer_noise_density=float(imu["accelerometer_noise_density"]),
                gyroscope_random_walk=float(imu["gyroscope_random_walk"]),
                accelerometer_random_walk=float(imu["accelerometer_random_walk"]),
                frequency=float(imu["rate_hz"]),
            )
        return RigSpec(cameras=[cam0, cam1], imu=imu_spec)

    # ----- events --------------------------------------------------------- #
    def _read_csv(self, path: str, kind: str) -> List[dict]:
        rows: List[dict] = []
        with open(path) as handle:
            next(handle)  # header
            for row in csv.reader(handle):
                if not row:
                    continue
                if kind == "camera":
                    rows.append({"timestamp": int(row[0]), "filename": row[1]})
                else:
                    rows.append({
                        "timestamp": int(row[0]),
                        "gyro": [float(x) for x in row[1:4]],
                        "accel": [float(x) for x in row[4:7]],
                    })
        return rows

    def __iter__(self) -> Iterator:
        left = self._read_csv(os.path.join(self.path, "cam0", "data.csv"), "camera")
        right = self._read_csv(os.path.join(self.path, "cam1", "data.csv"), "camera")
        n = min(len(left), len(right))

        events: List[Tuple[int, str, object]] = []
        for i in range(n):
            events.append((left[i]["timestamp"], "frame", (left[i]["filename"], right[i]["filename"])))
        if self.use_imu:
            imu_csv = os.path.join(self.path, "imu0", "data.csv")
            for m in self._read_csv(imu_csv, "imu"):
                events.append((m["timestamp"], "imu", m))
        events.sort(key=lambda e: e[0])

        cam0_dir = os.path.join(self.path, "cam0", "data")
        cam1_dir = os.path.join(self.path, "cam1", "data")
        for ts, kind, payload in events:
            if kind == "imu":
                yield ImuEvent(ts, payload["accel"], payload["gyro"])
            else:
                lf, rf = payload
                images = [
                    load_image(os.path.join(cam0_dir, lf), bgr=False),
                    load_image(os.path.join(cam1_dir, rf), bgr=False),
                ]
                yield FrameEvent(timestamp_ns=ts, images=images)
