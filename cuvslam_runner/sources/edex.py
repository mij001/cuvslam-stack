"""EDEX multi-camera source (TartanGround / R2B Galileo and similar).

EDEX bundles a JSON calibration file describing N cameras (intrinsics +
extrinsics, in an OpenGL-style frame) together with the image folders. This
loader builds the rig from the ``.edex`` file and yields one synchronized
multi-camera :class:`FrameEvent` per frame index.

Two frame layouts are supported:

  layout = "folders"  (TartanGround):
      images live in per-camera folders named ``image_<name>`` with files
      ``<frame:06d>_<name>.png``.

  layout = "jsonl"    (R2B Galileo edex extractor):
      a ``frame_metadata.jsonl`` lists, per frame, the relative file paths.

    [input]
    type = "edex"
    edex = "examples/multicamera_edex/tartan_ground.edex"
    layout = "folders"
    data_root = "dataset/tartan_ground/OldTownFall/Data_anymal/P2000"
    camera_names = ["lcam_front", "rcam_front", ...]   # folders layout only
"""

from __future__ import annotations

import json
import os
from typing import Iterator, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from ..specs import CameraSpec, DistortionSpec, PoseSpec, RigSpec
from ..images import load_image
from .base import FrameEvent, FrameSource

_MODEL_MAP = {
    "pinhole": "Pinhole",
    "fisheye": "Fisheye",
    "brown": "Brown",
    "polynomial": "Polynomial",
}

# OpenGL (EDEX) -> OpenCV basis change.
_K = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)


def _pose_from_edex(transform_16: List[float]) -> PoseSpec:
    mat = np.array(transform_16, dtype=float).reshape(-1, 4)
    rot = _K @ mat[:3, :3] @ _K.T
    trans = _K @ mat[:3, 3]
    quat = Rotation.from_matrix(rot).as_quat()
    return PoseSpec(rotation=[float(v) for v in quat], translation=[float(v) for v in trans])


class EdexSource(FrameSource):
    def __init__(self, table: dict):
        allowed = {"type", "edex", "layout", "data_root", "camera_names", "image_pattern"}
        unknown = set(table) - allowed
        if unknown:
            raise ValueError(f"[input] (edex): unknown key(s) {sorted(unknown)}")
        self.edex_path = str(table["edex"])
        if not os.path.exists(self.edex_path):
            raise FileNotFoundError(f"EDEX file not found: {self.edex_path}")
        self.layout = str(table.get("layout", "folders"))
        self.data_root = str(table.get("data_root", os.path.dirname(self.edex_path)))
        self.camera_names = table.get("camera_names")
        self.image_pattern = str(table.get("image_pattern", "{frame:06d}_{name}.png"))

        with open(self.edex_path) as handle:
            self._edex = json.load(handle)
        self._cam_records = self._edex[0]["cameras"]
        self.num_cameras = len(self._cam_records)

        if self.layout == "folders":
            if not self.camera_names:
                raise ValueError("edex layout='folders' requires 'camera_names'")
            if len(self.camera_names) != self.num_cameras:
                raise ValueError(
                    f"camera_names has {len(self.camera_names)} entries but "
                    f"edex describes {self.num_cameras} cameras"
                )
            first = os.path.join(self.data_root, f"image_{self.camera_names[0]}")
            self.frame_count = len(os.listdir(first))
            self._jsonl: Optional[List[dict]] = None
        elif self.layout == "jsonl":
            meta_path = os.path.join(self.data_root, "frame_metadata.jsonl")
            with open(meta_path) as handle:
                self._jsonl = [json.loads(line) for line in handle if line.strip()]
            self.frame_count = len(self._jsonl)
        else:
            raise ValueError("edex layout must be 'folders' or 'jsonl'")

    def build_rig_spec(self) -> RigSpec:
        cameras = []
        for rec in self._cam_records:
            intr = rec["intrinsics"]
            cam = CameraSpec()
            model = _MODEL_MAP.get(str(intr["distortion_model"]).lower())
            if model is None:
                raise ValueError(f"Unknown EDEX distortion model: {intr['distortion_model']}")
            cam.distortion = DistortionSpec(model, [float(v) for v in intr["distortion_params"]])
            cam.focal = [float(v) for v in intr["focal"]]
            cam.principal = [float(v) for v in intr["principal"]]
            cam.size = [int(v) for v in intr["size"]]
            cam.rig_from_camera = _pose_from_edex(rec["transform"])
            cameras.append(cam)
        return RigSpec(cameras=cameras)

    def __len__(self) -> int:
        return self.frame_count

    def __iter__(self) -> Iterator[FrameEvent]:
        for frame in range(self.frame_count):
            if self.layout == "folders":
                paths = [
                    os.path.join(self.data_root, f"image_{name}",
                                 self.image_pattern.format(frame=frame, name=name))
                    for name in self.camera_names
                ]
            else:
                entry = self._jsonl[frame]
                paths = [os.path.join(self.data_root, c["filename"]) for c in entry["cams"]]

            try:
                images = [load_image(p, bgr=False) for p in paths]
            except FileNotFoundError as exc:
                print(f"[edex] missing image for frame {frame}: {exc}")
                continue
            yield FrameEvent(timestamp_ns=frame, images=images, meta={"frame": frame})
