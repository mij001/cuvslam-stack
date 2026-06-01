"""Plain dataclasses describing a cuVSLAM run.

These specs intentionally contain *no* dependency on the ``cuvslam`` module.
Parsing (TOML, EuRoC yaml, EDEX json, ...) produces these specs; the
:mod:`cuvslam_runner.builders` module is the only place that turns them into
live ``cuvslam`` objects. This split keeps configuration and dataset parsing
unit-testable without a compiled cuVSLAM wheel present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# --------------------------------------------------------------------------- #
# Rig description
# --------------------------------------------------------------------------- #

@dataclass
class PoseSpec:
    """Rigid transform: quaternion (x, y, z, w) + translation (x, y, z)."""
    rotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    translation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class DistortionSpec:
    """Lens distortion model + coefficients.

    ``model`` is one of: Pinhole (0), Fisheye (4), Brown (5), Polynomial (8).
    """
    model: str = "Pinhole"
    parameters: List[float] = field(default_factory=list)


@dataclass
class CameraSpec:
    size: List[int] = field(default_factory=lambda: [0, 0])          # [width, height]
    principal: List[float] = field(default_factory=lambda: [0.0, 0.0])  # [cx, cy]
    focal: List[float] = field(default_factory=lambda: [0.0, 0.0])      # [fx, fy]
    rig_from_camera: PoseSpec = field(default_factory=PoseSpec)
    distortion: DistortionSpec = field(default_factory=DistortionSpec)
    border_top: int = 0
    border_bottom: int = 0
    border_left: int = 0
    border_right: int = 0


@dataclass
class ImuSpec:
    rig_from_imu: PoseSpec = field(default_factory=PoseSpec)
    gyroscope_noise_density: float = 0.0
    accelerometer_noise_density: float = 0.0
    gyroscope_random_walk: float = 0.0
    accelerometer_random_walk: float = 0.0
    frequency: float = 200.0


@dataclass
class RigSpec:
    cameras: List[CameraSpec] = field(default_factory=list)
    imu: Optional[ImuSpec] = None


# --------------------------------------------------------------------------- #
# Odometry / SLAM configuration
# --------------------------------------------------------------------------- #

@dataclass
class RGBDSpec:
    depth_scale_factor: float = 1000.0
    depth_camera_id: int = 0
    enable_depth_stereo_tracking: bool = False


@dataclass
class OdometrySpec:
    # Modes (string names, validated/translated to enums in builders.py)
    odometry_mode: str = "Multicamera"      # Multicamera | Inertial | RGBD | Mono
    multicam_mode: str = "Performance"      # Performance | Moderate | Precision
    # Optional knobs: None means "leave the library default untouched".
    use_gpu: Optional[bool] = None
    async_sba: Optional[bool] = None
    use_motion_model: Optional[bool] = None
    use_denoising: Optional[bool] = None
    rectified_stereo_camera: Optional[bool] = None
    # Export flags default True (needed for visualization / SLAM hand-off).
    enable_observations_export: bool = True
    enable_landmarks_export: bool = True
    enable_final_landmarks_export: Optional[bool] = None
    max_frame_delta_s: Optional[float] = None
    rgbd: Optional[RGBDSpec] = None


@dataclass
class LocalizeSpec:
    """Optional: localize inside an existing map before tracking starts."""
    map_path: str = ""
    guess: PoseSpec = field(default_factory=PoseSpec)
    horizontal_search_radius: float = 8.0
    vertical_search_radius: float = 2.0
    horizontal_step: float = 0.5
    vertical_step: float = 0.2
    angular_step_rads: float = 0.03


@dataclass
class SlamSpec:
    enabled: bool = False
    map_cache_path: Optional[str] = None
    use_gpu: Optional[bool] = None
    sync_mode: Optional[bool] = None
    enable_reading_internals: Optional[bool] = None
    planar_constraints: Optional[bool] = None
    gt_align_mode: Optional[bool] = None
    map_cell_size: Optional[float] = None
    max_landmarks_distance: Optional[float] = None
    max_map_size: Optional[int] = None
    throttling_time_ms: Optional[int] = None
    localize: Optional[LocalizeSpec] = None


# --------------------------------------------------------------------------- #
# Run / output behaviour
# --------------------------------------------------------------------------- #

@dataclass
class RunSpec:
    verbosity: int = 0           # cuvslam.set_verbosity: 0 silent .. 3 info
    warm_up_gpu: bool = False
    start_index: int = 0         # skip the first N frame events
    max_frames: int = 0          # 0 = unlimited
    sleep_ms: float = 0.0        # pause between frames (helps async SLAM catch up)


@dataclass
class OutputSpec:
    trajectory: str = ""             # path to TUM-format trajectory file ("" = none)
    pose_source: str = "auto"        # auto | odometry | slam
    timestamp_unit: str = "s"        # unit for the trajectory timestamp column
    save_map: str = ""               # folder to save SLAM map ("" = none)
    visualize: bool = False          # enable rerun visualization if installed
    print_every: int = 50            # progress print cadence (0 = silent)


@dataclass
class EvalSpec:
    """Optional post-run evaluation against ground truth."""
    enabled: bool = False
    ground_truth: str = ""           # path to GT file
    gt_format: str = "euroc"         # euroc | tum | kitti
    gt_time_unit: str = "s"          # for tum GT
    gt_fps: float = 10.0             # for kitti GT (index timing)
    align: str = "auto"             # auto | se3 | sim3 | none  (auto: sim3 for Mono)
    apply_gt_extrinsic: str = "auto"  # auto | euroc_cam0 | none
    max_time_diff: float = 0.02      # s, association window
    rpe_distances: Optional[List[float]] = None  # metres; None = auto, "kitti" handled in parser
    report: str = ""                 # optional path to write the report text


@dataclass
class Config:
    run: RunSpec
    input: dict                      # raw [input] table (must contain "type")
    odometry: OdometrySpec
    slam: SlamSpec
    output: OutputSpec
    rig: Optional[RigSpec] = None    # explicit rig; if None the source provides it
    eval: EvalSpec = field(default_factory=EvalSpec)
