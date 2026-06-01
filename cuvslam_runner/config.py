"""Load and validate a run configuration from a single TOML file."""

from __future__ import annotations

from typing import Any, Optional

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - exercised on 3.9/3.10
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit(
            "No TOML parser available. Use Python 3.11+ or `pip install tomli`."
        ) from exc

from .specs import (
    CameraSpec,
    Config,
    DistortionSpec,
    EvalSpec,
    ImuSpec,
    LocalizeSpec,
    OdometrySpec,
    OutputSpec,
    PoseSpec,
    RGBDSpec,
    RigSpec,
    RunSpec,
    SlamSpec,
)


class ConfigError(ValueError):
    """Raised when the TOML configuration is malformed."""


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _check_keys(table: dict, allowed: set, where: str) -> None:
    unknown = set(table) - allowed
    if unknown:
        raise ConfigError(
            f"Unknown key(s) {sorted(unknown)} in [{where}]. "
            f"Allowed: {sorted(allowed)}"
        )


def _pose(table: Optional[dict], where: str) -> PoseSpec:
    if table is None:
        return PoseSpec()
    _check_keys(table, {"rotation", "translation"}, where)
    pose = PoseSpec()
    if "rotation" in table:
        rot = list(table["rotation"])
        if len(rot) != 4:
            raise ConfigError(f"[{where}].rotation must be a quaternion [x,y,z,w]")
        pose.rotation = [float(v) for v in rot]
    if "translation" in table:
        tr = list(table["translation"])
        if len(tr) != 3:
            raise ConfigError(f"[{where}].translation must be [x,y,z]")
        pose.translation = [float(v) for v in tr]
    return pose


def _distortion(table: Optional[dict], where: str) -> DistortionSpec:
    if table is None:
        return DistortionSpec()
    _check_keys(table, {"model", "parameters"}, where)
    return DistortionSpec(
        model=str(table.get("model", "Pinhole")),
        parameters=[float(v) for v in table.get("parameters", [])],
    )


# --------------------------------------------------------------------------- #
# section parsers
# --------------------------------------------------------------------------- #

def _parse_camera(table: dict, idx: int) -> CameraSpec:
    where = f"rig.cameras[{idx}]"
    _check_keys(
        table,
        {
            "size", "principal", "focal", "rig_from_camera", "distortion",
            "border_top", "border_bottom", "border_left", "border_right",
        },
        where,
    )
    cam = CameraSpec()
    if "size" in table:
        cam.size = [int(v) for v in table["size"]]
    if "principal" in table:
        cam.principal = [float(v) for v in table["principal"]]
    if "focal" in table:
        cam.focal = [float(v) for v in table["focal"]]
    cam.rig_from_camera = _pose(table.get("rig_from_camera"), f"{where}.rig_from_camera")
    cam.distortion = _distortion(table.get("distortion"), f"{where}.distortion")
    cam.border_top = int(table.get("border_top", 0))
    cam.border_bottom = int(table.get("border_bottom", 0))
    cam.border_left = int(table.get("border_left", 0))
    cam.border_right = int(table.get("border_right", 0))
    return cam


def _parse_imu(table: dict) -> ImuSpec:
    _check_keys(
        table,
        {
            "rig_from_imu", "gyroscope_noise_density", "accelerometer_noise_density",
            "gyroscope_random_walk", "accelerometer_random_walk", "frequency",
        },
        "rig.imu",
    )
    return ImuSpec(
        rig_from_imu=_pose(table.get("rig_from_imu"), "rig.imu.rig_from_imu"),
        gyroscope_noise_density=float(table.get("gyroscope_noise_density", 0.0)),
        accelerometer_noise_density=float(table.get("accelerometer_noise_density", 0.0)),
        gyroscope_random_walk=float(table.get("gyroscope_random_walk", 0.0)),
        accelerometer_random_walk=float(table.get("accelerometer_random_walk", 0.0)),
        frequency=float(table.get("frequency", 200.0)),
    )


def _parse_rig(table: Optional[dict]) -> Optional[RigSpec]:
    if table is None:
        return None
    _check_keys(table, {"cameras", "imu"}, "rig")
    cameras_tbl = table.get("cameras")
    if not cameras_tbl:
        raise ConfigError("[rig] is present but defines no [[rig.cameras]].")
    cameras = [_parse_camera(c, i) for i, c in enumerate(cameras_tbl)]
    imu = _parse_imu(table["imu"]) if "imu" in table else None
    return RigSpec(cameras=cameras, imu=imu)


def _parse_odometry(table: Optional[dict]) -> OdometrySpec:
    spec = OdometrySpec()
    if table is None:
        return spec
    _check_keys(
        table,
        {
            "odometry_mode", "multicam_mode", "use_gpu", "async_sba",
            "use_motion_model", "use_denoising", "rectified_stereo_camera",
            "enable_observations_export", "enable_landmarks_export",
            "enable_final_landmarks_export", "max_frame_delta_s", "rgbd",
        },
        "odometry",
    )
    spec.odometry_mode = str(table.get("odometry_mode", spec.odometry_mode))
    spec.multicam_mode = str(table.get("multicam_mode", spec.multicam_mode))
    for key in ("use_gpu", "async_sba", "use_motion_model", "use_denoising",
                "rectified_stereo_camera"):
        if key in table:
            setattr(spec, key, bool(table[key]))
    if "enable_observations_export" in table:
        spec.enable_observations_export = bool(table["enable_observations_export"])
    if "enable_landmarks_export" in table:
        spec.enable_landmarks_export = bool(table["enable_landmarks_export"])
    if "enable_final_landmarks_export" in table:
        spec.enable_final_landmarks_export = bool(table["enable_final_landmarks_export"])
    if "max_frame_delta_s" in table:
        spec.max_frame_delta_s = float(table["max_frame_delta_s"])
    if "rgbd" in table:
        r = table["rgbd"]
        _check_keys(
            r,
            {"depth_scale_factor", "depth_camera_id", "enable_depth_stereo_tracking"},
            "odometry.rgbd",
        )
        spec.rgbd = RGBDSpec(
            depth_scale_factor=float(r.get("depth_scale_factor", 1000.0)),
            depth_camera_id=int(r.get("depth_camera_id", 0)),
            enable_depth_stereo_tracking=bool(r.get("enable_depth_stereo_tracking", False)),
        )
    return spec


def _parse_slam(table: Optional[dict]) -> SlamSpec:
    spec = SlamSpec()
    if table is None:
        return spec
    _check_keys(
        table,
        {
            "enabled", "map_cache_path", "use_gpu", "sync_mode",
            "enable_reading_internals", "planar_constraints", "gt_align_mode",
            "map_cell_size", "max_landmarks_distance", "max_map_size",
            "throttling_time_ms", "localize",
        },
        "slam",
    )
    # Presence of the [slam] table enables SLAM unless explicitly disabled.
    spec.enabled = bool(table.get("enabled", True))
    if "map_cache_path" in table:
        spec.map_cache_path = str(table["map_cache_path"])
    for key in ("use_gpu", "sync_mode", "enable_reading_internals",
                "planar_constraints", "gt_align_mode"):
        if key in table:
            setattr(spec, key, bool(table[key]))
    if "map_cell_size" in table:
        spec.map_cell_size = float(table["map_cell_size"])
    if "max_landmarks_distance" in table:
        spec.max_landmarks_distance = float(table["max_landmarks_distance"])
    if "max_map_size" in table:
        spec.max_map_size = int(table["max_map_size"])
    if "throttling_time_ms" in table:
        spec.throttling_time_ms = int(table["throttling_time_ms"])
    if "localize" in table:
        loc = table["localize"]
        _check_keys(
            loc,
            {
                "map_path", "guess", "horizontal_search_radius",
                "vertical_search_radius", "horizontal_step", "vertical_step",
                "angular_step_rads",
            },
            "slam.localize",
        )
        spec.localize = LocalizeSpec(
            map_path=str(loc.get("map_path", "")),
            guess=_pose(loc.get("guess"), "slam.localize.guess"),
            horizontal_search_radius=float(loc.get("horizontal_search_radius", 8.0)),
            vertical_search_radius=float(loc.get("vertical_search_radius", 2.0)),
            horizontal_step=float(loc.get("horizontal_step", 0.5)),
            vertical_step=float(loc.get("vertical_step", 0.2)),
            angular_step_rads=float(loc.get("angular_step_rads", 0.03)),
        )
    return spec


def _parse_eval(table: Optional[dict]) -> EvalSpec:
    spec = EvalSpec()
    if table is None:
        return spec
    _check_keys(
        table,
        {"enabled", "ground_truth", "gt_format", "gt_time_unit", "gt_fps",
         "align", "apply_gt_extrinsic", "max_time_diff", "rpe_distances", "report"},
        "eval",
    )
    spec.enabled = bool(table.get("enabled", True))
    if "ground_truth" in table:
        spec.ground_truth = str(table["ground_truth"])
    spec.gt_format = str(table.get("gt_format", "euroc"))
    if spec.gt_format not in ("euroc", "tum", "kitti"):
        raise ConfigError("[eval].gt_format must be euroc|tum|kitti")
    spec.gt_time_unit = str(table.get("gt_time_unit", "s"))
    spec.gt_fps = float(table.get("gt_fps", 10.0))
    spec.align = str(table.get("align", "auto"))
    if spec.align not in ("auto", "se3", "sim3", "none"):
        raise ConfigError("[eval].align must be auto|se3|sim3|none")
    spec.apply_gt_extrinsic = str(table.get("apply_gt_extrinsic", "auto"))
    if spec.apply_gt_extrinsic not in ("auto", "euroc_cam0", "none"):
        raise ConfigError("[eval].apply_gt_extrinsic must be auto|euroc_cam0|none")
    spec.max_time_diff = float(table.get("max_time_diff", 0.02))
    if "rpe_distances" in table:
        rd = table["rpe_distances"]
        if isinstance(rd, str):
            if rd != "kitti":
                raise ConfigError('[eval].rpe_distances string must be "kitti"')
            spec.rpe_distances = [100, 200, 300, 400, 500, 600, 700, 800]
        else:
            spec.rpe_distances = [float(x) for x in rd]
    spec.report = str(table.get("report", ""))
    if spec.enabled and not spec.ground_truth:
        raise ConfigError("[eval] is enabled but no 'ground_truth' path was given.")
    return spec


def _parse_run(table: Optional[dict]) -> RunSpec:
    spec = RunSpec()
    if table is None:
        return spec
    _check_keys(
        table,
        {"verbosity", "warm_up_gpu", "start_index", "max_frames", "sleep_ms"},
        "run",
    )
    spec.verbosity = int(table.get("verbosity", 0))
    spec.warm_up_gpu = bool(table.get("warm_up_gpu", False))
    spec.start_index = int(table.get("start_index", 0))
    spec.max_frames = int(table.get("max_frames", 0))
    spec.sleep_ms = float(table.get("sleep_ms", 0.0))
    return spec


def _parse_output(table: Optional[dict]) -> OutputSpec:
    spec = OutputSpec()
    if table is None:
        return spec
    _check_keys(
        table,
        {"trajectory", "pose_source", "timestamp_unit", "save_map",
         "visualize", "print_every"},
        "output",
    )
    spec.trajectory = str(table.get("trajectory", ""))
    spec.pose_source = str(table.get("pose_source", "auto"))
    spec.timestamp_unit = str(table.get("timestamp_unit", "s"))
    spec.save_map = str(table.get("save_map", ""))
    spec.visualize = bool(table.get("visualize", False))
    spec.print_every = int(table.get("print_every", 50))
    if spec.pose_source not in ("auto", "odometry", "slam"):
        raise ConfigError("[output].pose_source must be auto|odometry|slam")
    if spec.timestamp_unit not in ("s", "ms", "us", "ns"):
        raise ConfigError("[output].timestamp_unit must be s|ms|us|ns")
    return spec


# --------------------------------------------------------------------------- #
# top-level entry point
# --------------------------------------------------------------------------- #

def load_config(path: str) -> Config:
    """Parse a TOML file into a :class:`Config`."""
    with open(path, "rb") as handle:
        data: dict[str, Any] = tomllib.load(handle)

    _check_keys(data, {"run", "input", "rig", "odometry", "slam", "output", "eval"}, "<root>")

    if "input" not in data:
        raise ConfigError("Missing required [input] table.")
    input_table = data["input"]
    if "type" not in input_table:
        raise ConfigError('[input] must define a "type" (e.g. type = "image_folder").')

    return Config(
        run=_parse_run(data.get("run")),
        input=input_table,
        rig=_parse_rig(data.get("rig")),
        odometry=_parse_odometry(data.get("odometry")),
        slam=_parse_slam(data.get("slam")),
        output=_parse_output(data.get("output")),
        eval=_parse_eval(data.get("eval")),
    )
