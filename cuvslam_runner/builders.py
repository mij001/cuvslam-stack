"""Convert plain specs into live ``cuvslam`` objects.

This is the single place that imports ``cuvslam``; everything else operates on
the dependency-free dataclasses in :mod:`cuvslam_runner.specs`.
"""

from __future__ import annotations

from typing import Optional

import cuvslam

from .specs import (
    CameraSpec,
    DistortionSpec,
    ImuSpec,
    OdometrySpec,
    PoseSpec,
    RigSpec,
    SlamSpec,
)


def _enum_from_name(enum_cls, name: str, what: str):
    try:
        return getattr(enum_cls, name)
    except AttributeError:
        valid = [v for v in dir(enum_cls) if not v.startswith("_")]
        raise ValueError(f"Invalid {what} {name!r}. Valid values: {valid}") from None


# --------------------------------------------------------------------------- #
# rig
# --------------------------------------------------------------------------- #

def build_pose(spec: PoseSpec) -> "cuvslam.Pose":
    return cuvslam.Pose(rotation=list(spec.rotation), translation=list(spec.translation))


def build_distortion(spec: DistortionSpec) -> "cuvslam.Distortion":
    model = _enum_from_name(cuvslam.Distortion.Model, spec.model, "distortion model")
    return cuvslam.Distortion(model, list(spec.parameters))


def build_camera(spec: CameraSpec) -> "cuvslam.Camera":
    cam = cuvslam.Camera()
    cam.size = list(spec.size)
    cam.principal = list(spec.principal)
    cam.focal = list(spec.focal)
    cam.rig_from_camera = build_pose(spec.rig_from_camera)
    cam.distortion = build_distortion(spec.distortion)
    cam.border_top = spec.border_top
    cam.border_bottom = spec.border_bottom
    cam.border_left = spec.border_left
    cam.border_right = spec.border_right
    return cam


def build_imu(spec: ImuSpec) -> "cuvslam.ImuCalibration":
    return cuvslam.ImuCalibration(
        rig_from_imu=build_pose(spec.rig_from_imu),
        gyroscope_noise_density=spec.gyroscope_noise_density,
        accelerometer_noise_density=spec.accelerometer_noise_density,
        gyroscope_random_walk=spec.gyroscope_random_walk,
        accelerometer_random_walk=spec.accelerometer_random_walk,
        frequency=spec.frequency,
    )


def build_rig(spec: RigSpec) -> "cuvslam.Rig":
    if not spec.cameras:
        raise ValueError("Rig has no cameras.")
    rig = cuvslam.Rig()
    rig.cameras = [build_camera(c) for c in spec.cameras]
    rig.imus = [build_imu(spec.imu)] if spec.imu is not None else []
    return rig


# --------------------------------------------------------------------------- #
# odometry / slam config
# --------------------------------------------------------------------------- #

def build_odometry_config(spec: OdometrySpec) -> "cuvslam.Tracker.OdometryConfig":
    cfg = cuvslam.Tracker.OdometryConfig()
    cfg.odometry_mode = _enum_from_name(
        cuvslam.Tracker.OdometryMode, spec.odometry_mode, "odometry_mode"
    )
    cfg.multicam_mode = _enum_from_name(
        cuvslam.Tracker.MulticameraMode, spec.multicam_mode, "multicam_mode"
    )
    for attr in ("use_gpu", "async_sba", "use_motion_model", "use_denoising",
                 "rectified_stereo_camera", "max_frame_delta_s"):
        value = getattr(spec, attr)
        if value is not None:
            setattr(cfg, attr, value)
    cfg.enable_observations_export = spec.enable_observations_export
    cfg.enable_landmarks_export = spec.enable_landmarks_export
    if spec.enable_final_landmarks_export is not None:
        cfg.enable_final_landmarks_export = spec.enable_final_landmarks_export

    if spec.rgbd is not None or spec.odometry_mode == "RGBD":
        rgbd = cuvslam.Tracker.OdometryRGBDSettings()
        if spec.rgbd is not None:
            rgbd.depth_scale_factor = spec.rgbd.depth_scale_factor
            rgbd.depth_camera_id = spec.rgbd.depth_camera_id
            rgbd.enable_depth_stereo_tracking = spec.rgbd.enable_depth_stereo_tracking
        cfg.rgbd_settings = rgbd
    return cfg


def build_slam_config(spec: SlamSpec) -> Optional["cuvslam.Tracker.SlamConfig"]:
    if not spec.enabled:
        return None
    cfg = cuvslam.Tracker.SlamConfig()
    if spec.map_cache_path is not None:
        cfg.map_cache_path = spec.map_cache_path
    for attr in ("use_gpu", "sync_mode", "enable_reading_internals",
                 "planar_constraints", "gt_align_mode", "map_cell_size",
                 "max_landmarks_distance", "max_map_size", "throttling_time_ms"):
        value = getattr(spec, attr)
        if value is not None:
            setattr(cfg, attr, value)
    return cfg


def build_localization_settings(loc) -> "cuvslam.Tracker.SlamLocalizationSettings":
    return cuvslam.Tracker.SlamLocalizationSettings(
        horizontal_search_radius=loc.horizontal_search_radius,
        vertical_search_radius=loc.vertical_search_radius,
        horizontal_step=loc.horizontal_step,
        vertical_step=loc.vertical_step,
        angular_step_rads=loc.angular_step_rads,
    )
