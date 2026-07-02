"""stages.py — the kernel→stage→persistence-class taxonomy for cuVSLAM.

This is the Slice-2 formalization of the DAG: every CUDA kernel cuVSLAM launches
is assigned to a canonical V-SLAM pipeline stage, and every stage carries the
persistence-class HYPOTHESIS from the thesis (streaming / hot-persistent /
cold-persistent) that the measurements test. Rules are ordered regexes over the
base kernel name (see common.base_kernel_name); first match wins.

Kernel inventory source: nsys cuda_gpu_kern_sum of KITTI seq06 + EuRoC V1_01 runs
(42 unique kernels observed) + the cuVSLAM technical report's module list
[Korovko25]. Kernels seen only when [slam] is enabled (loop closure / pose graph)
match the SLAM rules; anything unmatched lands in 'other' and is reported, never
silently dropped.
"""
from __future__ import annotations

import re

# stage -> (persistence-class hypothesis, description)
STAGES: dict[str, tuple[str, str]] = {
    "preprocess":      ("streaming",       "image cast + Gaussian pyramid construction"),
    "feature_detect":  ("streaming",       "GFTT/Harris gradients, response, NMS, selection"),
    "keypoint_sort":   ("streaming",       "cub::DeviceMergeSort of detected keypoints"),
    "tracking":        ("streaming+hot",   "Lucas–Kanade pyramidal optical-flow tracking"),
    "bundle_adjust":   ("hot-persistent",  "sparse bundle adjustment: system build + reduce + update"),
    "ba_solver":       ("hot-persistent",  "dense linear solve (cuSOLVER getrf/trsv) for BA"),
    "slam_loop":       ("cold-persistent", "loop closure / localization / pose-graph (SLAM layer)"),
    "other":           ("unclassified",    "unmatched kernels — inspect and extend stages.py"),
}

# ordered (regex, stage) rules over the BASE kernel name
_RULES: list[tuple[str, str]] = [
    # SLAM layer first: only present with [slam] enabled. st_build_cache /
    # st_track_with_cache are the keyframe-cache build/match kernels (observed
    # only in [slam] runs on TUM long_office; 'st' = SLAM tracker).
    (r"st_\w*cache|loop|bow|vocab|localiz|relocal|pose_graph|posegraph|"
     r"graph_optim|map_match|keyframe|covis|essential|sim3|icp", "slam_loop"),
    # front-end
    (r"cast_image|cast_depth|gaussian_scaling|undistort|rectif", "preprocess"),
    (r"conv_grad|gftt|non_max_suppression|filter_maximums|"
     r"select_features|downsample",                              "feature_detect"),
    (r"cub::DeviceMergeSort|cub::DeviceRadixSort|cub::DeviceScan|"
     r"cub::DeviceSelect|cub::DeviceReduce",                     "keypoint_sort"),
    # matcher:: = RGBD photometric / point-to-point alignment (data association,
    # observed in odometry-only runs → front-end tracking, not SLAM)
    (r"lk_track|track_kernel|of_track|matcher::",                "tracking"),
    # back-end
    (r"^sba::|sba_",                                             "bundle_adjust"),
    (r"getrf|trsv|potrf|dtrsv|xxtrf|copy_info|gemm|gemv|syrk",   "ba_solver"),
]

_COMPILED = [(re.compile(pat, re.IGNORECASE), stage) for pat, stage in _RULES]


def stage_of(base_name: str) -> str:
    for rx, stage in _COMPILED:
        if rx.search(base_name):
            return stage
    return "other"


def persistence_of(stage: str) -> str:
    return STAGES.get(stage, STAGES["other"])[0]


def describe(stage: str) -> str:
    return STAGES.get(stage, STAGES["other"])[1]


# Canonical pipeline order for tables/plots
ORDER = ["preprocess", "feature_detect", "keypoint_sort", "tracking",
         "bundle_adjust", "ba_solver", "slam_loop", "other"]
