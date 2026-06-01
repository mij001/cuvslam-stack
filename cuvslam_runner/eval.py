"""Trajectory evaluation against ground truth.

Computes the standard SLAM accuracy metrics requested for benchmarking:

* **ATE / RMSE APE** - root-mean-square of the absolute position error after a
  rigid (SE3) or similarity (Sim3, for monocular) alignment to ground truth.
  Reported in metres.
* **RPE** - relative pose error normalised by travelled distance: translation
  drift in percent and rotation drift in deg/m, KITTI-style over a set of
  segment lengths.
* **avgRTE %** - average relative translation error (the mean of the per-segment
  translation drift), in percent.
* **avgRE deg** - average relative rotation error per segment, in degrees.

All geometry is done with numpy; rotations use scipy. The module has no
dependency on ``cuvslam`` so it can evaluate any TUM-format trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.spatial.transform import Rotation as _R
except ModuleNotFoundError:  # pragma: no cover
    _R = None


# --------------------------------------------------------------------------- #
# trajectory containers / loaders
# --------------------------------------------------------------------------- #

@dataclass
class Trajectory:
    timestamps_ns: np.ndarray          # (N,) int64
    poses: np.ndarray                  # (N, 4, 4) world_from_frame


def _quat_to_mat(qx, qy, qz, qw) -> np.ndarray:
    if _R is None:  # pragma: no cover
        raise RuntimeError("scipy is required for trajectory evaluation")
    return _R.from_quat([qx, qy, qz, qw]).as_matrix()


def _pose(translation, quat_xyzw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = translation
    return T


def load_tum(path: str, time_unit: str = "s") -> Trajectory:
    """Load ``timestamp tx ty tz qx qy qz qw`` lines."""
    scale = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0}[time_unit]
    ts, poses = [], []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            v = [float(x) for x in line.replace(",", " ").split()]
            ts.append(int(round(v[0] * scale)))
            poses.append(_pose(v[1:4], v[4:8]))
    return Trajectory(np.array(ts, dtype=np.int64), np.array(poses))


def load_gt_euroc(path: str) -> Trajectory:
    """EuRoC ``state_groundtruth_estimate0``: ts[ns], px,py,pz, qw,qx,qy,qz, ..."""
    ts, poses = [], []
    with open(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            v = [float(x) for x in line.split(",")]
            ts.append(int(v[0]))
            poses.append(_pose(v[1:4], [v[5], v[6], v[7], v[4]]))  # qx,qy,qz,qw
    return Trajectory(np.array(ts, dtype=np.int64), np.array(poses))


def load_gt_tum(path: str, time_unit: str = "s") -> Trajectory:
    return load_tum(path, time_unit)


def load_gt_kitti(path: str, fps: float = 10.0) -> Trajectory:
    """KITTI poses: 12 numbers per line (3x4 row-major), index-timed at ``fps``."""
    poses, ts = [], []
    with open(path) as handle:
        for i, line in enumerate(handle):
            if not line.strip():
                continue
            m = np.array([float(x) for x in line.split()]).reshape(3, 4)
            T = np.eye(4)
            T[:3, :4] = m
            poses.append(T)
            ts.append(int(round(i * 1e9 / fps)))
    return Trajectory(np.array(ts, dtype=np.int64), np.array(poses))


def apply_right_extrinsic(traj: Trajectory, extrinsic: np.ndarray) -> Trajectory:
    """Right-multiply every pose: world_from_B = world_from_A @ A_from_B.

    Used to move EuRoC ground truth from the body/IMU frame into the cam0 frame
    so it shares the cuVSLAM output frame.
    """
    poses = np.einsum("nij,jk->nik", traj.poses, extrinsic)
    return Trajectory(traj.timestamps_ns, poses)


def read_euroc_cam0_extrinsic(sensor_yaml_path: str) -> np.ndarray:
    """Return T_body_from_cam0 (the cam0 ``T_BS``) as a 4x4 matrix."""
    import yaml
    with open(sensor_yaml_path) as handle:
        cfg = yaml.safe_load(handle)
    for key in cfg:
        if key.startswith("T_"):
            return np.array(cfg[key]["data"]).reshape(4, 4)
    raise ValueError(f"No T_* transform in {sensor_yaml_path}")


# --------------------------------------------------------------------------- #
# association + alignment
# --------------------------------------------------------------------------- #

def associate(est: Trajectory, gt: Trajectory, max_diff_ns: int) -> Tuple[np.ndarray, np.ndarray]:
    """Match each estimate pose to the nearest GT pose within ``max_diff_ns``."""
    gt_ts = gt.timestamps_ns
    order = np.argsort(gt_ts)
    gt_sorted = gt_ts[order]
    est_idx, gt_idx = [], []
    for i, t in enumerate(est.timestamps_ns):
        pos = np.searchsorted(gt_sorted, t)
        cands = []
        if pos < len(gt_sorted):
            cands.append(pos)
        if pos > 0:
            cands.append(pos - 1)
        best = min(cands, key=lambda c: abs(int(gt_sorted[c]) - int(t)))
        if abs(int(gt_sorted[best]) - int(t)) <= max_diff_ns:
            est_idx.append(i)
            gt_idx.append(order[best])
    return np.array(est_idx, dtype=int), np.array(gt_idx, dtype=int)


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool) -> Tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (Umeyama 1991). Maps src -> dst: dst ~= s*R@src + t."""
    mu_src = src.mean(0)
    mu_dst = dst.mean(0)
    sc = src - mu_src
    dc = dst - mu_dst
    cov = (dc.T @ sc) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scale:
        var_src = (sc ** 2).sum() / src.shape[0]
        s = (D * np.diag(S)).sum() / var_src
    else:
        s = 1.0
    t = mu_dst - s * R @ mu_src
    return s, R, t


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def _rot_angle_deg(Rm: np.ndarray) -> float:
    tr = np.clip((np.trace(Rm) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(tr))


def _stats(errors: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
    }


def _path_lengths(positions: np.ndarray) -> np.ndarray:
    """Cumulative travelled distance at each pose (positions: (N,3))."""
    seg = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def rpe_segments(gt_poses: np.ndarray, est_poses: np.ndarray,
                 distances: List[float]) -> Dict:
    """KITTI-style relative pose error over fixed travelled-distance segments.

    Both trajectories must be in the SAME frame (apply the GT extrinsic first).
    Returns per-distance and aggregate translation (%) and rotation (deg/m, deg).
    """
    gt_pos = gt_poses[:, :3, 3]
    cum = _path_lengths(gt_pos)
    n = len(gt_poses)

    per_distance = {}
    all_trans_pct: List[float] = []
    all_rot_deg_per_m: List[float] = []
    all_rot_deg: List[float] = []

    for L in distances:
        trans_pct, rot_dpm, rot_deg = [], [], []
        last_start = 0
        for i in range(n):
            # first j with travelled distance >= L
            target = cum[i] + L
            j = np.searchsorted(cum, target)
            if j >= n:
                break
            rel_gt = np.linalg.inv(gt_poses[i]) @ gt_poses[j]
            rel_est = np.linalg.inv(est_poses[i]) @ est_poses[j]
            err = np.linalg.inv(rel_gt) @ rel_est
            t_err = np.linalg.norm(err[:3, 3])
            r_err = _rot_angle_deg(err[:3, :3])
            trans_pct.append(100.0 * t_err / L)
            rot_dpm.append(r_err / L)
            rot_deg.append(r_err)
        if trans_pct:
            per_distance[L] = {
                "trans_pct": float(np.mean(trans_pct)),
                "rot_deg_per_m": float(np.mean(rot_dpm)),
                "rot_deg": float(np.mean(rot_deg)),
                "count": len(trans_pct),
            }
            all_trans_pct += trans_pct
            all_rot_deg_per_m += rot_dpm
            all_rot_deg += rot_deg

    if not all_trans_pct:
        return {"per_distance": {}, "avg_trans_pct": float("nan"),
                "avg_rot_deg_per_m": float("nan"), "avg_rot_deg": float("nan"),
                "segments": 0}
    return {
        "per_distance": per_distance,
        "avg_trans_pct": float(np.mean(all_trans_pct)),
        "avg_rot_deg_per_m": float(np.mean(all_rot_deg_per_m)),
        "avg_rot_deg": float(np.mean(all_rot_deg)),
        "segments": len(all_trans_pct),
    }


@dataclass
class EvalResult:
    matched: int = 0
    align: str = "se3"
    scale: float = 1.0
    ate: Dict[str, float] = field(default_factory=dict)      # translation APE stats (m)
    ape_full_rmse: float = float("nan")                       # full SE3 APE trans RMSE (m)
    rpe: Dict = field(default_factory=dict)
    traj_length_m: float = 0.0


def evaluate(est: Trajectory, gt: Trajectory, *,
             align: str = "se3",
             max_diff_ns: int = 20_000_000,
             rpe_distances: Optional[List[float]] = None) -> EvalResult:
    ei, gi = associate(est, gt, max_diff_ns)
    if len(ei) < 3:
        raise ValueError(
            f"Only {len(ei)} GT/estimate matches within "
            f"{max_diff_ns/1e6:.1f} ms; cannot evaluate."
        )
    est_p = est.poses[ei]
    gt_p = gt.poses[gi]
    est_xyz = est_p[:, :3, 3]
    gt_xyz = gt_p[:, :3, 3]

    with_scale = (align == "sim3")
    if align == "none":
        s, R, t = 1.0, np.eye(3), np.zeros(3)
    else:
        s, R, t = umeyama(est_xyz, gt_xyz, with_scale)

    aligned_xyz = (s * (R @ est_xyz.T).T) + t
    ate_err = np.linalg.norm(gt_xyz - aligned_xyz, axis=1)

    # aligned full poses (for full APE + RPE)
    S = np.eye(4)
    S[:3, :3] = s * R
    S[:3, 3] = t
    aligned_poses = np.einsum("ij,njk->nik", S, est_p)
    ape_full = np.linalg.norm(gt_p[:, :3, 3] - aligned_poses[:, :3, 3], axis=1)

    res = EvalResult(matched=len(ei), align=align, scale=float(s))
    res.ate = _stats(ate_err)
    res.ape_full_rmse = float(np.sqrt(np.mean(ape_full ** 2)))
    res.traj_length_m = float(_path_lengths(gt_xyz)[-1])

    if rpe_distances is None:
        L = res.traj_length_m
        rpe_distances = [d for d in (1, 2, 4, 8, 16, 32, 64) if d < L] or [max(L / 4, 0.1)]
    # For RPE the estimate must be scaled into GT units (mono) and in GT frame.
    rpe_est = np.einsum("ij,njk->nik", S, est_p) if with_scale else est_p
    res.rpe = rpe_segments(gt_p, rpe_est, rpe_distances)
    return res


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def format_report(res: EvalResult, title: str = "") -> str:
    lines = []
    if title:
        lines.append(f"==== Trajectory evaluation: {title} ====")
    lines.append(f"matched poses        : {res.matched}")
    lines.append(f"GT trajectory length : {res.traj_length_m:.2f} m")
    lines.append(f"alignment            : {res.align}"
                 + (f"  (scale={res.scale:.5f})" if res.align == 'sim3' else ""))
    lines.append("")
    lines.append("Absolute error (after alignment):")
    lines.append(f"  ATE / RMSE APE      : {res.ate['rmse']*100:.2f} cm   ({res.ate['rmse']:.4f} m)")
    lines.append(f"  APE mean / median   : {res.ate['mean']:.4f} / {res.ate['median']:.4f} m")
    lines.append(f"  APE std / min / max : {res.ate['std']:.4f} / {res.ate['min']:.4f} / {res.ate['max']:.4f} m")
    lines.append("")
    rpe = res.rpe
    lines.append("Relative error (drift per unit distance):")
    lines.append(f"  avgRTE (translation): {rpe['avg_trans_pct']:.3f} %")
    lines.append(f"  RPE rotation        : {rpe['avg_rot_deg_per_m']:.4f} deg/m")
    lines.append(f"  avgRE (rotation)    : {rpe['avg_rot_deg']:.4f} deg")
    lines.append(f"  segments evaluated  : {rpe['segments']}")
    if rpe.get("per_distance"):
        lines.append("  per-distance breakdown:")
        lines.append("    len(m)   trans%    rot(deg/m)   n")
        for L in sorted(rpe["per_distance"]):
            d = rpe["per_distance"][L]
            lines.append(f"    {L:6.1f}   {d['trans_pct']:6.3f}   {d['rot_deg_per_m']:9.4f}   {d['count']}")
    return "\n".join(lines)
