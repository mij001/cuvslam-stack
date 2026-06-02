"""Trajectory evaluation against ground truth.

This module reproduces the three metrics the cuVSLAM technical report uses
(arXiv:2506.04359, Tables 2-4) and is cross-checked numerically against the
``evo`` library (Grupp 2017). All geometry is plain numpy; rotations use scipy.
There is no dependency on ``cuvslam`` so it can score any TUM-format trajectory.

------------------------------------------------------------------------------
THE THREE METRICS, AND WHY THE SAME RUN CAN LOOK GOOD OR TERRIBLE
------------------------------------------------------------------------------

1. **ATE / RMSE APE (Absolute Pose Error)** - global accuracy.
   Align the estimate to GT with one rigid transform (Umeyama 1991): SE(3) for
   metric sensors, or Sim(3) when scale is unknown (monocular) or to reproduce
   the paper's "with scale correction". Then RMSE of the per-pose position error.
   This is a single, well-defined number; it is what `evo_ape ... -a` (or
   `-a --correct_scale`) prints.

2. **RPE-by-distance (KITTI / Geiger 2013)** - `rpe_segments()`.
   For each *segment length* L in {100..800 m} (the KITTI leaderboard set), take
   every sub-path of travelled length ~L and measure the leftover pose error,
   then normalise: translation_error / L  -> percent ("avgRTE %"), and
   angle / L -> deg/m. KEY GOTCHA: because the translation error is divided by L,
   the SAME drift looks huge at short L and tiny at long L (a 3 cm error is 3 %
   of a 1 m segment but 0.03 % of a 100 m segment). KITTI uses 100-800 m exactly
   so that per-frame noise averages out; using 1-16 m segments inflates avgRTE
   ~6-20x and is the usual reason "our numbers" look far worse than a paper's.
   Use this (with `rpe_distances="kitti"`) for KITTI-scale trajectories.

3. **RPE-by-delta (TUM / Sturm 2012)** - `rpe_delta()`.
   For short trajectories (EuRoC ~60 m, TUM/TUM-VI rooms ~10-20 m) you cannot cut
   100 m segments, so the TUM convention fixes a *delta* between the two poses --
   a number of frames, seconds, or metres -- and reports the relative pose error
   over every such pair, as an RMSE in metres / degrees (NOT a percent). The
   cuVSLAM report's TUM-VI table (Table 4) is exactly "RMSE RPE over 1-second
   segments", i.e. `rpe_delta(delta=1, unit="s")`. This matches
   `evo_rpe ... --delta 1 --delta_unit s [--all_pairs]`.

In all RPE variants the relative pose error for a pose pair (i, j) is

      E_ij = (G_i^{-1} G_j)^{-1} (P_i^{-1} P_j)

i.e. "the GT motion from i to j, undone, then the estimated motion from i to j",
expressed in the start frame. ``trans(E_ij)`` is the leftover position error and
``angle(rot(E_ij))`` the leftover rotation error -- both are independent of the
global world frame, so RPE needs no alignment (translation), and is robust to
the constant body<->camera offset (rotation angle is conjugation-invariant).
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


def _delta_pairs(n: int, delta: float, unit: str,
                 timestamps_ns: np.ndarray, cum_dist: np.ndarray,
                 all_pairs: bool) -> List[Tuple[int, int]]:
    """Index pairs (i, j) separated by ``delta`` in frames / seconds / metres.

    Reproduces evo's pairing:
      * ``all_pairs=True``  -> every start i gets paired with the first j that is
        at least ``delta`` away (evo's ``--all_pairs``); pairs overlap.
      * ``all_pairs=False`` -> greedy non-overlapping chain 0->j0->j1->...
        (evo's default), so each pose is used once and the errors are independent.

    The "first j at least delta away" is found with ``searchsorted`` on a
    monotonically increasing key: the frame index itself (unit="f"), the
    timestamp array (unit="s"), or the cumulative travelled distance (unit="m").
    Both timestamps and cumulative distance are non-decreasing, so searchsorted is
    exact and O(log n).
    """
    def end_index(i: int):
        if unit == "f":
            j = i + int(round(delta))
        elif unit == "s":
            # timestamps are ns; delta is seconds.
            j = int(np.searchsorted(timestamps_ns, timestamps_ns[i] + delta * 1e9))
        elif unit == "m":
            j = int(np.searchsorted(cum_dist, cum_dist[i] + delta))
        else:
            raise ValueError("rpe delta unit must be 'f' (frames), 's' (seconds) or 'm' (metres)")
        return j if (j < n and j > i) else None

    pairs: List[Tuple[int, int]] = []
    if all_pairs:
        for i in range(n):
            j = end_index(i)
            if j is not None:
                pairs.append((i, j))
    else:
        i = 0
        while i < n:
            j = end_index(i)
            if j is None:
                break
            pairs.append((i, j))
            i = j  # jump to the end of this pair -> non-overlapping
    return pairs


def rpe_delta(gt_poses: np.ndarray, est_poses: np.ndarray, *,
              timestamps_ns: np.ndarray,
              delta: float = 1.0, unit: str = "s",
              all_pairs: bool = True) -> Dict:
    """TUM-style Relative Pose Error over a fixed delta (Sturm 2012 / evo_rpe).

    Both trajectories must be in the SAME frame and (for monocular) the SAME scale
    -- callers pass the scale-aligned estimate. Unlike ``rpe_segments`` this does
    NOT normalise by distance: it returns the leftover translation error in metres
    and rotation error in degrees over each pose pair separated by ``delta``,
    summarised as mean and RMSE -- exactly what ``evo_rpe`` reports.

    Equivalent evo invocation:
        evo_rpe tum gt.txt est.txt -a --delta <delta> --delta_unit <f|s|m> [--all_pairs]
    """
    # Cumulative travelled distance along the GT (needed only for unit="m").
    cum = _path_lengths(gt_poses[:, :3, 3])
    pairs = _delta_pairs(len(gt_poses), delta, unit, timestamps_ns, cum, all_pairs)

    t_err, r_err = [], []
    for i, j in pairs:
        rel_gt = np.linalg.inv(gt_poses[i]) @ gt_poses[j]      # GT motion i->j
        rel_est = np.linalg.inv(est_poses[i]) @ est_poses[j]   # estimated motion i->j
        err = np.linalg.inv(rel_gt) @ rel_est                  # leftover error E_ij
        t_err.append(float(np.linalg.norm(err[:3, 3])))        # metres
        r_err.append(_rot_angle_deg(err[:3, :3]))              # degrees

    if not pairs:
        nan = float("nan")
        return {"unit": unit, "delta": delta, "pairs": 0,
                "trans_rmse_m": nan, "trans_mean_m": nan,
                "rot_rmse_deg": nan, "rot_mean_deg": nan}
    t = np.asarray(t_err)
    r = np.asarray(r_err)
    return {
        "unit": unit, "delta": delta, "pairs": len(pairs),
        "trans_rmse_m": float(np.sqrt(np.mean(t ** 2))),
        "trans_mean_m": float(np.mean(t)),
        "rot_rmse_deg": float(np.sqrt(np.mean(r ** 2))),
        "rot_mean_deg": float(np.mean(r)),
    }


@dataclass
class EvalResult:
    matched: int = 0
    align: str = "se3"
    scale: float = 1.0
    ate: Dict[str, float] = field(default_factory=dict)      # translation APE stats (m)
    ape_full_rmse: float = float("nan")                       # full SE3 APE trans RMSE (m)
    rpe: Dict = field(default_factory=dict)                   # KITTI distance-segment RPE
    rpe_delta: Optional[Dict] = None                          # TUM fixed-delta RPE (optional)
    traj_length_m: float = 0.0


def evaluate(est: Trajectory, gt: Trajectory, *,
             align: str = "se3",
             max_diff_ns: int = 20_000_000,
             rpe_distances: Optional[List[float]] = None,
             index_assoc: bool = False,
             rpe_delta_value: Optional[float] = None,
             rpe_delta_unit: str = "s",
             rpe_all_pairs: bool = True) -> EvalResult:
    if index_assoc:
        # KITTI-style: GT poses are 1:1 with frames, so pair by index rather than
        # by (possibly mismatched) timestamps.
        n = min(len(est.poses), len(gt.poses))
        ei = np.arange(n)
        gi = np.arange(n)
    else:
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

    if rpe_distances == "kitti":
        rpe_distances = [100, 200, 300, 400, 500, 600, 700, 800]
    if rpe_distances is None:
        L = res.traj_length_m
        rpe_distances = [d for d in (1, 2, 4, 8, 16, 32, 64) if d < L] or [max(L / 4, 0.1)]
    # For RPE the estimate must be scaled into GT units (mono) and in GT frame.
    # (Sim3 alignment carries the recovered scale; for SE3 the estimate is metric
    # already so no scaling is applied.)
    rpe_est = np.einsum("ij,njk->nik", S, est_p) if with_scale else est_p
    res.rpe = rpe_segments(gt_p, rpe_est, rpe_distances)

    # Optional TUM-style fixed-delta RPE (e.g. the paper's "RMSE RPE over 1 s").
    # Uses the matched GT timestamps; for index association the estimate carries
    # the timestamps (GT had none).
    if rpe_delta_value is not None:
        ts = (est.timestamps_ns[ei] if index_assoc else gt.timestamps_ns[gi])
        res.rpe_delta = rpe_delta(
            gt_p, rpe_est,
            timestamps_ns=np.asarray(ts, dtype=np.float64),
            delta=rpe_delta_value, unit=rpe_delta_unit, all_pairs=rpe_all_pairs,
        )
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
    if res.rpe_delta is not None:
        rd = res.rpe_delta
        unit = {"s": "second", "f": "frame", "m": "metre"}.get(rd["unit"], rd["unit"])
        lines.append("")
        lines.append(f"TUM RPE over a fixed {rd['delta']:g}-{unit} delta "
                     f"({'all pairs' if True else ''}, {rd['pairs']} pairs):")
        lines.append(f"  RPE translation     : {rd['trans_rmse_m']:.4f} m rmse "
                     f"({rd['trans_mean_m']:.4f} m mean)")
        lines.append(f"  RPE rotation        : {rd['rot_rmse_deg']:.4f} deg rmse "
                     f"({rd['rot_mean_deg']:.4f} deg mean)")
    return "\n".join(lines)
