"""Rigid (rotation+translation only -- DA3's metric scale is trusted
consistent across independent inference calls, so no scale term needed)
alignment and merging of overlapping windowed DA3 reconstructions.

Only needed once a chain is long enough to require more than one DA3 call
(street_builder/reconstruct.py's reconstruct_chain_windowed) -- a single
window's own output is already in a usable frame on its own.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def _pose_pair_transform(center_from: np.ndarray, rot_from: np.ndarray, center_to: np.ndarray, rot_to: np.ndarray):
    """Rigid transform (R, t) mapping a point expressed in the 'from' frame to
    the 'to' frame, given the same physical pose (e.g. the same shared anchor
    image) expressed in both frames independently."""
    R = rot_to @ rot_from.T
    t = center_to - R @ center_from
    return R, t


def solve_rigid_alignment(shared_poses_from: list[tuple[np.ndarray, np.ndarray]], shared_poses_to: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Average rigid transform (R, t) mapping the 'from' window's frame onto
    the 'to' window's frame, given 2+ shared anchor poses expressed in both
    (each pose a (center, rotation) pair, same order in both lists).

    Each shared anchor independently implies its own (R, t); these are
    averaged (rotation via quaternion mean, translation directly) rather than
    trusting a single anchor, since DA3's own pose estimates carry some noise
    (we've seen dev_dist up to ~1m even on views that pass the filter).
    """
    if len(shared_poses_from) != len(shared_poses_to) or len(shared_poses_from) < 1:
        raise ValueError("Need at least 1 shared pose pair, and equal counts in both frames.")

    Rs, ts = [], []
    for (c_from, r_from), (c_to, r_to) in zip(shared_poses_from, shared_poses_to):
        R, t = _pose_pair_transform(c_from, r_from, c_to, r_to)
        Rs.append(R)
        ts.append(t)

    quats = np.array([Rotation.from_matrix(R).as_quat() for R in Rs])
    quats *= np.sign(quats @ quats[0])[:, None]  # flip to same hemisphere before averaging
    R_avg = Rotation.from_quat(quats.mean(axis=0)).as_matrix()
    t_avg = np.mean(ts, axis=0)
    return R_avg, t_avg


def compose_transforms(R_outer: np.ndarray, t_outer: np.ndarray, R_inner: np.ndarray, t_inner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) equivalent to applying the inner transform first, then the
    outer one -- used to fold a new window's local-to-previous-window
    transform into the running local-to-global-frame transform."""
    return R_outer @ R_inner, R_outer @ t_inner + t_outer


def apply_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return points @ R.T + t
