"""Place every piece of a scene in the world: one rigid fit (and scale) per piece putting each
camera on its GPS point CAM_H above its elevation, turned to its pano's heading, pitch and roll.
World frame: x east, y down, z north, metres from the scene's centre.

    python -m streetview_to_3d.postprocess.place SCENE_DIR
"""
import sys

import numpy as np

from streetview_to_3d.common import scene as scene_mod
from streetview_to_3d.models.da3 import DA3_UNITS_TO_METRES
from streetview_to_3d.common.geo import latlon_to_local_m

CAM_H = 2.45             # camera height above the ground
SIGMA_M = 0.5            # how far a GPS point is trusted
SIGMA_DEG = 2.0          # how far a pano's heading/pitch/roll is trusted
MIN_SCALE_SPAN_M = 8.0   # cameras closer than this: GPS noise swamps the scale
SCALE_RANGE = (0.9, 1.8) # fitted scales outside this are rejected


def place(scene_dir, log=print):
    """Solve every piece and save its transform onto its nodes."""
    sc = scene_mod.Scene.load(scene_dir)
    groups = sc.pieces()
    s0, why = scene_scale(sc, groups)
    log(f"scene scale: {s0:.2f} m per DA3 unit ({why})")
    for gi, members in enumerate(groups):
        nodes = [sc.nodes[m] for m in members]
        own = piece_scale(nodes, sc.origin)
        s = own if own is not None and SCALE_RANGE[0] <= own <= SCALE_RANGE[1] else s0
        T, off_m, off_deg = fit_piece(nodes, s, sc.origin)
        for m in members:
            sc.nodes[m].transform = T.tolist()
        whose = "its own" if s is own else "the scene's"
        log(f"  piece {gi}: {len(nodes)} node(s), scale {s:.2f} ({whose}), "
            f"cameras off their GPS point by up to {off_m:.2f} m, "
            f"off their pano's orientation by up to {off_deg:.1f} deg")
    sc.save(scene_dir)


def fit_piece(nodes, scale, origin):
    """(4x4 ply -> world, worst camera offset m, worst orientation offset deg).

    Maximises trace(R^T M), M = centred camera pairs plus each camera's
    world <- DA3 rotation, weighted so a SIGMA_DEG turn costs a SIGMA_M shift."""
    P = np.array([np.asarray(n.position, float) * scale for n in nodes])
    Q = np.array([target(n, origin) for n in nodes])
    turns = [r for n in nodes if (r := _world_from_da3(n)) is not None]
    pc, qc = P.mean(0), Q.mean(0)
    M = (Q - qc).T @ (P - pc) + (SIGMA_M / np.radians(SIGMA_DEG)) ** 2 / 2 * sum(turns, np.zeros((3, 3)))
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    t = qc - R @ pc
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = scale * R, t
    off_m = float(np.linalg.norm(P @ R.T + t - Q, axis=1).max())
    off_deg = max((float(np.degrees(np.arccos(np.clip((np.trace(R.T @ r) - 1) / 2, -1, 1))))
                   for r in turns), default=0.0)
    return T, off_m, off_deg


def target(node, origin):
    """Where a node's camera really was, in the world frame."""
    e, n = latlon_to_local_m(node.pano.lat, node.pano.lon, *origin)
    return np.array([e, -((node.pano.elevation or 0.0) + CAM_H), n])


def photo_from_world(pano):
    """Rotation from world directions to the pano's photo frame (x right, y down, z forward):
    heading, then pitch, then roll (roll sign flipped, measured)."""
    h, p, r = pano.heading, pano.pitch or 0.0, -(pano.roll or 0.0)
    H = np.array([[np.cos(h), 0, -np.sin(h)], [0, 1, 0], [np.sin(h), 0, np.cos(h)]])
    Pm = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rm = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rm @ Pm @ H


def _world_from_da3(node):
    """world <- DA3 rotation according to one camera, or None without a heading."""
    if node.pano.heading is None or node.rotation is None:
        return None
    return photo_from_world(node.pano).T @ np.asarray(node.rotation, float)


def piece_scale(nodes, origin):
    """A piece's metres per DA3 unit fitted to GPS, or None if its cameras span under MIN_SCALE_SPAN_M."""
    if len(nodes) < 2:
        return None
    gps = np.array([latlon_to_local_m(n.pano.lat, n.pano.lon, *origin) for n in nodes])
    if np.linalg.norm(gps[:, None] - gps[None], axis=2).max() < MIN_SCALE_SPAN_M:
        return None
    return _scale_2d(np.array([[n.position[0], n.position[2]] for n in nodes]), gps)


def scene_scale(sc, groups):
    """(metres per DA3 unit, reason): the median piece_scale, else models.da3.DA3_UNITS_TO_METRES."""
    fitted = [f for m in groups if (f := piece_scale([sc.nodes[k] for k in m], sc.origin)) is not None]
    if not fitted:
        return DA3_UNITS_TO_METRES, "fixed: no piece spans enough to fit one"
    s = float(np.median(fitted))
    each = ", ".join(f"{f:.2f}" for f in fitted)
    if not SCALE_RANGE[0] <= s <= SCALE_RANGE[1]:
        return DA3_UNITS_TO_METRES, f"fixed: fitted {s:.2f} from [{each}] is out of range"
    return s, f"fitted, median of {len(fitted)} piece(s): {each}"


def _scale_2d(src, dst):
    """The scale of the least-squares similarity src -> dst (Umeyama)."""
    src_c, dst_c = src - src.mean(0), dst - dst.mean(0)
    U, S, Vt = np.linalg.svd(src_c.T @ dst_c / len(src))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return float((S[0] + d * S[1]) / (src_c ** 2).sum(1).mean())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__.splitlines()[-1].strip())
    place(sys.argv[1])
