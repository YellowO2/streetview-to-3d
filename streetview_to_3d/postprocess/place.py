"""Place every piece of a scene in the world, from its panoramas alone.

A piece is the nodes sharing a DA3 frame (scene.Scene.pieces). Its cameras
sit in that frame's arbitrary units, turned and tilted however DA3 started
it. Each of its panoramas knows where it really was and which way it
really pointed, so one rigid fit per piece puts it in the world:

  - where: every camera's DA3 centre onto its GPS point, CAM_H above its
    pano's elevation
  - which way: every camera's DA3 orientation onto its pano's own heading,
    pitch and roll. The photo keeps the camera's tilt -- a car on a slope,
    a backpack leaning 8 deg on a flat street -- and DA3 builds each view
    from the photo, so pitch and roll are what stand it upright. Without
    them a piece on a slope was levelled off the slope (NTU), and a lone
    pano had no tilt at all.

Both at once, least squares, weighed by how far each is trusted (SIGMA_M,
SIGMA_DEG): one rotation and one shift per piece, the piece itself never
bent. A lone pano lands exactly on its GPS point, turned exactly by its
heading, pitch and roll. On NTU (hilly) and Stockholm (a backpack capture)
every camera came out within 0.2 m of its elevation and 0.8 deg of its
pano's orientation.

Scale is one number per SCENE, never per piece (scene_scale): per-piece
scales turn GPS noise into pieces of different sizes.

This replaced a road-by-road alignment (road lines, sliding onto them,
matching kerbs across the road, a fitted elevation surface, tilt from
DA3's own ground): once pitch and roll were used, none of it helped.

World frame: x east, y down (so -elevation), z north, metres from the
scene's centre.

    python -m streetview_to_3d.postprocess.place SCENE_DIR
"""
import sys

import numpy as np

from streetview_to_3d import scene as scene_mod
from streetview_to_3d.config import DA3_UNITS_TO_METRES
from streetview_to_3d.services.geo import latlon_to_local_m

CAM_H = 2.45             # camera above the ground its pano's elevation gives
SIGMA_M = 0.5            # how far a GPS point is trusted
SIGMA_DEG = 2.0          # how far a pano's heading/pitch/roll is trusted
MIN_SCALE_SPAN_M = 8.0   # cameras closer than this: GPS noise swamps the scale
SCALE_RANGE = (0.9, 1.8) # a fitted scale outside this is a bad link, not a real scale


def place(scene_dir, log=print):
    """Solve every piece and save its transform onto its nodes."""
    sc = scene_mod.Scene.load(scene_dir)
    groups = sc.pieces()
    s, why = scene_scale(sc, groups)
    log(f"scale: {s:.2f} m per DA3 unit ({why})")
    for gi, members in enumerate(groups):
        nodes = [sc.nodes[m] for m in members]
        T, off_m, off_deg = fit_piece(nodes, s, sc.origin)
        for m in members:
            sc.nodes[m].transform = T.tolist()
        log(f"  piece {gi}: {len(nodes)} node(s), cameras off their GPS point by up to {off_m:.2f} m, "
            f"off their pano's orientation by up to {off_deg:.1f} deg")
    sc.save(scene_dir)


def fit_piece(nodes, scale, origin):
    """(4x4 ply -> world, worst camera offset m, worst orientation offset deg).

    Maximises trace(R^T M), M = sum of the centred camera pairs plus each
    camera's own world <- DA3 rotation, weighted so a SIGMA_DEG turn costs
    what a SIGMA_M shift does (chordal |R - R'|^2 ~ 2 theta^2)."""
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
    """The rotation carrying world directions into this pano's photo frame
    (x right, y down, z forward): its heading, then pitch, then roll. Signs
    measured: pitch as given, roll the other way (NTU, all four tried)."""
    h, p, r = pano.heading, pano.pitch or 0.0, -(pano.roll or 0.0)
    H = np.array([[np.cos(h), 0, -np.sin(h)], [0, 1, 0], [np.sin(h), 0, np.cos(h)]])
    Pm = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rm = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rm @ Pm @ H


def _world_from_da3(node):
    """world <- DA3 frame, as this one camera says: DA3's rotation carries
    its frame into the photo, the pano's orientation carries the photo into
    the world. None without a heading."""
    if node.pano.heading is None or node.rotation is None:
        return None
    return photo_from_world(node.pano).T @ np.asarray(node.rotation, float)


def scene_scale(sc, groups):
    """(metres per DA3 unit, reason) for the whole scene.

    Every multi-node piece whose cameras span at least MIN_SCALE_SPAN_M
    fits its own scale against GPS, top-down; the scene takes their median,
    so one bad piece cannot drag it. With no piece to measure, or a median
    outside SCALE_RANGE, it falls back to config.DA3_UNITS_TO_METRES."""
    fitted = []
    for members in groups:
        nodes = [sc.nodes[m] for m in members]
        if len(nodes) < 2:
            continue
        gps = np.array([latlon_to_local_m(n.pano.lat, n.pano.lon, *sc.origin) for n in nodes])
        if np.linalg.norm(gps[:, None] - gps[None], axis=2).max() < MIN_SCALE_SPAN_M:
            continue
        da3 = np.array([[n.position[0], n.position[2]] for n in nodes])
        fitted.append(_scale_2d(da3, gps))
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
