"""Put every piece of a scene into one shared metre frame.

A piece is the nodes sharing a DA3 frame -- a connected component of the
scene's edges. Its cameras sit in that frame's arbitrary units; their
panoramas sit at real lat/lons. Fitting one to the other gives the
rotation and offset that put the piece in the world, and doing it for
every piece puts them all in the SAME world, which is what makes aligning
them to each other possible at all.

Scale is fitted once per SCENE, never per piece (see scene_scale). The
fixed constant (config.DA3_UNITS_TO_METRES) was measured on one street and
other runs of the same model fit 1.17-1.26, so it is only the fallback.
Fitting each piece its own scale would turn GPS noise into geometry: pieces
come out different sizes and no amount of moving them makes them meet.
"""
import math
import os

import numpy as np

from streetview_to_3d import scene as scene_mod
from streetview_to_3d.config import DA3_UNITS_TO_METRES
from streetview_to_3d.postprocess.gps_fit.fit import fit_similarity_2d, real_en, use_origin
from streetview_to_3d.postprocess.ply_io import read_ply


MIN_SCALE_SPAN_M = 8.0   # cameras closer than this: GPS noise swamps the fit
SCALE_RANGE = (0.9, 1.8) # a fit outside this is a bad link, not a real scale


def scene_scale(sc, groups):
    """(metres per DA3 unit, reason) for the whole scene.

    Every multi-node piece whose cameras span at least MIN_SCALE_SPAN_M
    fits its own scale against GPS; the scene takes their median, so one
    bad piece cannot drag it. A lone node has nothing to fit and takes the
    scene's value like everything else. With no piece to measure, or a
    median outside SCALE_RANGE, it falls back to DA3_UNITS_TO_METRES.
    """
    fitted = []
    for members in groups:
        nodes = [sc.nodes[m] for m in members]
        if len(nodes) < 2:
            continue
        cams = np.array([real_en(n.pano.lat, n.pano.lon) for n in nodes])
        if np.linalg.norm(cams[:, None] - cams[None], axis=2).max() < MIN_SCALE_SPAN_M:
            continue
        src = np.array([[n.position[0], n.position[2]] for n in nodes])
        fitted.append(fit_similarity_2d(src, cams)[1])
    if not fitted:
        return DA3_UNITS_TO_METRES, "fixed: no piece spans enough to fit one"
    s = float(np.median(fitted))
    each = ", ".join(f"{f:.2f}" for f in fitted)
    if not SCALE_RANGE[0] <= s <= SCALE_RANGE[1]:
        return DA3_UNITS_TO_METRES, f"fixed: fitted {s:.2f} from [{each}] is out of range"
    return s, f"fitted, median of {len(fitted)} piece(s): {each}"


def load_pieces(directory, min_confidence=None, log=print):
    """(fits, clouds) keyed by piece index, in the scene's metre frame.

    A single-node piece cannot fit its own rotation -- one point fixes a
    position and says nothing about a heading -- so it takes one from its
    panorama's own heading (see heading_rotation). Only a node without a
    heading still borrows the nearest multi-node piece's rotation.
    """
    sc = scene_mod.Scene.load(directory)
    use_origin(*sc.origin)
    groups = sc.pieces(min_confidence=min_confidence)
    scale, why = scene_scale(sc, groups)
    log(f"scale: {scale:.2f} m per DA3 unit ({why})")

    fits, singles = {}, []
    for i, members in enumerate(groups):
        nodes = [sc.nodes[m] for m in members]
        cams = np.array([real_en(n.pano.lat, n.pano.lon) for n in nodes])
        src = np.array([[n.position[0] * scale, n.position[2] * scale] for n in nodes])
        if len(nodes) == 1:
            singles.append(i)
            fits[i] = {"cams": cams, "n": 1, "resid": None, "src_xz": src,
                       "members": members}
            continue
        R, _, t = fit_similarity_2d(src, cams)
        res = np.linalg.norm(src @ R.T + t - cams, axis=1)
        fits[i] = {"R": R, "scale": scale, "t": t, "cams": cams, "n": len(nodes),
                   "resid": float(np.median(res)), "src_xz": src, "members": members}

    multi = [i for i in fits if fits[i]["n"] > 1]
    for i in singles:
        c = fits[i]["cams"][0]
        R = heading_rotation(sc.nodes[fits[i]["members"][0]])
        if R is not None:
            fits[i].update(R=R, scale=scale, rotation_from="heading")
        elif multi:
            near = min(multi, key=lambda j: np.linalg.norm(fits[j]["cams"] - c, axis=1).min())
            fits[i].update(R=fits[near]["R"], scale=scale, borrowed_from=near)
        else:
            raise ValueError("a panorama with no heading came out on its own, and no linked "
                             "piece exists to borrow a rotation from")
        fits[i]["t"] = c - fits[i]["R"] @ fits[i]["src_xz"][0]

    clouds = {}
    for i, f in fits.items():
        # each piece keeps its own rotation, but its offset is re-solved so
        # its cameras still land on their GPS positions
        f["t"] = (f["cams"] - f["src_xz"] @ f["R"].T).mean(0)
        # where the cameras END UP, which is not where GPS put them: the fit
        # spreads its residual across them, and the cloud follows these, not
        # the GPS points
        f["placed"] = f["src_xz"] @ f["R"].T + f["t"]
        pts, cols = _read_cloud(directory, sc, f["members"])
        xz = pts[:, [0, 2]] * scale @ f["R"].T + f["t"]
        clouds[i] = (xz, pts[:, 1] * scale, cols)
    return fits, clouds


def heading_rotation(node):
    """The 2D rotation (DA3's x, z -> east, north) that points this node's
    camera along its panorama's own heading, or None without one.

    The camera looks down its own +z; rotation.T carries that into DA3's
    frame. Measured on real placed pieces (the GPS fit's rotation against
    each node's heading): Google agrees to within 1.2 deg on 13 nodes.
    Apple faces the other way -- its heading (already converted to Street
    View's convention, see fetch_nodes._apple_heading) is the direction of
    travel, but the image's forward is 180 deg from it, within 1.3 deg on
    2 nodes. heading_agreement logs the same check on every run.
    """
    if node.rotation is None or node.pano.heading is None:
        return None
    v = np.asarray(node.rotation, float).T @ np.array([0.0, 0.0, 1.0])
    bearing = node.pano.heading + (math.pi if node.pano.source == "apple" else 0.0)
    th = math.atan2(v[0], v[2]) - bearing
    return np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])


def floor_normal_from_pitch(node):
    """The up-pointing floor normal DA3 should hand back for this node, in
    its piece's DA3 frame, from the panorama's own pitch -- or None.

    Measured on Stockholm: DA3's floor tilts along the heading by the
    pano's pitch, within 1-2 deg on 4 of 6 nodes (the other two were 4 and
    9 deg off), with the same sign rule as heading_rotation's -- Apple's
    image faces backwards. Roll matched nothing, so it is left out. Only
    good for holding a floor fit that can't pin itself, not replacing one.
    """
    if node.rotation is None or node.pano.pitch is None:
        return None
    p = node.pano.pitch
    s = 1.0 if node.pano.source == "apple" else -1.0
    # camera frame: x right, y down, z forward; up is -y
    return np.asarray(node.rotation, float).T @ np.array([0.0, -math.cos(p), s * math.sin(p)])


def heading_agreement(fits, sc):
    """{source: [deg, ...]}: for every node of a multi-node piece, how far
    heading_rotation's direction is from where the GPS fit actually
    points it. Near zero means single-node pieces are being turned right."""
    out = {}
    for f in fits.values():
        if f["n"] < 2:
            continue
        fitted = math.atan2(f["R"][1, 0], f["R"][0, 0])
        for m in f["members"]:
            R = heading_rotation(sc.nodes[m])
            if R is None:
                continue
            d = math.degrees(math.atan2(R[1, 0], R[0, 0]) - fitted)
            out.setdefault(sc.nodes[m].pano.source, []).append((d + 180) % 360 - 180)
    return out


def _read_cloud(directory, sc, members):
    """One piece's points, from the nodes of it that have any."""
    read = [read_ply(os.path.join(directory, sc.nodes[m].ply))
            for m in members if sc.nodes[m].ply]
    if not read:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return (np.concatenate([p for p, _ in read]),
            np.concatenate([c for _, c in read]))
