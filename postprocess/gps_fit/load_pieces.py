"""Put every piece of a scene into one shared metre frame.

A piece is the nodes sharing a DA3 frame -- a connected component of the
scene's edges. Its cameras sit in that frame's arbitrary units; their
panoramas sit at real lat/lons. Fitting one to the other gives the
rotation and offset that put the piece in the world, and doing it for
every piece puts them all in the SAME world, which is what makes aligning
them to each other possible at all.

Scale is not fitted. DA3 is internally consistent, so one measured
constant converts its units to metres (config.DA3_UNITS_TO_METRES). Fitting
it per piece would turn GPS noise into geometry: pieces come out different
sizes and no amount of moving them makes them meet.
"""
import math
import os

import numpy as np

import scene as scene_mod
from config import DA3_UNITS_TO_METRES
from postprocess.gps_fit.fit import fit_similarity_2d, real_en, use_origin
from reconstruct.join_segments import _read_ply_points


def load_pieces(directory, min_confidence=None):
    """(fits, clouds) keyed by piece index, in the scene's metre frame.

    A single-node piece cannot fit its own rotation -- one point fixes a
    position and says nothing about a heading -- so it takes one from its
    panorama's own heading (see heading_rotation). Only a node without a
    heading still borrows the nearest multi-node piece's rotation.
    """
    sc = scene_mod.Scene.load(directory)
    use_origin(*sc.origin)
    groups = sc.pieces(min_confidence=min_confidence)
    scale = DA3_UNITS_TO_METRES

    fits, singles = {}, []
    for i, members in enumerate(groups):
        nodes = [sc.nodes[m] for m in members]
        cams = np.array([real_en(n.pano.lat, n.pano.lon) for n in nodes])
        src = np.array([[n.camera[0], n.camera[2]] for n in nodes])
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
    read = [_read_ply_points(os.path.join(directory, sc.nodes[m].ply))
            for m in members if sc.nodes[m].ply]
    if not read:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return (np.concatenate([p for p, _ in read]),
            np.concatenate([c for _, c in read]))
