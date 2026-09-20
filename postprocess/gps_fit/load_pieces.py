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
import os

import numpy as np

import scene as scene_mod
from config import DA3_UNITS_TO_METRES
from postprocess.gps_fit.fit import fit_similarity_2d, real_en, use_origin
from reconstruct.join_segments import _read_ply_points


def load_pieces(directory, min_confidence=None):
    """(fits, clouds) keyed by piece index, in the scene's metre frame.

    A single-node piece cannot fit its own rotation -- one point fixes a
    position and says nothing about a heading -- so it borrows one from the
    nearest multi-node piece. Road alignment recovers it properly, which is
    the whole point.
    """
    sc = scene_mod.Scene.load(directory)
    use_origin(*sc.origin)
    groups = sc.pieces(min_confidence=min_confidence)
    scale = DA3_UNITS_TO_METRES

    fits, singles = {}, []
    for i, members in enumerate(groups):
        nodes = [sc.nodes[m] for m in members]
        cams = np.array([real_en(n.pano.lat, n.pano.lon) for n in nodes])
        src = np.array([[n.position[0], n.position[2]] for n in nodes])
        if len(nodes) == 1:
            singles.append(i)
            fits[i] = {"cams": cams, "n": 1, "resid": None, "src_xz": src,
                       "members": members}
            continue
        R, _, t = fit_similarity_2d(src * scale, cams)
        res = np.linalg.norm(src * scale @ R.T + t - cams, axis=1)
        fits[i] = {"R": R, "scale": scale, "t": t, "cams": cams, "n": len(nodes),
                   "resid": float(np.median(res)), "src_xz": src, "members": members}

    multi = [i for i in fits if fits[i]["n"] > 1]
    for i in singles:
        c = fits[i]["cams"][0]
        near = min(multi, key=lambda j: np.linalg.norm(fits[j]["cams"] - c, axis=1).min())
        R = fits[near]["R"]
        fits[i].update(R=R, scale=scale, borrowed_from=near,
                       t=c - scale * (R @ fits[i]["src_xz"][0]))

    clouds = {}
    for i, f in fits.items():
        # each piece keeps its own rotation, but its offset is re-solved so
        # its cameras still land on their GPS positions
        f["t"] = (f["cams"] - scale * (f["src_xz"] @ f["R"].T)).mean(0)
        # where the cameras END UP, which is not where GPS put them: the fit
        # spreads its residual across them, and the cloud follows these, not
        # the GPS points
        f["placed"] = f["src_xz"] @ f["R"].T * scale + f["t"]
        pts, cols = _read_cloud(directory, sc, f["members"])
        xz = pts[:, [0, 2]] @ f["R"].T * scale + f["t"]
        clouds[i] = (xz, pts[:, 1] * scale, cols)
    return fits, clouds


def _read_cloud(directory, sc, members):
    """One piece's points, from the nodes of it that have any."""
    read = [_read_ply_points(os.path.join(directory, sc.nodes[m].ply))
            for m in members if sc.nodes[m].ply]
    if not read:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return (np.concatenate([p for p, _ in read]),
            np.concatenate([c for _, c in read]))
