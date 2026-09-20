"""Load a directory of reconstructed pieces into one shared metre frame.

Each piece arrives as a .ply in DA3's own arbitrary units plus a meta.json
listing the panoramas that built it. Fitting a piece's node positions to
those panoramas' GPS gives the rotation, scale and offset that put it in
the world -- and doing that for every piece puts them all in the SAME
world, which is what makes aligning them to each other possible at all.
"""
import os

import numpy as np

from street_builder.reconstruction.join_segments import _read_ply_points
import scene as scene_mod
from postprocess.gps_fit.fit import fit_nodes, use_origin, real_en

# Trust a piece's own fitted scale only if its GPS fit is this good.
# Scale is the worst-determined part of a similarity fit, so a piece with
# a poor fit reports a badly wrong one -- measured here, every piece with
# a residual under 0.2 m agreed on 1.23-1.30, while the only two outliers
# (1.43 and 1.70) came from the only two pieces with residuals above 1 m.
GOOD_FIT_RESIDUAL_M = 0.25

def load_pieces(directory):
    """Each piece's GPS fit + point cloud, all in the shared GLOBAL_ORIGIN
    frame. A single-node piece cannot fit its own rotation/scale, so it
    borrows them from the nearest multi-node piece -- its heading is then
    recovered properly by road alignment, which is the whole point.

    Every piece is scaled by ONE shared factor, not by its own. DA3
    reconstructs all of them in the same units, so a genuine per-piece
    scale difference should not exist; what the per-piece fits actually
    measure is how noisy each piece's GPS was. Letting each piece keep
    its own turns that noise into geometry -- pieces come out different
    sizes, and no amount of moving them will ever make them meet.

    The shared factor is taken from the pieces whose GPS fit is good,
    and it is applied to HEIGHT as well as to x and z. It converts DA3
    units to metres, and the height is in DA3 units like everything
    else; scaling only two axes of three leaves every piece squashed
    vertically, which quietly corrupts every slope and height in the
    scene."""
    sc = scene_mod.Scene.load(directory)
    use_origin(*sc.origin)

    fits, singles = {}, []
    for i, piece in enumerate(sc.pieces):
        cams = np.array([real_en(n.lat, n.lon) for n in piece.nodes])
        if len(piece) == 1:
            singles.append(i)
            fits[i] = {"cams": cams, "n": 1, "resid": None,
                       "da3": np.array(piece.nodes[0].position)[[0, 2]]}
            continue
        nodes = [{"da3_xz": [n.position[0], n.position[2]],
                  "lat": n.lat, "lon": n.lon} for n in piece.nodes]
        R, scale, t, _, _, res = fit_nodes(nodes)
        fits[i] = {"R": R, "scale": scale, "t": t, "cams": cams,
                   "n": len(piece), "resid": float(np.median(res)),
                   "src_xz": np.array([n["da3_xz"] for n in nodes])}

    multi = [i for i in fits if fits[i]["n"] > 1]
    for i in singles:
        c = fits[i]["cams"][0]
        near = min(multi, key=lambda j: np.linalg.norm(fits[j]["cams"] - c, axis=1).min())
        R, scale = fits[near]["R"], fits[near]["scale"]
        fits[i].update(R=R, scale=scale, t=c - scale * (R @ fits[i]["da3"]),
                       borrowed_from=near)

    scale = global_scale(fits)
    clouds = {}
    for i in fits:
        # keep each piece's own rotation, but re-solve its offset for the
        # shared scale, so its cameras still land on their GPS positions
        f = fits[i]
        f["own_scale"], f["scale"] = f["scale"], scale
        src = f["da3"][None, :] if f["n"] == 1 else f["src_xz"]
        f["t"] = (f["cams"] - scale * (src @ f["R"].T)).mean(0)

        pts, cols = _read_ply_points(os.path.join(directory, sc.pieces[i].ply))
        xz = pts[:, [0, 2]] @ f["R"].T * scale + f["t"]
        clouds[i] = (xz, pts[:, 1] * scale, cols)
    return fits, clouds


def global_scale(fits):
    """One DA3-units-to-metres factor for the whole scene, averaged over
    the pieces whose GPS fit was good enough to have measured it."""
    good = [f["scale"] for f in fits.values()
            if f["resid"] is not None and f["resid"] <= GOOD_FIT_RESIDUAL_M]
    if not good:
        good = [f["scale"] for f in fits.values() if f["resid"] is not None]
    return float(np.median(good))
