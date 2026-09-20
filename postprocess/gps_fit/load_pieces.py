"""Load a directory of reconstructed pieces into one shared metre frame.

Each piece arrives as a .ply in DA3's own arbitrary units plus a meta.json
listing the panoramas that built it. Fitting a piece's node positions to
those panoramas' GPS gives the rotation, scale and offset that put it in
the world -- and doing that for every piece puts them all in the SAME
world, which is what makes aligning them to each other possible at all.
"""
import os

import numpy as np

from reconstruct.join_segments import _read_ply_points
import scene as scene_mod
from config import DA3_UNITS_TO_METRES
from postprocess.gps_fit.fit import fit_nodes, use_origin, real_en

def load_pieces(directory):
    """Each piece's GPS fit + point cloud, all in the shared GLOBAL_ORIGIN
    frame. A single-node piece cannot fit its own rotation/scale, so it
    borrows them from the nearest multi-node piece -- its heading is then
    recovered properly by road alignment, which is the whole point.

    Every piece is scaled by the SAME measured constant
    (config.DA3_UNITS_TO_METRES), never by its own fitted scale. DA3
    reconstructs every piece in the same units, so a real per-piece
    difference does not exist -- what a per-piece fit measures is how
    noisy that piece's GPS was. Letting each keep its own turns that
    noise into geometry, and deriving one factor from the run's own
    pieces makes the world a different size depending on which pieces
    you happened to select.

    It applies to HEIGHT as well as x and z: height is in DA3 units like
    everything else, and scaling only two axes of three leaves every
    piece squashed vertically, quietly corrupting every slope in the
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

    scale = DA3_UNITS_TO_METRES
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

