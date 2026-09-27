"""Finish a placed scene: one ground, Google's walls in DA3's gaps, colour.

Runs once, right after placement (postprocess.pipeline), on the scene in
place. DA3's shape is never moved; only its ground points are replaced.

    one_ground.py  every cloud's ground and Google's become one smooth
                   surface, the blind disc under each camera included
    google.py      Google's walls where DA3 has nothing, slid onto DA3's
                   own copy of a wall where it has one
    paint.py       colour for all of that, each point from the nearest pano
                   that sees it cleanly; DA3 keeps its own colours

Points belong to nodes (see scene.py), so every added point is written into
the node whose pano coloured it -- in that node's own frame, like the rest
of its cloud -- and the viewer needs nothing new.

    python -m streetview_to_3d.fill SCENE_DIR [OUT_DIR]
"""
import json
import os
import time

import numpy as np
from scipy.spatial import cKDTree

from streetview_to_3d import scene as scene_mod
from streetview_to_3d.fill.google import fill
from streetview_to_3d.fill.one_ground import one_ground
from streetview_to_3d.fill.paint import Camera, paint
from streetview_to_3d.postprocess.ground import normals_from_neighbours
from streetview_to_3d.postprocess.ply_io import read_ply, write_ply


def _photo(pano):
    """(image path, drop mask) of a pano, or None: the same download the
    reconstruction used (cached), masked like its points were -- on the
    CPU, since this runs after the GPU call."""
    from streetview_to_3d.services.segment import drop_movers
    from streetview_to_3d.services.streetview_fetch import DA3_ONLY_ZOOM, download_pano_by_id, run_async
    if pano.source != "google":
        return None
    path = run_async(download_pano_by_id(pano.id, zoom=DA3_ONLY_ZOOM))
    return (path, drop_movers([path], device="cpu")[0]) if path else None


def run(scene_dir, log=print):
    """Fill the placed scene at scene_dir in place."""
    from streetview_to_3d.google_base import build, gather
    t0 = time.monotonic()
    sc = scene_mod.Scene.load(scene_dir)
    nodes = [n for n in sc.nodes if n.ply and n.transform]
    if not nodes:
        return
    clouds, colours = [], []
    for n in nodes:
        p, c = read_ply(os.path.join(scene_dir, n.ply))
        T = np.asarray(n.transform, float)
        clouds.append(p @ T[:3, :3].T + T[:3, 3])
        colours.append(c if c is not None else np.full((len(p), 3), 0.5))
    cameras = [Camera(n) for n in nodes]
    cams = np.array([c.centre for c in cameras])
    normals = [normals_from_neighbours(x) for x in clouds]
    base = build(gather(json.load(open(os.path.join(scene_dir, scene_mod.FILENAME)))), log=log)
    t1 = time.monotonic()

    keep, ground = one_ground(clouds, cams, normals, base.ground)
    clouds = [x[k] for x, k in zip(clouds, keep)]
    colours = [c[k] for c, k in zip(colours, keep)]
    da3 = np.concatenate(clouds)
    walls = fill(base, da3, np.concatenate([nm[k] for nm, k in zip(normals, keep)]),
                 np.concatenate([da3, ground]))
    t2 = time.monotonic()

    added = np.concatenate([ground, walls])
    photos = [_photo(n.pano) for n in nodes]
    col, who = paint(added, np.concatenate([da3, added]), cameras, photos)
    is_ground = np.arange(len(added)) < len(ground)
    unseen = who < 0
    if (is_ground & unseen).any() and (is_ground & ~unseen).any():
        # ground nobody sees cleanly: its nearest painted neighbour's colour and pano
        _, nb = cKDTree(added[is_ground & ~unseen]).query(added[is_ground & unseen])
        idx = np.flatnonzero(is_ground & ~unseen)[nb]
        col[is_ground & unseen], who[is_ground & unseen] = col[idx], who[idx]
    ok = who >= 0

    for k, n in enumerate(nodes):
        mine = ok & (who == k)
        x = np.concatenate([clouds[k], added[mine]])
        T = np.asarray(n.transform, float)
        write_ply(os.path.join(scene_dir, n.ply), (x - T[:3, 3]) @ np.linalg.inv(T[:3, :3]).T,
                  np.concatenate([colours[k], col[mine]]))
    log(f"fill: one ground {len(ground)} points ({(~unseen[is_ground]).mean() * 100:.0f}% seen), "
        f"Google walls {int(ok[~is_ground].sum())} of {len(walls)} (unseen dropped), "
        f"{int(sum(len(c) for c in clouds))} DA3 points kept  "
        f"[base {t1 - t0:.1f}s, fill {t2 - t1:.1f}s, paint {time.monotonic() - t2:.1f}s]")
