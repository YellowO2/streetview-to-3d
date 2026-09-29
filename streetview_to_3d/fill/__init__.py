"""Finish a placed scene: one ground, and its colour.

Runs once, right after placement (postprocess.pipeline), on the scene in
place. DA3's shape is never moved; only its ground points are replaced.

    one_ground.py  every cloud's ground becomes one smooth surface, its
                   holes and the blind disc under each camera included
    paint.py       colour for it, patch by patch from the nearest
                   pano that sees it cleanly; what no camera has in view
                   at all is left out.
                   DA3 keeps its own colours

Points belong to nodes (see scene.py), so every added point is written into
the node whose pano coloured it -- in that node's own frame, like the rest
of its cloud -- and the viewer needs nothing new.

    python -m streetview_to_3d.fill SCENE_DIR [OUT_DIR]
"""
import os
import time

import numpy as np
from scipy.spatial import cKDTree

from streetview_to_3d import scene as scene_mod
from streetview_to_3d.fill.one_ground import one_ground
from streetview_to_3d.fill.paint import Camera, blurred, paint
from streetview_to_3d.postprocess.ground import normals_from_neighbours
from streetview_to_3d.postprocess.ply_io import read_ply, write_ply
from streetview_to_3d.services.segment import pano_mask


def _photo(pano, scene_dir):
    """(image path, class map) of a pano, or None: the same download the
    reconstruction used (cached) and the class map the scene keeps for it
    (made here on the CPU if it has none)."""
    from streetview_to_3d.services.segment import labels_path, pano_labels
    from streetview_to_3d.services.streetview_fetch import DA3_ONLY_ZOOM, download_pano_by_id, run_async
    if pano.source != "google":
        return None
    path = run_async(download_pano_by_id(pano.id, zoom=DA3_ONLY_ZOOM))
    return (path, pano_labels(path, device="cpu", saved=labels_path(scene_dir, pano.id))) if path else None


def run(scene_dir, log=print):
    """Fill the placed scene at scene_dir in place. (Google's depth maps --
    their ground and their walls above DA3's reach -- once filled in too;
    the terrain's land and OSM buildings do that now, without a download.)"""
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
    from streetview_to_3d.postprocess.water import wet_map
    keep, ground = one_ground(clouds, cams, normals, wet_map(sc))
    clouds = [x[k] for x, k in zip(clouds, keep)]
    colours = [c[k] for c, k in zip(colours, keep)]
    da3 = np.concatenate(clouds)
    photos = [_photo(n.pano, scene_dir) for n in nodes]
    t1 = time.monotonic()

    added = ground
    col, who, in_view = paint(added, da3, cameras,
                              [ph and (ph[0], pano_mask(ph[1]) | blurred(ph[0])) for ph in photos])
    # in view but no camera may colour it (the spot under a camera, masked
    # spots): the colour of the nearest painted point. Out of every
    # camera's view (who stays -1): left out.
    painted, fallback = who >= 0, in_view & (who < 0)
    if painted.any() and fallback.any():
        _, nb = cKDTree(added[painted]).query(added[fallback])
        src = np.flatnonzero(painted)[nb]
        col[fallback], who[fallback] = col[src], who[src]
    ok = who >= 0

    for k, n in enumerate(nodes):
        mine = ok & (who == k)
        x = np.concatenate([clouds[k], added[mine]])
        T = np.asarray(n.transform, float)
        write_ply(os.path.join(scene_dir, n.ply), (x - T[:3, 3]) @ np.linalg.inv(T[:3, :3]).T,
                  np.concatenate([colours[k], col[mine]]))
    log(f"fill: one ground {int(ok.sum())} of {len(ground)} points (the rest in no camera's view), "
        f"{int(sum(len(c) for c in clouds))} DA3 points kept  "
        f"[ground {t1 - t0:.1f}s, paint {time.monotonic() - t1:.1f}s]")
