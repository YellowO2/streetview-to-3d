"""Finish a placed scene in place: level the pieces (level.py), lay one ground under them
(one_ground.py) and colour it from the panos (paint.py). Added ground goes into the node whose
pano painted it and is also saved as seams.SceneGround; DA3's points gain a sway value.

    python -m streetview_to_3d.fill SCENE_DIR [OUT_DIR]
"""
import os
import time

import numpy as np
from scipy.spatial import cKDTree

from streetview_to_3d import scene as scene_mod
from streetview_to_3d.fill import level
from streetview_to_3d.fill.one_ground import grounds, one_ground
from streetview_to_3d.postprocess.seams import SceneGround
from streetview_to_3d.fill.paint import Camera, blurred, paint
from streetview_to_3d.postprocess.ply_io import read_node, write_ply
from streetview_to_3d.services.segment import pano_mask

SWAY_M = 6.0      # vegetation sways fully at this height over the ground


def _photo(pano, scene_dir):
    """(image path, class map) of a Google pano, or None; both cached."""
    from streetview_to_3d.services.segment import labels_path, pano_labels
    from streetview_to_3d.services.streetview_fetch import fetch_da3_pano
    if pano.source != "google":
        return None
    path = fetch_da3_pano(pano.id)
    return (path, pano_labels(path, device="cpu", saved=labels_path(scene_dir, pano.id))) if path else None


def _classed(points, camera, photo, classes, unknown):
    """Which of a node's points its pano's class map labels one of classes (unknown without a map)."""
    from streetview_to_3d.fill.paint import _at
    from streetview_to_3d.services.segment import LABEL_IDS
    if photo is None:
        return np.full(len(points), unknown)
    u, v, _, _ = camera.look(points)
    return np.isin(_at(photo[1], u, v), [LABEL_IDS[c] for c in classes])


def _walkable(points, camera, photo):
    """Which of a node's own points its pano calls WALKABLE; all of them without a class map."""
    from streetview_to_3d.fill.one_ground import WALKABLE
    return _classed(points, camera, photo, WALKABLE, True)


def run(scene_dir, log=print):
    """Fill the placed scene at scene_dir in place."""
    t0 = time.monotonic()
    sc = scene_mod.Scene.load(scene_dir)
    nodes = [n for n in sc.nodes if n.ply and n.transform]
    if not nodes:
        return
    clouds, colours = [], []
    for n in nodes:
        p, c, world = read_node(scene_dir, n)
        clouds.append(world)
        colours.append(c if c is not None else np.full((len(p), 3), 0.5))
    cameras = [Camera(n) for n in nodes]
    cams = np.array([c.centre for c in cameras])
    photos = [_photo(n.pano, scene_dir) for n in nodes]
    G = grounds(clouds, cams, [_walkable(x, cam, ph) for x, cam, ph in zip(clouds, cameras, photos)])

    log(level.level(sc, nodes, clouds, G))
    sc.save(scene_dir)
    cameras = [Camera(n) for n in nodes]
    cams = np.array([c.centre for c in cameras])
    keep, ground = one_ground(clouds, cams, G)
    clouds = [x[k] for x, k in zip(clouds, keep)]
    colours = [c[k] for c, k in zip(colours, keep)]
    da3 = np.concatenate(clouds)
    t1 = time.monotonic()

    col, who = paint(ground, da3, cameras,
                     [ph and (ph[0], pano_mask(ph[1]) | blurred(ph[0])) for ph in photos])
    # points no camera can colour take the nearest painted point's colour: dropping them would leave holes
    painted, fallback = who >= 0, who < 0
    if painted.any() and fallback.any():
        _, nb = cKDTree(ground[painted]).query(ground[fallback])
        src = np.flatnonzero(painted)[nb]
        col[fallback], who[fallback] = col[src], who[src]
    ok = who >= 0
    # road as the painting pano labels it (life.py keeps cars to it)
    road = np.zeros(len(ground), bool)
    for k in range(len(nodes)):
        mine = ok & (who == k)
        if mine.any():
            road[mine] = _classed(ground[mine], cameras[k], photos[k], ("road",), False)
    scene_ground = SceneGround.from_points(ground[ok], col[ok], road[ok])
    scene_ground.save(scene_dir)

    for k, n in enumerate(nodes):
        mine = ok & (who == k)
        x = np.concatenate([clouds[k], ground[mine]])
        # vegetation sways more the higher it is over the ground (world: y down)
        tree = _classed(clouds[k], cameras[k], photos[k], ("vegetation",), False)
        over = -clouds[k][:, 1] - scene_ground.at(clouds[k][:, [0, 2]])[1]
        sway = np.r_[tree * np.clip(np.nan_to_num(over) / SWAY_M, 0, 1), np.zeros(int(mine.sum()))]
        T = np.asarray(n.transform, float)
        write_ply(os.path.join(scene_dir, n.ply), (x - T[:3, 3]) @ np.linalg.inv(T[:3, :3]).T,
                  np.concatenate([colours[k], col[mine]]), sway=sway)
    log(f"fill: one ground {int(ok.sum())} of {len(ground)} points, "
        f"{int(sum(len(c) for c in clouds))} DA3 points kept  "
        f"[ground {t1 - t0:.1f}s, paint {time.monotonic() - t1:.1f}s]")
