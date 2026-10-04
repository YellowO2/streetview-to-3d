"""Experimental: Google 3D Tiles dabs filling what DA3 has not. google.ply (saved by the viewer's
?google-save) is laid onto DA3's points (shift), gets the panos' look (seams.unhazed), loses
the spots DA3 has (has) and what stands where a pano looked through (seen_through), and takes
DA3's colour, look and surface where it meets them (seams.toward, seams.pulled). Rewrites
google.ply.

    python -m streetview_to_3d.postprocess.world.google SCENE_DIR
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

from streetview_to_3d.common import scene as scene_mod
from streetview_to_3d.postprocess import seams
from streetview_to_3d.postprocess.ply_io import read_node, read_vertices, write_ply
from streetview_to_3d.postprocess.sight import Sight
from streetview_to_3d.postprocess.world import terrain

FILENAME = "google.ply"
MERGED = "merged"
FIT_REACH_M, FIT_N, FIT_NEAR_M = 15.0, 40_000, 0.3     # shift: Google points this near a pano, about this many
FIT_STEPS_M = 1.0, 0.3, 0.1                            # ... moved by up to 3 of each step in turn
SPOT_M, SAME_M, SPOT_K = 0.3, 1.5, 8  # has: DA3 this near a Google point's normal line, this far along it, of this many nearest
CLEAR_M = 0.5, 1.5          # seen_through: a Google point this much nearer a pano than what it saw; surely by this much
REACH_M = 25.0              # as far as a pano's DA3 points go (clean.blobs.FAR_M)


def shift(pts, scene, cams):
    """The move (x, y, z; m) that lays Google's points onto DA3's: the one putting most of the
    Google points near the panos within FIT_NEAR_M of a DA3 point, searched coarse to fine."""
    near = cKDTree(cams).query(pts, distance_upper_bound=FIT_REACH_M)[0] < np.inf
    sample = pts[near][::max(1, int(near.sum() // FIT_N))]
    tree = cKDTree(scene)
    best = np.zeros(3)
    for step in FIT_STEPS_M:
        moves = best + step * np.stack(np.meshgrid(*[np.arange(-3, 4)] * 3, indexing="ij"), -1).reshape(-1, 3)
        hits = [np.isfinite(tree.query(sample + m, distance_upper_bound=FIT_NEAR_M, workers=-1)[0]).sum()
                for m in moves]
        best = moves[int(np.argmax(hits))]
    return best


def has(pts, normal, tree):
    """Per Google point, whether DA3 has that spot of its surface: one of the nearest DA3 points
    (tree) within SPOT_M of the line through it along its normal, no further along it than
    SAME_M -- the same wall or ground, a little off or not. Where DA3 has a hole there is none,
    and the Google point stays to fill it."""
    d, k = tree.query(pts, k=SPOT_K, distance_upper_bound=SAME_M, workers=-1)
    to = tree.data[np.minimum(k, tree.n - 1)] - pts[:, None]
    along = (to * normal[:, None]).sum(2)
    aside = np.sqrt(np.maximum((to ** 2).sum(2) - along ** 2, 0))
    return (np.isfinite(d) & (aside < SPOT_M)).any(1)


def seen_through(pts, clouds, cams):
    """0-1 per Google point: how surely a pano looked through where it stands (1: gone): CLEAR_M
    nearer the pano than what it saw there (sight.Sight), in clear air in front of DA3's points."""
    gone = np.zeros(len(pts))
    tree = cKDTree(pts)
    for cloud, cam in zip(clouds, cams):
        near = np.asarray(tree.query_ball_point(cam, REACH_M))
        if not len(cloud) or not len(near):
            continue
        behind, sure = Sight(cloud, cam).of(pts[near])
        gone[near] = np.maximum(gone[near], seams.ramp((-behind - CLEAR_M[0]) / (CLEAR_M[1] - CLEAR_M[0])) * sure)
    return gone


def merge(scene_dir, log=print):
    path = os.path.join(scene_dir, FILENAME)
    v, header = read_vertices(path)
    if f"comment {MERGED}\n" in header:
        log("google: already merged")
        return
    pts = np.stack([v["x"], v["y"], v["z"]], 1).astype(float)      # world: y down
    cols = np.stack([v["red"], v["green"], v["blue"]], 1) / 255.0
    gap, normal = v["gap"].astype(float), np.stack([v["nx"], v["ny"], v["nz"]], 1)

    sc = scene_mod.Scene.load(scene_dir)
    scene, scene_cols = terrain.scene_points(sc, scene_dir)
    nodes = [nd for nd in sc.nodes if nd.ply and nd.transform]
    cams = np.array([(np.asarray(nd.transform, float) @ [*nd.position, 1])[:3] for nd in nodes])
    move = shift(pts, scene, cams)
    log(f"google: moved {move.round(2)} m (x east, y down, z north) onto DA3's points")
    pts = pts + move
    xy = pts[:, [0, 2]]

    by_place = cKDTree(xy)

    def google_at(q):
        d, k = by_place.query(q, distance_upper_bound=2 * seams.CELL_M, workers=-1)
        out = np.full((len(q), 3), np.nan)
        out[np.isfinite(d)] = cols[k[np.isfinite(d)]]
        return out
    cols = seams.unhazed(seams.SceneGround.load(scene_dir), google_at)(cols)

    # Google stays wherever DA3 has not that spot of its surface, and no pano looked through it
    clouds = [read_node(scene_dir, nd)[2] for nd in nodes]
    spot = has(pts, normal, cKDTree(np.concatenate(clouds)))
    chance = np.sin(pts @ [12.9898, 78.233, 37.719]) * 43758.5453 % 1
    air = chance < seen_through(pts, clouds, cams)
    keep = ~spot & ~air
    tree = cKDTree(scene) if len(scene) else None
    cols, near = seams.toward(pts, cols, tree, scene_cols)
    pts = seams.pulled(pts, normal, near, tree)

    credit = [l[len("comment "):] for l in header.splitlines() if l.startswith("comment credit")]
    write_ply(path, pts[keep], cols[keep], gap=gap[keep], normal=normal[keep], near=near[keep],
              comments=credit + [MERGED])
    log(f"google: {int(keep.sum()):,} of {len(v):,} points kept ({int(spot.sum()):,} where DA3 has the spot, "
        f"{int((air & ~spot).sum()):,} a pano looked through)")

if __name__ == "__main__":
    merge(sys.argv[1])
