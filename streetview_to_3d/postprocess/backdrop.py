"""A backdrop beyond the scene, from what DA3 dropped as unsure.

Every node keeps, beside its points, what DA3's confidence filter dropped
(its far cloud, services.da3_ops.FAR_EVERY): the near part of that is
unsure edges, the rest mostly buildings and trees too far away for DA3 to
be confident about. Here all of them are placed and cleaned into one
backdrop.ply beside scene.json (its "backdrop"):

1. far: only what is more than NEAR_M from every camera -- nothing a
   camera could walk into, and within it the scene's own points are better
2. another pano agrees: a point stays only if some other node's far cloud
   has a point within AGREE of its distance (at least AGREE_MIN_M) --
   DA3's error grows with distance; one pano's streak along an edge has
   no partner
3. no loose clumps: joined groups (CELL_M cells, LINK_M apart) under
   MIN_CELLS go, as in blobs.py but sized for far things
4. flat: every point moved onto the plane through the FLAT_K cell centres
   nearest it (a patch a few metres wide), FLAT_PASSES times -- nothing is
   thrown away, the bumps are flattened

On a 7-pano NTU patch at 15 m: 209k far points, 90k agreed, 88k kept;
flattening moved them a median 16 cm, then 9. 15 m came too close to the
scene; 20 now.

    python -m streetview_to_3d.postprocess.backdrop SCENE_DIR
"""
import os
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from streetview_to_3d import scene as scene_mod
from streetview_to_3d.postprocess.ply_io import read_ply, write_ply

FILENAME = "backdrop.ply"
NEAR_M = 20.0
AGREE, AGREE_MIN_M = 0.03, 0.5
CELL_M, LINK_M, MIN_CELLS = 0.5, 1.0, 50
FLAT_K, FLAT_PASSES = 30, 2


def far_points(sc, scene_dir):
    """(points, colours, which node, camera centres), placed in the world."""
    pts, cols, who, cams = [], [], [], []
    for n in sc.nodes:
        if not (n.ply and n.transform):
            continue
        T = np.asarray(n.transform, float)
        cams.append(T[:3, :3] @ np.asarray(n.position, float) + T[:3, 3])
        f = os.path.join(scene_dir, n.far_ply)
        if os.path.exists(f):
            p, c = read_ply(f)
            pts.append(p @ T[:3, :3].T + T[:3, 3])
            cols.append(c if c is not None else np.full((len(p), 3), .5))
            who.append(np.full(len(p), len(cams)))
    if not pts:
        return None
    return np.concatenate(pts), np.concatenate(cols), np.concatenate(who), np.array(cams)


def agreed(pts, who, dist):
    """True where another node's far cloud has a point close enough."""
    out = np.zeros(len(pts), bool)
    for w in np.unique(who):
        mine, others = who == w, who != w
        if others.any():
            gap = cKDTree(pts[others]).query(pts[mine])[0]
            out[mine] = gap <= np.maximum(AGREE_MIN_M, AGREE * dist[mine])
    return out


def solid(pts):
    """True on points in a joined group of MIN_CELLS cells or more."""
    cells, inv = np.unique(np.floor(pts / CELL_M).astype(np.int64), axis=0, return_inverse=True)
    pairs = cKDTree(cells * CELL_M).query_pairs(LINK_M, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(cells), len(cells)))
    _, group = connected_components(graph, directed=False)
    return (np.bincount(group)[group] >= MIN_CELLS)[inv.ravel()]


def flatten(x):
    """Each point moved onto the least-squares plane of the FLAT_K cell
    centres nearest it."""
    _, inv, n = np.unique(np.floor(x / CELL_M).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    centres = np.stack([np.bincount(inv, x[:, j]) for j in range(3)], 1) / n[:, None]
    tree = cKDTree(centres)
    nb = centres[tree.query(centres, k=min(FLAT_K, len(centres)))[1]]
    mean = nb.mean(1)
    cov = np.einsum("nki,nkj->nij", nb - mean[:, None], nb - mean[:, None])
    normal = np.linalg.eigh(cov)[1][:, :, 0]                  # the direction of least spread
    near = tree.query(x)[1]
    return x - np.einsum("ni,ni->n", x - mean[near], normal[near])[:, None] * normal[near]


def build(scene_dir, log=print):
    """Write scene_dir/FILENAME from the placed scene's far clouds."""
    sc = scene_mod.Scene.load(scene_dir)
    got = far_points(sc, scene_dir)
    if got is None:
        log("backdrop: no far clouds")
        return
    pts, cols, who, cams = got
    dist = cKDTree(cams).query(pts)[0]
    far = dist > NEAR_M
    pts, cols, who, dist = pts[far], cols[far], who[far], dist[far]
    ok = agreed(pts, who, dist)
    n_agreed = int(ok.sum())
    ok[ok] = solid(pts[ok])
    x = pts[ok]
    for _ in range(FLAT_PASSES):
        x = flatten(x) if len(x) else x
    write_ply(os.path.join(scene_dir, FILENAME), x, cols[ok])
    sc.backdrop = FILENAME
    sc.save(scene_dir)
    log(f"backdrop: {len(pts)} far points past {NEAR_M:.0f} m, {n_agreed} agreed, {len(x)} kept")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__.splitlines()[-1].strip())
    build(sys.argv[1])
