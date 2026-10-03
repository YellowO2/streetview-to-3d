"""Remove floating bits from a placed scene's DA3 points: points too far from their pano (past
FAR_M, or past SURE_M unless another pano confirms them) and small clumps touching nothing else.

    python -m streetview_to_3d.postprocess.clean.blobs SCENE_DIR
"""
import os
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from streetview_to_3d.common import scene as scene_mod
from streetview_to_3d.postprocess.ply_io import read_node, write_ply

VOXEL_M, LINK_M, MIN_CELLS = 0.1, 0.25, 300
FAR_M = 25.0                  # DA3 points this far from their pano are dropped
SURE_M, AGREE_M = 12.0, 0.3   # past SURE_M, kept only if a pano within SURE_M has a point within AGREE_M
EVERY = 2                     # subsampling of the confirming pano's points


def loose_bits(points):
    """True on points in a joined group of under MIN_CELLS cells."""
    cell = np.floor(points / VOXEL_M).astype(np.int64)
    cells, inv = np.unique(cell, axis=0, return_inverse=True)
    pairs = cKDTree(cells * VOXEL_M).query_pairs(LINK_M, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(cells), len(cells)))
    _, group = connected_components(graph, directed=False)
    return (np.bincount(group)[group] < MIN_CELLS)[inv.ravel()]


def confirmed(clouds, panos):
    """Per cloud (world points), True where another pano within SURE_M has a point within AGREE_M."""
    trees = [cKDTree(c[::EVERY]) if len(c) else None for c in clouds]
    out = [np.zeros(len(c), bool) for c in clouds]
    for k, (c, pk) in enumerate(zip(clouds, panos)):
        if pk is None or not len(c):
            continue
        for j, (t, pj) in enumerate(zip(trees, panos)):
            if j == k or t is None or pj is None or np.linalg.norm(pj - pk) > FAR_M + SURE_M:
                continue
            todo = np.flatnonzero(~out[k] & (np.linalg.norm(c - pj, axis=1) <= SURE_M))
            if len(todo):
                out[k][todo] = np.isfinite(t.query(c[todo], distance_upper_bound=AGREE_M, workers=-1)[0])
    return out


def drop_blobs(scene_dir, log=print):
    """Remove far, unconfirmed and floating points from every placed node's .ply, in place."""
    sc = scene_mod.Scene.load(scene_dir)
    nodes = [n for n in sc.nodes if n.ply and n.transform]
    if not nodes:
        return
    clouds, panos, away = [], [], []
    for n in nodes:
        p, c, world = read_node(scene_dir, n)
        T = np.asarray(n.transform, float)
        pano = None if n.position is None else T[:3, :3] @ np.asarray(n.position, float) + T[:3, 3]
        clouds.append((p, c, world))
        panos.append(pano)
        away.append(np.zeros(len(p)) if pano is None else np.linalg.norm(world - pano, axis=1))
    sure = confirmed([w for _, _, w in clouds], panos)
    unsure = 0
    for i, (p, c, world) in enumerate(clouds):
        near = (away[i] <= SURE_M) | ((away[i] <= FAR_M) & sure[i])
        unsure += int(((away[i] > SURE_M) & (away[i] <= FAR_M) & ~sure[i]).sum())
        clouds[i] = (p, c, world, near)
    loose = loose_bits(np.concatenate([w[k] for _, _, w, k in clouds]))
    start, gone = 0, 0
    for n, (p, c, _, near) in zip(nodes, clouds):
        keep = near.copy()
        keep[near] = ~loose[start:start + near.sum()]
        start += near.sum()
        gone += int((~near).sum())
        if not keep.all():
            write_ply(os.path.join(scene_dir, n.ply), p[keep], None if c is None else c[keep])
    log(f"floating bits: {gone} points past {FAR_M:g} m of their pano or past {SURE_M:g} m and no other "
        f"pano's ({unsure} of them), then {int(loose.sum())} of {len(loose)} points in floating bits removed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__.splitlines()[-1].strip())
    drop_blobs(sys.argv[1])
