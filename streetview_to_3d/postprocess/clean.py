"""Remove loose sheets from a placed scene: DA3's points far sparser than
the densest of what is near them -- the thin waves hanging off a wall,
which blobs.py keeps (they touch it).

A point goes if its K-th nearest point is SPARSE times as far off as on
the densest surface near it (its own cell or one beside it, CELL_M across;
a cell's densest: its SHARE-th densest point). Surfaces uniformly sparse
(far off, at a slant) keep all their points, their neighbours as sparse.
A sheet only a few of the wall's own spacings off it borrows the wall's
points as its neighbours, and stays.

Across the whole scene, not per node. Points are only ever removed. After
blobs.py, before the fill, so its ground is laid on what is left.

    python -m streetview_to_3d.postprocess.clean SCENE_DIR
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

from streetview_to_3d import scene as scene_mod
from streetview_to_3d.postprocess.ply_io import read_ply, write_ply

K, SPARSE, CELL_M, SHARE = 8, 3.0, 0.5, 0.2


def sparse(points):
    """True on points far sparser than the densest near them."""
    if len(points) <= K:
        return np.zeros(len(points), bool)
    gap = cKDTree(points).query(points, k=K + 1, workers=-1)[0][:, -1]
    # each cell's densest (its SHARE-th smallest gap), then the densest of it and its neighbours
    X, Y = 1_000_003 * 1_000_033, 1_000_033                  # a cell's key: one number, its neighbours' a step off
    cell = np.floor(points / CELL_M).astype(np.int64)
    keys, which = np.unique(cell @ [X, Y, 1], return_inverse=True)
    which = which.ravel()
    order = np.lexsort((gap, which))
    start = np.searchsorted(which[order], np.arange(len(keys)))
    count = np.bincount(which, minlength=len(keys))
    dense = gap[order[start + (count * SHARE).astype(int)]]
    best = dense.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                k = keys + dx * X + dy * Y + dz
                at = np.minimum(np.searchsorted(keys, k), len(keys) - 1)
                hit = keys[at] == k
                best[hit] = np.minimum(best[hit], dense[at[hit]])
    return gap > SPARSE * best[which]


def clean(scene_dir, log=print):
    """Remove the loose sheets from every placed node's .ply, in place."""
    sc = scene_mod.Scene.load(scene_dir)
    nodes = [n for n in sc.nodes if n.ply and n.transform]
    if not nodes:
        return
    raw, world = [], []
    for n in nodes:
        p, c = read_ply(os.path.join(scene_dir, n.ply))
        T = np.asarray(n.transform, float)
        raw.append((p, c))
        world.append(p @ T[:3, :3].T + T[:3, 3])
    loose = sparse(np.concatenate(world))
    start = 0
    for n, (p, c) in zip(nodes, raw):
        keep = ~loose[start:start + len(p)]
        start += len(p)
        if not keep.all():
            write_ply(os.path.join(scene_dir, n.ply), p[keep], None if c is None else c[keep])
    log(f"loose sheets: {int(loose.sum())} of {len(loose)} points removed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__.splitlines()[-1].strip())
    clean(sys.argv[1])
