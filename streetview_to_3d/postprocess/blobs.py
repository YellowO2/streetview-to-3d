"""Remove floating bits from a placed scene: small clumps of DA3 points
touching nothing else (the remains of a half-masked lamp, a speck of sky).

Every node's points, in the world, go into VOXEL_M cells; cells within
LINK_M of each other are joined, and a joined group under MIN_CELLS cells
is a floating bit. Across the whole scene, not per node: a clump in one
node's cloud that touches another node's surface is kept.

On NTU (15 nodes, 2.5M points): 1664 bits, 0.7% of the points, all of them
the kind of junk meant; the eight loose pieces above the limit were real
(ground patches, a far building, 2-17 m across) and stay. About 5 s.

    python -m streetview_to_3d.postprocess.blobs SCENE_DIR
"""
import os
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from streetview_to_3d import scene as scene_mod
from streetview_to_3d.postprocess.ply_io import read_ply, write_ply

VOXEL_M, LINK_M, MIN_CELLS = 0.1, 0.25, 300


def loose_bits(points):
    """True on points in a joined group of under MIN_CELLS cells."""
    cell = np.floor(points / VOXEL_M).astype(np.int64)
    cells, inv = np.unique(cell, axis=0, return_inverse=True)
    pairs = cKDTree(cells * VOXEL_M).query_pairs(LINK_M, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(cells), len(cells)))
    _, group = connected_components(graph, directed=False)
    return (np.bincount(group)[group] < MIN_CELLS)[inv.ravel()]


def drop_blobs(scene_dir, log=print):
    """Remove the floating bits from every placed node's .ply, in place."""
    sc = scene_mod.Scene.load(scene_dir)
    nodes = [n for n in sc.nodes if n.ply and n.transform]
    if not nodes:
        return
    clouds = []
    for n in nodes:
        p, c = read_ply(os.path.join(scene_dir, n.ply))
        T = np.asarray(n.transform, float)
        clouds.append((p, c, p @ T[:3, :3].T + T[:3, 3]))
    loose = loose_bits(np.concatenate([w for _, _, w in clouds]))
    start = 0
    for n, (p, c, _) in zip(nodes, clouds):
        drop = loose[start:start + len(p)]
        start += len(p)
        if drop.any():
            write_ply(os.path.join(scene_dir, n.ply), p[~drop], None if c is None else c[~drop])
    log(f"floating bits: {int(loose.sum())} of {len(loose)} points removed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__.splitlines()[-1].strip())
    drop_blobs(sys.argv[1])
