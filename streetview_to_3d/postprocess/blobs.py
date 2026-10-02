"""Remove floating bits from a placed scene: small clumps of DA3 points
touching nothing else (the remains of a half-masked lamp, a speck of sky).
First every point further than FAR_M from its own pano: DA3 at that
distance is guesswork, and a sky it did not mask comes out as a far plane
around the scene (1.9% of Stockholm's points) -- the map's land and
buildings stand there instead. Then, past SURE_M, every point no other
pano's points confirm (none within AGREE_M of it): a real wall comes out
where each pano that sees it puts it, a wall one pano made up only in its
own (Himi: walls far off, misplaced) -- where only one pano saw anything,
its points past SURE_M go, as they would under a shorter cut.

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
FAR_M = 25.0   # past this DA3 is guesswork: gone
SURE_M, AGREE_M = 12.0, 0.3   # past this from its pano, a point stays only if another pano's are this near it
EVERY = 2                     # another pano's points: every this many of them, enough to confirm by


def loose_bits(points):
    """True on points in a joined group of under MIN_CELLS cells."""
    cell = np.floor(points / VOXEL_M).astype(np.int64)
    cells, inv = np.unique(cell, axis=0, return_inverse=True)
    pairs = cKDTree(cells * VOXEL_M).query_pairs(LINK_M, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(cells), len(cells)))
    _, group = connected_components(graph, directed=False)
    return (np.bincount(group)[group] < MIN_CELLS)[inv.ravel()]


def confirmed(clouds, panos):
    """For each cloud (world points, (n, 3)) whose pano (panos, (3,) or
    None) is known, True on its points another pano's points have one of
    within AGREE_M -- only clouds whose panos could see the same place
    (2 FAR_M apart or less) asked."""
    trees = [cKDTree(c[::EVERY]) if len(c) else None for c in clouds]
    out = [np.zeros(len(c), bool) for c in clouds]
    for k, (c, pk) in enumerate(zip(clouds, panos)):
        if pk is None or not len(c):
            continue
        for j, (t, pj) in enumerate(zip(trees, panos)):
            if j == k or t is None or pj is None or np.linalg.norm(pj - pk) > 2 * FAR_M:
                continue
            todo = ~out[k]
            if todo.any():
                out[k][todo] = np.isfinite(t.query(c[todo], distance_upper_bound=AGREE_M, workers=-1)[0])
    return out


def drop_blobs(scene_dir, log=print):
    """Remove the points too far from their pano (FAR_M), those past SURE_M
    no other pano confirms, then the floating bits, from every placed
    node's .ply, in place."""
    sc = scene_mod.Scene.load(scene_dir)
    nodes = [n for n in sc.nodes if n.ply and n.transform]
    if not nodes:
        return
    clouds, panos, away = [], [], []
    for n in nodes:
        p, c = read_ply(os.path.join(scene_dir, n.ply))
        T = np.asarray(n.transform, float)
        world = p @ T[:3, :3].T + T[:3, 3]
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
