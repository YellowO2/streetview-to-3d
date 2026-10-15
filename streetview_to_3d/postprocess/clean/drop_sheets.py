"""Remove the loose sheets of points hanging off a placed scene's surfaces.

A loose sheet: DA3 points whose K-th neighbour is SPARSE times farther than on the
densest surface nearby. Runs scene-wide after drop_blobs.py, before the fill.

    python -m streetview_to_3d.postprocess.clean.drop_sheets SCENE_DIR
"""
import logging
import sys

import numpy as np
from scipy.spatial import cKDTree

from streetview_to_3d.common.Cloud import Cloud

logger = logging.getLogger(__name__)

K, SPARSE, CELL_M, SHARE = 8, 2.5, 0.5, 0.2


def _sparse(points):
    """True on points far sparser than the densest near them."""
    if len(points) <= K:
        return np.zeros(len(points), bool)
    gap = cKDTree(points).query(points, k=K + 1, workers=-1)[0][:, -1]
    # each cell's densest (its SHARE-th smallest gap), then the densest of it and its neighbours
    X, Y = 1_000_003 * 1_000_033, 1_000_033                  # cell key strides: neighbours are a step off
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


def drop_sheets(sc):
    """Remove the loose sheets from every placed node's cloud; returns the scene."""
    nodes = sc.placed()
    if not nodes:
        return sc
    raw, world = [], []
    for n in nodes:
        raw.append((n.cloud.points, n.cloud.colours))
        world.append(n.cloud.world(n.transform))
    loose = _sparse(np.concatenate(world))
    start = 0
    for n, (p, c) in zip(nodes, raw):
        keep = ~loose[start:start + len(p)]
        start += len(p)
        if not keep.all():
            n.cloud = Cloud(p[keep], c[keep])
    logger.info(f"loose sheets: {int(loose.sum())} of {len(loose)} points removed")
    return sc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) != 2:
        sys.exit(__doc__.splitlines()[-1].strip())
    from streetview_to_3d.load import load
    from streetview_to_3d.save import save
    save(drop_sheets(load(sys.argv[1])), None, None, sys.argv[1])
