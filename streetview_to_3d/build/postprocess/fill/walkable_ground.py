"""Which of each placed cloud's points are ground its pano calls walkable.

Ground for a cloud: up-facing points, lowest in their square, connected to a camera through
small steps (_ground); walkable where its pano's class map says road, sidewalk or terrain.
"""
from collections import deque

import numpy as np

from streetview_to_3d.build.postprocess.tools.ground_maths import normals_from_neighbours, square_keys

DA3_SEED_M = 7.0        # the ground spread may start this far out (DA3's blind disc reaches ~4.5 m)
LOW_M = 1.0             # only points this near the lowest in their square can be ground
WALKABLE = ("road", "sidewalk", "terrain")      # the masker's classes (models.Segmenter)
# _ground: how up a point faces, its squares, how near the lowest, the steps it spreads by
UP_DEG = 20
CELL_M = 0.5
LOWEST_M = 0.3
STEP_M = 0.35
SEED_M = 2.0              # squares this close (sideways) to a camera start the spread
SEED_BELOW = (1.0, 4.0)   # ...if their ground is this far below that camera


def walkable_ground(clouds, cameras):
    """Per cloud (world Clouds, one per camera), True on its walkable ground."""
    points = [c.points for c in clouds]
    return _grounds(points, np.array([c.centre for c in cameras]).reshape(-1, 3),
                    [cam.classed(x, WALKABLE, True) for x, cam in zip(points, cameras)])


def _grounds(clouds, cams, walkable=None):
    """Which of each cloud's points (world, y down) are walkable ground; walkable[k] masks cloud k (None: all)."""
    G = [_ground(x, c, normals=_low_normals(x), seed_m=DA3_SEED_M) for x, c in zip(clouds, cams)]
    return G if walkable is None else [g & w for g, w in zip(G, walkable)]


def _low_normals(x):
    """Normals of the points within LOW_M of the lowest in their ground square; zero
    (never ground) for the rest, which saves computing most normals."""
    _, inv = np.unique(square_keys(x[:, [0, 2]], CELL_M), return_inverse=True)
    low = np.full(inv.max() + 1, -np.inf)
    np.maximum.at(low, inv, x[:, 1])                  # y is down: the largest y is the lowest
    near = x[:, 1] > low[inv] - LOW_M
    n = np.zeros_like(x)
    n[near] = normals_from_neighbours(x[near])
    return n


def _ground(x, cams, normals=None, seed_m=SEED_M):
    """Which points of x (world metres, y down) are ground.

    cams: one or more (N, 3) camera positions to spread from; seed_m: how far
    out from a camera the spread may start."""
    n = normals_from_neighbours(x) if normals is None else normals
    idx = np.flatnonzero(np.abs(n[:, 1]) > np.cos(np.radians(UP_DEG)))
    key = square_keys(x[idx][:, [0, 2]], CELL_M)
    uk, inv = np.unique(key, return_inverse=True)
    low = np.full(len(uk), -np.inf)
    np.maximum.at(low, inv, x[idx, 1])                  # y is down: the largest y is the lowest
    keep = x[idx, 1] > low[inv] - LOWEST_M
    idx, inv = idx[keep], inv[keep]
    h = np.full(len(uk), np.nan)                        # each square's ground height
    np.fmax.at(h, inv, x[idx, 1])

    cx = ((uk >> 21) - 2 ** 20 + .5) * CELL_M
    cz = ((uk & (2 ** 21 - 1)) - 2 ** 20 + .5) * CELL_M
    where = {k: i for i, k in enumerate(uk.tolist())}
    ok = np.zeros(len(uk), bool)
    todo = deque()
    for cam in np.atleast_2d(cams):
        below = h - cam[1]
        for i in np.flatnonzero((np.hypot(cx - cam[0], cz - cam[2]) < seed_m)
                                & (below > SEED_BELOW[0]) & (below < SEED_BELOW[1])):
            if not ok[i]:
                ok[i] = True
                todo.append(i)
    while todo:
        i = todo.popleft()
        k = int(uk[i])
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                j = where.get(k + (dx << 21) + dz)
                if j is not None and not ok[j] and abs(h[j] - h[i]) < STEP_M:
                    ok[j] = True
                    todo.append(j)
    out = np.zeros(len(x), bool)
    out[idx[ok[inv]]] = True
    return out
