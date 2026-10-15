"""Find which of a cloud's points are its walkable ground.

Ground detection for a cloud: up-facing points, lowest in their square, connected to a
camera through small steps.
"""
from collections import deque

import numpy as np

from streetview_to_3d.postprocess.fill.square_keys import square_keys
from streetview_to_3d.postprocess.normals_from_neighbours import normals_from_neighbours

UP_DEG = 20
CELL_M = 0.5
LOWEST_M = 0.3
STEP_M = 0.35
SEED_M = 2.0              # squares this close (sideways) to a camera start the spread
SEED_BELOW = (1.0, 4.0)   # ...if their ground is this far below that camera


def ground(x, cams, normals=None, seed_m=SEED_M):
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
