"""The ground detector: which points of a cloud are the ground, slopes included.

One detector for every cloud (Google's depth, DA3's points), so every step
that needs "the ground" agrees on what it is. A point is ground when:

1. it faces up: its surface within UP_DEG of level (normals given, or
   found from the point's neighbours)
2. it is the lowest up-facing surface in its spot: per CELL_M square seen
   from above, only up-facing points within LOWEST_M of the lowest one --
   car roofs, awnings and tables stand above the ground, not on it
3. it connects to where the cameras stand: starting from the squares right
   under each camera, the ground spreads square by square into neighbours
   whose height changes by at most STEP_M. A slope, ramp or kerb changes
   gradually and connects; a planter, roof or wall top jumps and does not.

There is deliberately no "so far below the camera" rule beyond the start:
going uphill the ground rises toward camera height, and a fixed rule would
stop counting it as ground a few metres out.
"""
from collections import deque

import numpy as np
from scipy.spatial import cKDTree

UP_DEG = 20
CELL_M = 0.5
LOWEST_M = 0.3
STEP_M = 0.35
SEED_M = 2.0              # squares this close (sideways) to a camera start the spread
SEED_BELOW = (1.0, 4.0)   # ...if their ground is this far below that camera


def normals_from_neighbours(x, k=12):
    """Each point's surface normal, from the flattest direction of its k
    nearest neighbours."""
    _, nb = cKDTree(x).query(x, k=k)
    P = x[nb] - x[nb].mean(1, keepdims=True)
    return np.linalg.eigh(np.einsum("nki,nkj->nij", P, P))[1][:, :, 0]


def ground(x, cams, normals=None):
    """Boolean mask over x (world metres, y down): which points are ground.
    cams: one camera position, or several (N x 3), to start from."""
    n = normals_from_neighbours(x) if normals is None else normals
    idx = np.flatnonzero(np.abs(n[:, 1]) > np.cos(np.radians(UP_DEG)))
    c = np.floor(x[idx][:, [0, 2]] / CELL_M).astype(np.int64)
    key = (c[:, 0] + 2 ** 20) << 21 | (c[:, 1] + 2 ** 20)
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
        for i in np.flatnonzero((np.hypot(cx - cam[0], cz - cam[2]) < SEED_M)
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
