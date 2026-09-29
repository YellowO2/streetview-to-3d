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

GroundMap turns several clouds' ground into ONE ground: each square takes
the ground of the cloud whose camera is nearest (it sees that spot best),
lightly smoothed, and can be laid out as an even grid -- no stacked floors
where clouds disagree by a few cm.
"""
from collections import deque

import numpy as np
from scipy.ndimage import gaussian_filter
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


def ground(x, cams, normals=None, seed_m=SEED_M):
    """Boolean mask over x (world metres, y down): which points are ground.
    cams: one camera position, or several (N x 3), to start from. seed_m:
    how far out from a camera the spread may start (DA3 sees nothing within
    ~4 m of its own camera, so it needs more than Google's depth does)."""
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


class GroundMap:
    """One ground height per cell x cell square (world metres, y down).

    xs[k] are cloud k's ground points and cams[k] its camera; each square
    takes the median height of the cloud whose camera is nearest to it, and
    `height` is that, lightly smoothed (smooth_m), ignoring empty squares.
    """

    def __init__(self, xs, cams, cell, smooth_m):
        who = np.concatenate([np.full(len(x), k) for k, x in enumerate(xs)])
        X = np.concatenate(xs)
        span = X[:, [0, 2]]
        self.cell, self.smooth_m = cell, smooth_m
        self.lo = span.min(0) - 2 * cell
        self.dims = np.floor((span - self.lo) / cell).astype(int).max(0) + 3
        ij = self.squares(X[:, [0, 2]])
        centre = (ij + .5) * cell + self.lo
        dcam = np.hypot(centre[:, 0] - cams[who, 0], centre[:, 1] - cams[who, 2])
        flat = ij[:, 0] * self.dims[1] + ij[:, 1]
        order = np.lexsort((dcam, flat))
        fs = flat[order]
        first = np.r_[True, fs[1:] != fs[:-1]]
        win = np.full(self.dims.prod(), -1)
        win[fs[first]] = who[order][first]
        mine = win[flat] == who
        self.raw = np.full(self.dims.prod(), np.nan)
        self.raw[np.unique(flat[mine])] = _medians(flat[mine], X[mine, 1])
        self.raw = self.raw.reshape(self.dims)
        self.owner = win.reshape(self.dims)
        self.height = self.smoothed()

    def squares(self, xz):
        return np.floor((xz - self.lo) / self.cell).astype(int)

    @property
    def have(self):
        """Squares some cloud saw ground in."""
        return np.isfinite(self.raw)

    def smoothed(self):
        have = self.have
        sigma = self.smooth_m / self.cell
        num = gaussian_filter(np.where(have, self.raw, 0), sigma)
        den = gaussian_filter(have.astype(float), sigma)
        return np.where(den > 1e-3, num / np.maximum(den, 1e-9), np.nan)

    def spread(self):
        """height carried into empty squares nearby, wider and wider, so a
        hole can be laid out on it."""
        h = self.height.copy()
        known = np.isfinite(h)
        for s in (1, 2, 4, 8):
            num = gaussian_filter(np.where(known, h, 0), s)
            den = gaussian_filter(known.astype(float), s)
            grow = ~known & (den > 0.05)
            h[grow] = num[grow] / den[grow]
            known |= grow
        self.height = h

    def at(self, xz):
        """(heights at xz, which had any): bilinear on height, empty
        corners left out."""
        g = (xz - self.lo) / self.cell - .5
        i0 = np.clip(np.floor(g).astype(int), 0, self.dims - 2)
        f = g - i0
        val = np.zeros(len(xz))
        wsum = np.zeros(len(xz))
        for a in (0, 1):
            for b in (0, 1):
                c = self.height[i0[:, 0] + a, i0[:, 1] + b]
                w = (f[:, 0] if a else 1 - f[:, 0]) * (f[:, 1] if b else 1 - f[:, 1])
                w = np.where(np.isfinite(c), w, 0)
                val += w * np.nan_to_num(c)
                wsum += w
        ok = wsum > 0
        return np.where(ok, val / np.maximum(wsum, 1e-12), np.nan), ok

    def grid(self, squares, step):
        """(points, owner cloud of each) laid out every `step` metres over
        the squares marked True, on height."""
        per = int(round(self.cell / step))
        sq = np.argwhere(squares)
        off = (np.arange(per) + .5) * step
        ox, oz = np.meshgrid(off, off, indexing="ij")
        xz = (self.lo + sq[:, None, :] * self.cell + np.stack([ox.ravel(), oz.ravel()], 1)[None]).reshape(-1, 2)
        owner = np.repeat(self.owner[sq[:, 0], sq[:, 1]], per * per)
        y, ok = self.at(xz)
        return np.stack([xz[ok, 0], y[ok], xz[ok, 1]], 1), owner[ok]


def _medians(flat, v):
    o = np.argsort(flat)
    _, start = np.unique(flat[o], return_index=True)
    return [np.median(s) for s in np.split(v[o], start[1:])]
