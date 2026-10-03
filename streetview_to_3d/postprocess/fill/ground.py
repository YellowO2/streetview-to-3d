"""Ground detection for a cloud (up-facing, lowest in its square, connected to a camera through
small steps), and GroundMap: one blended height map from several clouds' ground."""
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
BLEND_M = 2.0             # a camera this much further than the nearest counts 1/e as much


def square_keys(xz, cell):
    """One int64 key per point's cell x cell square; neighbours differ by (dx << 21) + dz."""
    c = np.floor(xz / cell).astype(np.int64) + 2 ** 20
    return (c[:, 0] << 21) | c[:, 1]


def normals_from_neighbours(x, k=12):
    """Each point's surface normal, from the flattest direction of its k
    nearest neighbours."""
    _, nb = cKDTree(x).query(x, k=k, workers=-1)
    P = x[nb] - x[nb].mean(1, keepdims=True)
    return np.linalg.eigh(np.einsum("nki,nkj->nij", P, P))[1][:, :, 0]


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


def blend(dist, nearest):
    """Weight of a camera dist away where the nearest is `nearest` away: 1 for the nearest,
    1/e one BLEND_M further, 0 for inf. Used for both ground height and colour."""
    near = np.where(np.isfinite(nearest), nearest, 0.0)
    return np.where(np.isfinite(dist), np.exp(-(np.where(np.isfinite(dist), dist, 0.0) - near) / BLEND_M), 0.0)


class GroundMap:
    """One ground height per cell x cell square (world metres, y down).

    xs[k] are cloud k's ground points, cams[k] its camera. Each square mixes
    the clouds' median heights by blend; `height` is that smoothed over
    smooth_m, `owner` the nearest camera's cloud."""

    def __init__(self, xs, cams, cell, smooth_m):
        who = np.concatenate([np.full(len(x), k) for k, x in enumerate(xs)])
        X = np.concatenate(xs)
        span = X[:, [0, 2]]
        self.cell, self.smooth_m = cell, smooth_m
        self.lo = span.min(0) - 2 * cell
        self.dims = np.floor((span - self.lo) / cell).astype(int).max(0) + 3
        ij = self.squares(X[:, [0, 2]])
        n, K = self.dims.prod(), len(xs)
        # one median per (square, cloud)
        key = (ij[:, 0] * self.dims[1] + ij[:, 1]) * K + who
        pair = np.unique(key)
        med = np.array(_medians(key, X[:, 1]))
        sq, k = pair // K, pair % K
        centre = (np.stack([sq // self.dims[1], sq % self.dims[1]], 1) + .5) * cell + self.lo
        dcam = np.hypot(centre[:, 0] - cams[k, 0], centre[:, 1] - cams[k, 2])
        best = np.full(n, np.inf)
        np.minimum.at(best, sq, dcam)
        w = blend(dcam, best[sq])
        den = np.bincount(sq, w, n)
        self.raw = np.where(den > 0, np.bincount(sq, w * med, n) / np.maximum(den, 1e-12), np.nan).reshape(self.dims)
        owner = np.full(n, -1)
        nearest = dcam == best[sq]
        owner[sq[nearest]] = k[nearest]
        self.owner = owner.reshape(self.dims)
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
        """Carry height into nearby empty squares, wider each pass, so holes can be filled."""
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
        """(heights at xz, which had any): bilinear, ignoring empty corners."""
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
        """(points, owner cloud of each) every step metres over the marked squares, on height."""
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
