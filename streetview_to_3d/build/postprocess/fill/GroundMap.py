"""Blend several clouds' ground into one height map."""
import numpy as np
from scipy.ndimage import gaussian_filter

from streetview_to_3d.build.postprocess.tools.ground_maths import blend


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
