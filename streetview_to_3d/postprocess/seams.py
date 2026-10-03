"""Where the map meets the scene: tint and meet blend colour and height at the scene ground's
edge, toward turns map points near DA3's into them; SceneGround is the fill's ground (ground.npz)."""
import os

import numpy as np
from scipy.ndimage import distance_transform_edt

FILENAME = "ground.npz"
CELL_M = 0.5              # the fill's ground squares (also fill.one_ground's grid)
PAD_M = 50.0              # grid margin past the ground: the widest band that asks (terrain.ROAD_MEET_M)
BLEND_M = 3.0             # map points this near DA3's turn into them (toward)
LOCAL_K = 6               # DA3's local spacing: distance to its LOCAL_K-th nearest point
GREY_C, GREY_N = 8.0, 200   # satellite tint learnt from ground under this Lab chroma, at least this many squares
LIGHT_RANGE = (0.7, 1.6)    # lightness contrast scale limits
BOOST_MAX, VIVID_C = 2.5, 60.0  # max chroma boost; no boost at this Lab chroma


def ramp(t):
    """0 below 0, 1 above 1, smooth between."""
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def tint(cols, near, dist, cut_m, band_m, strength):
    """cols mixed toward near by strength at cut_m, fading back to their own over band_m."""
    w = strength * (1 - ramp((dist - cut_m) / band_m))
    w = np.where(np.isnan(near).any(1), 0.0, w)[:, None]
    return cols * (1 - w) + np.nan_to_num(near) * w


def toward(pts, cols, gap, tree, da3_cols, every=1):
    """(cols, near, keep) for map points pts approaching DA3's points.

    near: 1 on a DA3 point, 0 at BLEND_M or more. Colours mix toward the
    nearest DA3 point's by near; keep thins points toward DA3's density
    (all kept when gap is None, e.g. mesh corners). tree: a cKDTree of every
    every-th DA3 point, da3_cols their colours."""
    if tree is None or not len(pts):
        return cols, np.zeros(len(pts)), np.ones(len(pts), bool)
    d, k = tree.query(pts, distance_upper_bound=BLEND_M, workers=-1)
    near = 1 - ramp(d / BLEND_M)                               # d is inf past BLEND_M
    k = np.minimum(k, len(da3_cols) - 1)
    cols = cols + (da3_cols[k] - cols) * near[:, None]
    # DA3's local spacing, corrected for the tree holding every every-th point
    keep = np.ones(len(pts), bool)
    close = np.flatnonzero(near > 0)
    if len(close) and gap is not None:
        far = tree.query(tree.data[k[close]], k=LOCAL_K + 1, workers=-1)[0][:, -1]
        spacing = far / np.sqrt(LOCAL_K / np.pi) / np.sqrt(every)
        share = np.minimum(1, (gap[close] / np.maximum(spacing, 1e-6)) ** 2)
        chance = np.sin(pts[close] @ [12.9898, 78.233, 37.719]) * 43758.5453 % 1
        keep[close] = chance < 1 - near[close] * (1 - share)
    return cols, near, keep


def lab(rgb):
    """sRGB (n, 3) 0-1 -> CIE Lab (D65)."""
    c = np.clip(rgb, 0, 1)
    xyz = np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4) @ _XYZ.T / _WHITE
    f = np.where(xyz > .008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.c_[116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])]


def rgb(lab_):
    """CIE Lab (n, 3) -> sRGB 0-1, clipped."""
    fy = (lab_[:, 0] + 16) / 116
    f = np.c_[fy + lab_[:, 1] / 500, fy, fy - lab_[:, 2] / 200]
    lin = np.where(f > .2069, f ** 3, (f - 16 / 116) / 7.787) * _WHITE @ np.linalg.inv(_XYZ).T
    return np.clip(np.where(lin <= .0031308, lin * 12.92, 1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - .055), 0, 1)


_XYZ = np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]])
_WHITE = np.array([.9505, 1, 1.089])


def unhazed(ground, satellite):
    """f(colours) -> colours: satellite colours corrected to look like the panos'.

    Learnt in Lab where both see the scene's ground (ground: SceneGround;
    satellite(east/north) -> (n, 3), NaN where missing): the colour cast from
    grey ground, lightness matched but never darkened, and chroma boosted up
    to BOOST_MAX (less for vivid colours). Hue is not changed. Identity if
    there is too little to learn from."""
    have = np.argwhere(np.isfinite(ground.height)) if ground.height.size else np.zeros((0, 2), int)
    if len(have) < GREY_N:
        return lambda cols: cols
    pano = ground.colour[tuple(have.T)]
    sat = satellite((have + ground.lo + .5) * CELL_M)
    ok = np.isfinite(sat).all(1) & np.isfinite(pano).all(1)
    if ok.sum() < GREY_N:
        return lambda cols: cols
    P, S = lab(pano[ok]), lab(sat[ok])
    grey = np.hypot(P[:, 1], P[:, 2]) < GREY_C
    tint = (P[grey, 1:] - S[grey, 1:]).mean(0) if grey.sum() >= GREY_N else np.zeros(2)
    S[:, 1:] += tint
    k = np.clip(P[:, 0].std() / max(S[:, 0].std(), 1e-6), *LIGHT_RANGE)
    o = P[:, 0].mean() - k * S[:, 0].mean()
    boost = np.clip(np.hypot(P[:, 1], P[:, 2]).mean() / max(np.hypot(S[:, 1], S[:, 2]).mean(), 1e-6), 1, BOOST_MAX)

    def fix(cols):
        cols = np.asarray(cols, float)
        out = cols.copy()
        good = np.isfinite(cols).all(1)
        L = lab(cols[good])
        ab = L[:, 1:] + tint
        c = np.hypot(ab[:, 0], ab[:, 1])
        ab *= (boost / (1 + (boost - 1) * np.minimum(c / VIVID_C, 1)))[:, None]
        out[good] = rgb(np.c_[np.clip(np.maximum(k * L[:, 0] + o, L[:, 0]), 0, 100), ab])
        return out
    return fix


def meet(h, ground, dist, band_m, max_m=np.inf):
    """Heights h pulled onto the scene's ground: fully at its edge, not at all band_m out,
    and not where they differ by more than max_m."""
    w = 1 - ramp(dist / band_m)
    w = np.where(np.isnan(ground) | (np.abs(np.nan_to_num(ground) - h) > max_m), 0.0, w)
    return h + (np.nan_to_num(ground) - h) * w


class SceneGround:
    """The scene's ground as the fill laid it, in CELL_M squares seen from above.

    height (m up) and colour per square, NaN where there is none; road: the
    share its panos call road (None if unknown). at() gives distance to the
    ground and the nearest square's height and colour."""

    def __init__(self, lo, height, colour, road=None):
        self.lo, self.height, self.colour, self.road = np.asarray(lo, int), height, colour, road
        have = np.isfinite(height)
        self.dist, self.nearest = (distance_transform_edt(~have, return_indices=True) if have.any()
                                   else (None, None))

    @classmethod
    def from_points(cls, pts, cols, road=None):
        """Average ground points (world, y down), colours and road flags per square."""
        if not len(pts):
            return cls.none()
        ij = np.floor(pts[:, [0, 2]] / CELL_M).astype(int)
        pad = int(np.ceil(PAD_M / CELL_M))
        lo = ij.min(0) - pad
        shape = tuple(ij.max(0) - lo + pad + 1)
        at = tuple((ij - lo).T)
        n = np.zeros(shape)
        np.add.at(n, at, 1)
        h = np.zeros(shape)
        np.add.at(h, at, -pts[:, 1])
        c = np.zeros(shape + (3,))
        for j in range(3):
            np.add.at(c[..., j], at, cols[:, j])
        have = n > 0
        r = None
        if road is not None:
            r = np.zeros(shape)
            np.add.at(r, at, np.asarray(road, float))
        with np.errstate(invalid="ignore", divide="ignore"):
            return cls(lo, np.where(have, h / n, np.nan), np.where(have[..., None], c / n[..., None], np.nan),
                       None if r is None else np.where(have, r / n, np.nan))

    @classmethod
    def none(cls):
        return cls(np.zeros(2, int), np.zeros((0, 0)), np.zeros((0, 0, 3)))

    def save(self, scene_dir):
        road = {} if self.road is None else {"road": self.road.astype(np.float32)}
        np.savez_compressed(os.path.join(scene_dir, FILENAME), lo=self.lo,
                            height=self.height.astype(np.float32), colour=self.colour.astype(np.float32), **road)
        return FILENAME

    @classmethod
    def load(cls, scene_dir):
        path = os.path.join(scene_dir, FILENAME)
        if not os.path.exists(path):
            return cls.none()
        with np.load(path) as f:
            return cls(f["lo"], f["height"].astype(float), f["colour"].astype(float),
                       f["road"].astype(float) if "road" in f else None)

    def at(self, xy):
        """(distance m, ground height m, ground colour (n, 3)) at east/north
        points; past the grid: infinitely far, no ground."""
        dist = np.full(len(xy), np.inf)
        ground = np.full(len(xy), np.nan)
        colour = np.full((len(xy), 3), np.nan)
        if self.dist is None:
            return dist, ground, colour
        c = np.floor(np.asarray(xy) / CELL_M).astype(int) - self.lo
        inside = ((c >= 0) & (c < self.height.shape)).all(1)
        ci = tuple(c[inside].T)
        dist[inside] = self.dist[ci] * CELL_M
        ni, nj = self.nearest[0][ci], self.nearest[1][ci]
        ground[inside] = self.height[ni, nj]
        colour[inside] = self.colour[ni, nj]
        return dist, ground, colour

    def covers(self, xy):
        """Whether the scene has ground of its own at each east/north point."""
        return self.at(xy)[0] == 0

    def road_at(self, xy):
        """Road share (0-1) of the ground at each east/north point; NaN off it or unknown."""
        out = np.full(len(xy), np.nan)
        if self.road is None or self.dist is None:
            return out
        c = np.floor(np.asarray(xy) / CELL_M).astype(int) - self.lo
        inside = ((c >= 0) & (c < self.height.shape)).all(1)
        out[inside] = self.road[tuple(c[inside].T)]
        return out
