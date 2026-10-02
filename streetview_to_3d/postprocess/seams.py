"""Where the map (terrain.py) meets the scene: faded in, not cut.

The scene always wins -- the map only fills what it lacks (terrain.build,
buildings.seam) -- but a hard edge between two sources shows even when they
agree to a metre. So across a band past the scene's edge the map, at full
density (thinning it to interleave left a sparse strip, DA3 thinning out at
its edges too):

  - takes the scene's ground's colour (tint), mixed back to its own over
    the band: satellite colour is darker and bluer than the panos'
  - meets the scene's ground (meet): its height pulled onto it at the
    nearest edge, less so further out

and then, once everything -- land, roads, buildings -- stands where it
will, every point of it within BLEND_M of DA3's turns into them (toward):
their colour, as few as they are, and (the viewer) their look. One rule
for all of it, nothing moved after.

The scene's ground is the fill's (SceneGround): found once, from what the
panos' class maps call road, sidewalk or terrain, and read here as it is
-- never guessed again from the lowest of the scene's points, which at an
edge with only a wall's or a tree's took them for the ground (Lake Como:
a road pulled 7 m up to one, dropped 7 m at the scene's edge).

Everything is aligned before it is faded (terrain.from_panos,
buildings.fit_to_scene): a fade only hides a seam where the two already
roughly agree; across a real offset it would only blur a double wall.
"""
import os

import numpy as np
from scipy.ndimage import distance_transform_edt

FILENAME = "ground.npz"   # beside scene.json: the fill's ground, as SceneGround keeps it
CELL_M = 0.5              # the fill's own squares (fill.one_ground.CELL_M)
PAD_M = 50.0              # the grid this far past the ground: the widest band asking of it (terrain.ROAD_MEET_M)
BLEND_M = 3.0             # the map's points this near DA3's turn into them: their colour, their look (toward)
LOCAL_K = 6               # DA3's spacing somewhere: how far its LOCAL_K-th nearest point is
GREY_C, GREY_N = 8.0, 200   # the satellite's tint: from ground the panos see this grey (Lab chroma), this much of it
LIGHT_RANGE = (0.7, 1.6)    # its lightness varied at most this much more or less
BOOST_MAX, VIVID_C = 2.5, 60.0  # its colourfulness: at most this many times, none for colours this vivid (Lab chroma)


def ramp(t):
    """0 below 0, 1 above 1, smooth between."""
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def tint(cols, near, dist, cut_m, band_m, strength):
    """cols mixed toward near (the scene's colour there) by strength at the
    cut, back to their own over band_m."""
    w = strength * (1 - ramp((dist - cut_m) / band_m))
    w = np.where(np.isnan(near).any(1), 0.0, w)[:, None]
    return cols * (1 - w) + np.nan_to_num(near) * w


def toward(pts, cols, gap, tree, da3_cols, every=1):
    """(cols, near, keep): the map's points pts -- buildings', roads', the
    land's corners -- turning into DA3's as they come within BLEND_M of
    them, once everything stands where it will: near 0 that far off or
    more, 1 on one; their colour mixed toward their nearest DA3 point's by
    it, and as few of them kept as DA3's are there (keep: each its own
    chance, fixed in the world, from all of them far off to as sparse as
    DA3's points on one; never more than were; all of them with no gap, a
    mesh's corners). tree: a cKDTree of DA3's points (every every-th of
    them), da3_cols their colours. The viewer turns their look into DA3's
    points' by near too (effects/blocks.js)."""
    if tree is None or not len(pts):
        return cols, np.zeros(len(pts)), np.ones(len(pts), bool)
    d, k = tree.query(pts, distance_upper_bound=BLEND_M, workers=-1)
    near = 1 - ramp(d / BLEND_M)                               # d is inf past BLEND_M: 0
    k = np.minimum(k, len(da3_cols) - 1)
    cols = cols + (da3_cols[k] - cols) * near[:, None]
    # DA3's spacing round its nearest point: its LOCAL_K-th neighbour's distance, as
    # if all its points were there, not every every-th
    keep = np.ones(len(pts), bool)
    close = np.flatnonzero(near > 0)
    if len(close) and gap is not None:
        far = tree.query(tree.data[k[close]], k=LOCAL_K + 1, workers=-1)[0][:, -1]
        spacing = far / np.sqrt(LOCAL_K / np.pi) / np.sqrt(every)
        share = np.minimum(1, (gap[close] / np.maximum(spacing, 1e-6)) ** 2)    # of these, as many as DA3's
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
    """f(colours) -> colours: the satellite's colours as the panos see them,
    learnt where both see the scene's own ground (ground: SceneGround;
    satellite(east/north (n, 2)) -> (n, 3), NaN where it has none). Seen
    through the air a satellite's are paler and bluer; undone in Lab, so a
    grey stays grey:

      - tint: the satellite's cast, from the ground the panos see grey (a
        road, a pavement: GREY_C or less colourful), at least GREY_N of it
      - lightness: as bright and as varied as the panos' ground, never
        darker than the satellite's (the panos see their ground in the shade
        of what stands round it; the world should not look the gloomier)
      - colourfulness: as the panos', up to BOOST_MAX times -- the least
        colourful the most, none past VIVID_C (a red roof stays red, not neon)

    Only ground is seen by both, so only what ground cannot mistake is
    learnt: no hue (grass greener than the satellite has it is the grass,
    not the air). Nothing to learn from: as they are."""
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
    """Heights h pulled onto the scene's ground at the nearest edge, fully
    at it, not at all band_m out -- unless they differ by more than max_m
    (a road: another road, passing over or under)."""
    w = 1 - ramp(dist / band_m)
    w = np.where(np.isnan(ground) | (np.abs(np.nan_to_num(ground) - h) > max_m), 0.0, w)
    return h + (np.nan_to_num(ground) - h) * w


class SceneGround:
    """The scene's own ground as the fill laid it (fill.one_ground), seen
    from above in CELL_M squares: each one's height (metres up) and colour,
    NaN where the scene has no ground, and how much of it its panos call
    road (road: 0-1, None if not known). For any east/north point: how far
    it is from that ground (0 on it), and the height and colour of the
    nearest square of it (at). The fill saves it beside the scene (save); a
    scene without one has no ground of its own, and nothing is near it."""

    def __init__(self, lo, height, colour, road=None):
        self.lo, self.height, self.colour, self.road = np.asarray(lo, int), height, colour, road
        have = np.isfinite(height)
        self.dist, self.nearest = (distance_transform_edt(~have, return_indices=True) if have.any()
                                   else (None, None))

    @classmethod
    def from_points(cls, pts, cols, road=None):
        """The ground's points (world: x east, y down, z north) and their
        colours (0-1), and whether each is road (road (n,): bool, or None),
        averaged in each square."""
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
        """How much of the ground at each east/north point its panos call
        road (0-1); NaN off it, or not known."""
        out = np.full(len(xy), np.nan)
        if self.road is None or self.dist is None:
            return out
        c = np.floor(np.asarray(xy) / CELL_M).astype(int) - self.lo
        inside = ((c >= 0) & (c < self.height.shape)).all(1)
        out[inside] = self.road[tuple(c[inside].T)]
        return out
