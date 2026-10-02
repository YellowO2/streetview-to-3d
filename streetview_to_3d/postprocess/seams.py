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
    NaN where the scene has no ground. For any east/north point: how far it
    is from that ground (0 on it), and the height and colour of the nearest
    square of it (at). The fill saves it beside the scene (save); a scene
    without one has no ground of its own, and nothing is near it."""

    def __init__(self, lo, height, colour):
        self.lo, self.height, self.colour = np.asarray(lo, int), height, colour
        have = np.isfinite(height)
        self.dist, self.nearest = (distance_transform_edt(~have, return_indices=True) if have.any()
                                   else (None, None))

    @classmethod
    def from_points(cls, pts, cols):
        """The ground's points (world: x east, y down, z north) and their
        colours (0-1), averaged in each square."""
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
        with np.errstate(invalid="ignore", divide="ignore"):
            return cls(lo, np.where(have, h / n, np.nan), np.where(have[..., None], c / n[..., None], np.nan))

    @classmethod
    def none(cls):
        return cls(np.zeros(2, int), np.zeros((0, 0)), np.zeros((0, 0, 3)))

    def save(self, scene_dir):
        np.savez_compressed(os.path.join(scene_dir, FILENAME), lo=self.lo,
                            height=self.height.astype(np.float32), colour=self.colour.astype(np.float32))
        return FILENAME

    @classmethod
    def load(cls, scene_dir):
        path = os.path.join(scene_dir, FILENAME)
        if not os.path.exists(path):
            return cls.none()
        with np.load(path) as f:
            return cls(f["lo"], f["height"].astype(float), f["colour"].astype(float))

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
