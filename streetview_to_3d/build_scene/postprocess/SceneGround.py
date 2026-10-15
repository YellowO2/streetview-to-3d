"""The scene's ground as the fill laid it, seen from above (ground.npz)."""
import os

import numpy as np
from scipy.ndimage import distance_transform_edt

FILENAME = "ground.npz"
CELL_M = 0.5              # the fill's ground squares (also fill.one_ground's grid)
PAD_M = 50.0              # grid margin past the ground: the widest band that asks (world/roads/RoadHeight.py ROAD_MEET_M)


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
        self.depth = distance_transform_edt(have) if have.any() else None

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

    def inside(self, xy):
        """Metres in from the edge of the scene's own ground at each east/north point; 0 off it."""
        out = np.zeros(len(xy))
        if self.depth is None:
            return out
        c = np.floor(np.asarray(xy) / CELL_M).astype(int) - self.lo
        ok = ((c >= 0) & (c < self.height.shape)).all(1)
        out[ok] = self.depth[tuple(c[ok].T)] * CELL_M
        return out

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
