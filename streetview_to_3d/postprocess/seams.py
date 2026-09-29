"""Where the map (terrain.py) meets the scene: faded in, not cut.

The scene always wins -- the map only fills what it lacks (terrain.build,
buildings.seam) -- but a hard edge between two sources shows even when they
agree to a metre. So across a band past the scene's edge the map, at full
density (thinning it to interleave left a sparse strip, DA3 thinning out at
its edges too):

  - takes the scene's colour (tint), mixed back to its own over the band:
    satellite colour is darker and bluer than the panos'
  - meets the scene's ground (meet): its height pulled onto the scene's
    lowest points at the nearest edge, less so further out -- unless they
    differ by more than MEET_MAX_M, which is a tree's canopy, not ground

Everything is aligned before it is faded (terrain.correction,
buildings.fit_to_scene): a fade only hides a seam where the two already
roughly agree; across a real offset it would only blur a double wall.
"""
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

CELL_M, GROW, MIN_POINTS = 1.0, 1, 3   # the footprint: cells holding MIN_POINTS, grown GROW cells
PAD_M = 20.0                           # past this beyond the footprint, nothing is near it
LOW_M = 0.5                            # a cell's ground colour: its points this near its lowest
MEET_MAX_M = 3.0


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


def meet(h, ground, dist, band_m):
    """Heights h pulled onto the scene's ground at the nearest edge, fully
    at it, not at all band_m out."""
    w = 1 - ramp(dist / band_m)
    w = np.where(np.isnan(ground) | (np.abs(np.nan_to_num(ground) - h) > MEET_MAX_M), 0.0, w)
    return h + (np.nan_to_num(ground) - h) * w


class Footprint:
    """Where the scene stands, seen from above: for any east/north point,
    how far it is from the scene's footprint (0 on it) and the scene's
    ground height and colour at the nearest cell of it."""

    def __init__(self, pts, cols):
        xz = np.floor(pts[:, [0, 2]] / CELL_M).astype(int)
        pad = int(np.ceil(PAD_M / CELL_M))
        self.lo = xz.min(0) - pad
        self.shape = tuple(xz.max(0) - self.lo + pad + 1)
        at = tuple((xz - self.lo).T)
        count = np.zeros(self.shape, int)
        np.add.at(count, at, 1)
        low = np.full(self.shape, -np.inf)         # y is down: the lowest point has the largest y
        np.maximum.at(low, at, pts[:, 1])
        base = pts[:, 1] >= low[at] - LOW_M
        n = np.zeros(self.shape)
        np.add.at(n, tuple(a[base] for a in at), 1)
        colour = np.zeros(self.shape + (3,))
        for j in range(3):
            np.add.at(colour[..., j], tuple(a[base] for a in at), cols[base, j])
        real = count >= MIN_POINTS
        self.ground = np.where(real, -low, np.nan)
        self.colour = np.where(real[..., None], colour / np.maximum(n, 1)[..., None], np.nan)
        on = binary_dilation(real, np.ones((2 * GROW + 1,) * 2, bool))
        self.dist = distance_transform_edt(~on) * CELL_M
        # the nearest cell with points of its own, for its ground and colour
        self.nearest = distance_transform_edt(~real, return_distances=False, return_indices=True)

    def at(self, xy):
        """(distance m, ground height m, ground colour (n, 3)) at east/north
        points; beyond the padded grid: infinitely far, no ground."""
        c = np.floor(xy / CELL_M).astype(int) - self.lo
        inside = ((c >= 0) & (c < self.shape)).all(1)
        dist = np.full(len(xy), np.inf)
        ground = np.full(len(xy), np.nan)
        colour = np.full((len(xy), 3), np.nan)
        ci = tuple(c[inside].T)
        dist[inside] = self.dist[ci]
        ni, nj = self.nearest[0][ci], self.nearest[1][ci]
        ground[inside] = self.ground[ni, nj]
        colour[inside] = self.colour[ni, nj]
        return dist, ground, colour
