"""One ground under the whole scene.

Every DA3 cloud has its own ground, a little bumpy, and none of them has
anything under its own camera: DA3's views stop about 29 degrees below the
horizon, so each camera leaves a blind disc reaching about 4.5 m out. Here all of
them become one smooth surface:

1. each cloud's ground (postprocess.ground, started from a ring around the
   camera since the disc under it is empty), only what its pano's class
   map calls WALKABLE: flat at street height is not enough -- DA3 laid the
   water it saw through a bridge's railing there too, and the fill paved
   the harbour with it
2. one height map (postprocess.ground.GroundMap): each square the clouds'
   ground there, the nearest camera's most, blended across where two
   cameras' squares meet -- the pieces already lifted or lowered to meet
   (level.py), so what is left to blend is a few cm
3. the ground area: every square with ground, gaps up to CLOSE_M closed,
   enclosed holes filled, plus each camera's blind disc -- out to where its
   views stop, but never past the nearest wall in that direction
4. the area laid out as an even STEP_M grid on the map, and each cloud's
   own points lying on it (within ON_GROUND_M) removed: one ground, not
   several stacked a few cm apart
"""
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes

from streetview_to_3d.postprocess.ground import CELL_M as GROUND_CELL_M, GroundMap, ground, normals_from_neighbours
from streetview_to_3d.services.da3_ops import VIEW_HFOV

CELL_M, SMOOTH_M, STEP_M = 0.5, 0.5, 0.08   # its points 8 cm apart: drawn as big as DA3's, solid still, and a third as many as at 5 cm
DA3_SEED_M = 7.0        # the ground detector may start this far out (DA3's blind disc reaches ~4.5 m)
CLOSE_M = 1.5           # gaps in the ground area this wide are closed
ON_GROUND_M = 0.12      # a cloud's points this close to the surface are the surface
LOW_M = 1.0             # only points this near the lowest in their square can be ground
BLIND_MARGIN_M = 1.0    # the blind disc under a camera, plus this
WALL_ABOVE_M = 0.3      # points this far above the ground stop a blind disc
CAM_H = 2.45            # camera height, where a camera has no ground under it
WALKABLE = ("road", "sidewalk", "terrain")      # the masker's classes (services.segment)
# how far below the horizon DA3's 16:9 views reach
VIEW_BOTTOM_DEG = np.degrees(np.arctan(np.tan(np.radians(VIEW_HFOV / 2)) * 9 / 16))


def grounds(clouds, cams, walkable=None):
    """Which of each cloud's points are its walkable ground (step 1).

    clouds[k]: cloud k's points in world metres (y down), cams[k] its
    camera, walkable[k] which of them its pano calls WALKABLE (None: all)."""
    G = [ground(x, c, normals=_low_normals(x), seed_m=DA3_SEED_M) for x, c in zip(clouds, cams)]
    return G if walkable is None else [g & w for g, w in zip(G, walkable)]


def _low_normals(x):
    """The normals of the points within LOW_M of the lowest in their square
    (the ground detector's, postprocess.ground.CELL_M); zero -- facing no
    way, never ground -- for the rest, which stand too high to be ground.
    That is most of a cloud, and every point's normal was a third of the
    fill's time."""
    sq = np.floor(x[:, [0, 2]] / GROUND_CELL_M).astype(np.int64) + 2 ** 20
    _, inv = np.unique((sq[:, 0] << 21) | sq[:, 1], return_inverse=True)
    low = np.full(inv.max() + 1, -np.inf)
    np.maximum.at(low, inv, x[:, 1])                  # y is down: the largest y is the lowest
    near = x[:, 1] > low[inv] - LOW_M
    n = np.zeros_like(x)
    n[near] = normals_from_neighbours(x[near])
    return n


def one_ground(clouds, cams, G):
    """(clouds without their ground points, the one ground's points).

    clouds, cams as grounds'; G its answer."""
    m = GroundMap([x[g] for x, g in zip(clouds, G)], cams, CELL_M, SMOOTH_M)

    area = np.zeros(m.dims, bool)
    for x, g in zip(clouds, G):
        ij = m.squares(x[g][:, [0, 2]])
        area[ij[:, 0], ij[:, 1]] = True
    r = int(round(CLOSE_M / CELL_M))
    padded = np.pad(area, r + 1)                       # closing must not eat the map's rim
    area = binary_fill_holes(binary_closing(padded, np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r))[r + 1:-r - 1, r + 1:-r - 1]
    m.spread()                                         # heights under the cameras too, to tell walls there
    area |= _blind_discs(m, clouds, cams)
    area &= np.isfinite(m.height)
    surface, _ = m.grid(area, STEP_M)
    kept = []
    for x in clouds:
        ij = np.clip(m.squares(x[:, [0, 2]]), 0, m.dims - 1)
        y, ok = m.at(x[:, [0, 2]])
        kept.append(~(area[ij[:, 0], ij[:, 1]] & ok & (np.abs(x[:, 1] - np.nan_to_num(y)) < ON_GROUND_M)))
    return kept, surface


def _blind_discs(m, clouds, cams):
    """Each camera's blind disc: under it, out to where its views stop, but
    never past the nearest wall in that direction (any cloud's)."""
    area = np.zeros(m.dims, bool)
    allx = np.concatenate(clouds)
    ii, jj = np.indices(m.dims)
    centre = m.lo + (np.stack([ii, jj], -1) + .5) * m.cell
    for c in cams:
        cij = np.clip(m.squares(c[None, [0, 2]])[0], 0, m.dims - 1)
        h = m.height[tuple(cij)] - c[1] if np.isfinite(m.height[tuple(cij)]) else CAM_H
        reach = h / np.tan(np.radians(VIEW_BOTTOM_DEG)) + BLIND_MARGIN_M
        x = allx[np.all(np.abs(allx[:, [0, 2]] - c[[0, 2]]) < reach + 5, axis=1)]
        y, ok = m.at(x[:, [0, 2]])
        x = x[ok & (x[:, 1] < y - WALL_ABOVE_M)]          # standing above the ground (y down)
        rel = x[:, [0, 2]] - c[[0, 2]]
        wall = np.full(72, np.inf)                        # nearest wall per 5 degrees
        np.minimum.at(wall, _az(rel), np.hypot(*rel.T))
        n = int(reach / m.cell) + 2
        box = tuple(slice(max(0, a - n), a + n + 1) for a in cij)
        rc = centre[box] - c[[0, 2]]
        dc = np.hypot(rc[..., 0], rc[..., 1])
        area[box] |= (dc < reach) & (dc < wall[_az(rc)])
    return area


def _az(rel):
    return ((np.degrees(np.arctan2(rel[..., 0], rel[..., 1])) + 360) // 5).astype(int) % 72
