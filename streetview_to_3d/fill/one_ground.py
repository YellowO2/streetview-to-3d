"""One ground under the whole scene.

Every DA3 cloud has its own ground, a little bumpy, and none of them has
anything under its own camera: DA3's views stop about 34 degrees below the
horizon, so each camera leaves a blind disc about 4 m across. Here all of
them become one smooth surface:

1. each cloud's ground (postprocess.ground, started from a ring around the
   camera since the disc under it is empty)
2. one height map (postprocess.ground.GroundMap): each square takes the
   ground of the cloud whose camera is nearest; where DA3 has no ground,
   Google's own ground fills in, shifted by the typical height difference
   between the two where both have it, so they meet without a step
3. the ground area: every square with ground, gaps up to CLOSE_M closed,
   enclosed holes filled, plus each camera's blind disc -- out to where its
   views stop, but never past the nearest wall in that direction
4. the area laid out as an even STEP_M grid on the map, and each cloud's
   own points lying on it (within ON_GROUND_M) removed: one ground, not
   several stacked a few cm apart
"""
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes

from streetview_to_3d.postprocess.ground import GroundMap, ground
from streetview_to_3d.services.da3_ops import VIEW_HFOV

CELL_M, SMOOTH_M, STEP_M = 0.5, 0.5, 0.05
DA3_SEED_M = 7.0        # the ground detector may start this far out (DA3's blind disc is ~4 m)
CLOSE_M = 1.5           # gaps in the ground area this wide are closed
ON_GROUND_M = 0.12      # a cloud's points this close to the surface are the surface
BLIND_MARGIN_M = 1.0    # the blind disc under a camera, plus this
WALL_ABOVE_M = 0.3      # points this far above the ground stop a blind disc
CAM_H = 2.45            # camera height, where a camera has no ground under it
# how far below the horizon DA3's 16:9 views reach
VIEW_BOTTOM_DEG = np.degrees(np.arctan(np.tan(np.radians(VIEW_HFOV / 2)) * 9 / 16))


def one_ground(clouds, cams, normals, google_ground):
    """(clouds without their ground points, the one ground's points).

    clouds[k]: cloud k's points in world metres (y down), cams[k] its
    camera, normals[k] its points' normals. google_ground: the Google
    base's ground points, for where DA3 has none."""
    G = [ground(x, c, normals=n, seed_m=DA3_SEED_M) for x, c, n in zip(clouds, cams, normals)]
    m = GroundMap([x[g] for x, g in zip(clouds, G)], cams, CELL_M, SMOOTH_M, extent=google_ground)

    # Google's ground where DA3 has none, shifted onto DA3's height
    g_raw = m.medians(google_ground)
    both = m.have & np.isfinite(g_raw)
    shift = float(np.median(m.raw[both] - g_raw[both])) if both.sum() > 20 else 0.0
    from_google = ~m.have & np.isfinite(g_raw)
    m.raw[from_google] = g_raw[from_google] + shift
    m.height = m.smoothed()

    area = from_google.copy()
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
