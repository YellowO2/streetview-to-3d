"""Lay one ground under the whole scene in place of the clouds' own.

Every cloud's walkable ground merged into one height map, gaps, holes and each camera's blind
disc filled, laid out as a STEP_M grid that replaces the clouds' own ground points."""
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes

from streetview_to_3d.common.Cloud import Cloud
from streetview_to_3d.postprocess.fill.GroundMap import GroundMap
from streetview_to_3d.postprocess.place import CAM_H
from streetview_to_3d.postprocess.SceneGround import CELL_M
from streetview_to_3d.models.da3 import VIEW_HFOV

SMOOTH_M, STEP_M = 0.5, 0.08   # height smoothing; ground point spacing
CLOSE_M = 1.5           # gaps in the ground area this wide are closed
ON_GROUND_M = 0.12      # a cloud's points this close to the surface are the surface
BLIND_MARGIN_M = 1.0    # the blind disc under a camera, plus this
WALL_ABOVE_M = 0.3      # points this far above the ground stop a blind disc
# how far below the horizon DA3's 16:9 views reach
VIEW_BOTTOM_DEG = np.degrees(np.arctan(np.tan(np.radians(VIEW_HFOV / 2)) * 9 / 16))


def one_ground(clouds, cameras, walkable):
    """(the world Clouds without their ground points, the one ground's points)."""
    keep, ground = _one_ground([c.points for c in clouds], np.array([c.centre for c in cameras]).reshape(-1, 3),
                               walkable)
    return [Cloud(c.points[k], c.colours[k]) for c, k in zip(clouds, keep)], ground


def _one_ground(clouds, cams, G):
    """(per-cloud keep masks dropping their ground points, the one ground's points); G from walkable_ground._grounds()."""
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
    """Each camera's blind disc out to where its views stop, cut at the nearest wall per direction."""
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
