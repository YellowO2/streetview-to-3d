"""Google's points laid onto DA3's and given the panos' look."""
import logging
from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from streetview_to_3d.build.postprocess.SceneGround import CELL_M
from streetview_to_3d.build.postprocess.world.tools.seams import unhazed

logger = logging.getLogger(__name__)

FIT_REACH_M, FIT_N, FIT_NEAR_M = 15.0, 40_000, 0.3     # shift: Google points this near a pano, about this many
FIT_STEPS_M = 1.0, 0.3, 0.1                            # ... moved by up to 3 of each step in turn


def aligned(points, site, centres):
    """GooglePoints slid onto DA3's points (site: Site; centres: the panos', pano_centres) and
    coloured to look like the panos (unhazed); as they are if already merged."""
    if points.merged:
        logger.info("google: already merged")
        return points
    move = _shift(points.pts, site.points, centres)
    logger.info(f"google: moved {move.round(2)} m (x east, y down, z north) onto DA3's points")
    pts = points.pts + move
    by_place = cKDTree(pts[:, [0, 2]])
    cols = points.cols

    def google_at(q):
        d, k = by_place.query(q, distance_upper_bound=2 * CELL_M, workers=-1)
        out = np.full((len(q), 3), np.nan)
        out[np.isfinite(d)] = cols[k[np.isfinite(d)]]
        return out
    return replace(points, pts=pts, cols=unhazed(site.ground, google_at)(cols))


def _shift(pts, scene, cams):
    """The move (x, y, z; m) that lays Google's points onto DA3's: the one putting most of the
    Google points near the panos within FIT_NEAR_M of a DA3 point, searched coarse to fine."""
    near = cKDTree(cams).query(pts, distance_upper_bound=FIT_REACH_M)[0] < np.inf
    sample = pts[near][::max(1, int(near.sum() // FIT_N))]
    tree = cKDTree(scene)
    best = np.zeros(3)
    for step in FIT_STEPS_M:
        moves = best + step * np.stack(np.meshgrid(*[np.arange(-3, 4)] * 3, indexing="ij"), -1).reshape(-1, 3)
        hits = [np.isfinite(tree.query(sample + m, distance_upper_bound=FIT_NEAR_M, workers=-1)[0]).sum()
                for m in moves]
        best = moves[int(np.argmax(hits))]
    return best
