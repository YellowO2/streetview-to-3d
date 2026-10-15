"""Colour the one ground from the panos that see it."""
import logging

import numpy as np
from scipy.spatial import cKDTree

from streetview_to_3d.tools.pano_mask import pano_mask
from streetview_to_3d.postprocess.fill.PaintedGround import PaintedGround
from streetview_to_3d.postprocess.tools.paint import blurred, paint

logger = logging.getLogger(__name__)


def painted(ground, clouds, cameras):
    """The PaintedGround of points ground, coloured from the cameras' panos past DA3's points (clouds);
    points no camera can colour take the nearest painted point's colour, dropping them would
    leave holes. With nothing painted at all, an empty PaintedGround."""
    photos = [c.photo for c in cameras]
    da3 = np.concatenate([c.points for c in clouds])
    col, who = paint(ground, da3, cameras,
                     [ph and (ph[0], pano_mask(ph[1]) | blurred(ph[0])) for ph in photos])
    done, fallback = who >= 0, who < 0
    if done.any() and fallback.any():
        _, nb = cKDTree(ground[done]).query(ground[fallback])
        src = np.flatnonzero(done)[nb]
        col[fallback], who[fallback] = col[src], who[src]
    ok = who >= 0
    logger.info(f"fill: one ground {int(ok.sum())} of {len(ground)} points, "
                f"{int(sum(len(c.points) for c in clouds))} DA3 points kept")
    return PaintedGround(ground[ok], col[ok], who[ok])
