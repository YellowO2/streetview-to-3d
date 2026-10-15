"""Roofs without a colour of their own given the satellite's."""
import logging
import warnings
from dataclasses import replace

import numpy as np

from streetview_to_3d.build_scene.postprocess.world.Satellite import LIFT
from streetview_to_3d.build_scene.postprocess.world.buildings.tools.colours import cheer
from streetview_to_3d.build_scene.postprocess.world.tools.geometry import inset

logger = logging.getLogger(__name__)

ROOF_ZOOM = 18
ROOF_INSET_M, ROOF_STEP_M = 1.0, 1.0   # Google's roof sampling: inset from the edge, spacing
SAT_INSET_M, SAT_STEP_M = 3.0, 4.0     # Sentinel-2's, coarser


def roof_colours(outlines, site, satellite):
    """Outlines whose Forms have their roofs coloured from Google's satellite (Satellite), else
    Sentinel-2's, unless OSM gave them one; satellite_roofs: how many. site: Site."""
    items, to_ll = outlines.items, site.to_ll
    if not items:
        return outlines
    n_sat = 0
    try:                                # roof colours from Google's satellite
        roof_at = satellite.google_map(ROOF_ZOOM)
        roof_at.fetch(*to_ll(np.concatenate([o[0] for o in items])))
        n_sat = _satellite_roofs(items, lambda en: satellite.unhazed(roof_at(*to_ll(en))),
                                 ROOF_INSET_M, ROOF_STEP_M, lively=False)
    except OSError as e:
        logger.warning(f"terrain: no Google roofs ({e!r}), Sentinel-2's")
    try:                                # ... else Sentinel-2's
        n_sat += (_satellite_roofs(items, lambda en: satellite.unhazed(satellite.sentinel(*to_ll(en)) ** LIFT))
                  if satellite.sentinel else 0)
    except OSError as e:
        logger.warning(f"terrain: no satellite roofs ({e!r})")
    return replace(outlines, satellite_roofs=n_sat)


def _satellite_roofs(outlines, colour_at, inset_m=SAT_INSET_M, step=SAT_STEP_M, lively=True):
    """Give roofs without a colour the satellite's median colour (inset_m in, step apart; cheer'd if lively).

    colour_at(east/north (n, 2)) -> RGB. Returns how many roofs were coloured."""
    n = 0
    for xy, _, _, form, *_ in outlines:
        if form.roof_colour is not None:
            continue
        ring = inset(xy, inset_m)
        if ring is None:
            continue
        lo, hi = ring.min(0), ring.max(0)
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0], step) + step / 2,
                             np.arange(lo[1], hi[1], step) + step / 2)
        grid = np.stack([gx.ravel(), gy.ravel()], 1)
        at = np.concatenate([grid[_inside(grid, ring)], ring[:-1]])
        with warnings.catch_warnings():     # all-NaN slices are skipped below
            warnings.simplefilter("ignore", RuntimeWarning)
            c = np.nanmedian(colour_at(at), 0)
        if not np.isfinite(c).all():        # the map has none there
            continue
        form.roof_colour = cheer(c) if lively else c
        n += 1
    return n


def _inside(pts, ring):
    """True where pts (m, 2) fall inside the closed ring (n, 2)."""
    a, b = ring[:-1], ring[1:]
    x, y = pts[:, :1], pts[:, 1:]
    crosses = ((a[:, 1] > y) != (b[:, 1] > y)) & \
        (x < (b[:, 0] - a[:, 0]) * (y - a[:, 1]) / (b[:, 1] - a[:, 1] + 1e-12) + a[:, 0])
    return crosses.sum(1) % 2 == 1
