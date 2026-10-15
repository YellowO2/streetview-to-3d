"""Google's elevations of the other Street View panos round the scene."""
import logging
import math

import numpy as np

from streetview_to_3d.build_scene.postprocess.world.Elevations import Elevations
from streetview_to_3d.build_scene.postprocess.world.tools.reach import M_PER_LAT

logger = logging.getLogger(__name__)

TILE_ZOOM = 17
REACH_M = 300.0         # panos this far past the scene's cameras


def elevations_around(site):
    """Elevations of the panos within REACH_M of the box round site's cameras (Site), from
    Google's coverage tiles; none if Google does not answer."""
    lat0, lon0 = site.origin
    try:
        la, lo, el = _around(*site.to_ll(site.cameras))
    except (OSError, ValueError) as e:
        logger.warning(f"terrain: no other panos' elevation ({e!r})")
        return Elevations(np.zeros((0, 2)), np.zeros(0))
    return Elevations(np.c_[(lo - lon0) * site.m_per_lon, (la - lat0) * M_PER_LAT], el)


def _around(lats, lons):
    """(lat, lon, elevation) arrays of the panos within REACH_M of the box round (lats, lons)."""
    import aiohttp
    from streetlevel import streetview
    from streetview_to_3d.tools.streetview_fetch.browser_session import BROWSER_HEADERS
    from streetview_to_3d.build_scene.postprocess.world.tools.tiles import tile_xy
    from streetview_to_3d.tools.run_async import run_async
    lat0 = float(np.mean(lats))
    dlat, dlon = REACH_M / M_PER_LAT, REACH_M / (M_PER_LAT * math.cos(math.radians(lat0)))
    s, w, n, e = min(lats) - dlat, min(lons) - dlon, max(lats) + dlat, max(lons) + dlon
    xs, ys = tile_xy(np.array([n, s]), np.array([w, e]), TILE_ZOOM)
    cells = [(x, y) for x in range(int(xs[0]), int(xs[1]) + 1) for y in range(int(ys[0]), int(ys[1]) + 1)]

    async def fetch():
        async with aiohttp.ClientSession(headers=BROWSER_HEADERS) as session:
            import asyncio
            return await asyncio.gather(*(streetview.get_coverage_tile_async(x, y, session) for x, y in cells),
                                        return_exceptions=True)
    seen = {}
    for got in run_async(fetch()):
        if isinstance(got, Exception):
            continue
        for p in got:
            if p.elevation is not None and s <= p.lat <= n and w <= p.lon <= e:
                seen[p.id] = (p.lat, p.lon, p.elevation)
    out = np.array(list(seen.values()), float).reshape(-1, 3)
    return out[:, 0], out[:, 1], out[:, 2]
