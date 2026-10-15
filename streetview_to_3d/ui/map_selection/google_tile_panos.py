"""Find every official Street View pano on the map tiles around a place.

The tile listing holds only car panos, so the coverage lines it draws are probed for the
walked (scout) ones it missed. Tiles are cached: a lookup nearby fetches only what is new.
"""
import asyncio
import io
import logging
import math

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from streetlevel import streetview
from streetlevel.geo import wgs84_to_tile_coord

from streetview_to_3d.common.geo import latlon_to_local_m
from streetview_to_3d.common.run_async import run_async
from streetview_to_3d.common.streetview_fetch.browser_session import browser_session

logger = logging.getLogger(__name__)

# Street View publishes coverage on zoom-17 Slippy Map tiles.
_TILE_ZOOM = 17

# The tile listing holds only car panos, not walked (scout) ones, so where a tile's official
# coverage lines run farther than _GAP_M from every known pano, a pano is searched for.
_GAP_M = 12.0
_PROBE_CONCURRENCY = 16
_MAX_PROBE_WAVES = 6
_MAX_CACHED_TILES = 512
_tile_cache = {}  # (tx, ty) -> [panorama]; coverage barely changes within a run


def _official_lines_url(tx, ty):
    return (f"https://www.google.com/maps/vt?pb=!1m5!1m4!1i{_TILE_ZOOM}!2i{tx}!3i{ty}!4i256"
            "!2m8!1e2!2ssvv!4m2!1scc!2s*211m3*211e2*212b1*213e2*212b1*214b1"
            "!4m2!1ssvl!2s*211b0*212b1!3m8!2sen!3sus!5e1105!12m4!1e68!2m2!1sset!2sRoadmap"
            "!4e0!5m4!1e0!8m2!1e1!1e1!6m6!1e12!2i2!11e0!39b0!44e0!50e0")


def _tile_pixel_to_latlon(tx, ty, px, py, size):
    n = 2 ** _TILE_ZOOM
    x, y = (tx + px / size) / n, (ty + py / size) / n
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y)))), x * 360 - 180


def _to_metres(lat0, latlons):
    """(lat, lon) rows -> flat-earth (north, east) metres, for distances only."""
    a = np.asarray(latlons, float).reshape(-1, 2)
    east, north = latlon_to_local_m(a[:, 0], a[:, 1], lat0, 0.0)
    return np.c_[north, east]


def _is_official(p):
    return p is not None and not (p.source or "").startswith("photos:")


def _with_link_positions(panos):
    out = [(p.lat, p.lon) for p in panos]
    out += [(l.pano.lat, l.pano.lon) for p in panos for l in (p.links or []) if l.pano.lat is not None]
    return out


async def _probe_line_gaps(session, sem, tx, ty, known_panos):
    """Official panos on this tile's blue lines that the tile listing missed.

    Line pixels (every 4th px, ~4 m) farther than _GAP_M from any known pano
    or link are gaps; each wave probes gap points spaced 2*_GAP_M apart, all
    at once. A hit's own links mark their stretch known too, so a walked
    path is filled from few probes; a miss retires its surroundings. Waves
    repeat until no gap is left.
    """
    async with session.get(_official_lines_url(tx, ty)) as r:
        r.raise_for_status()
        rgba = np.asarray(Image.open(io.BytesIO(await r.read())).convert("RGBA"))
    size = rgba.shape[0]
    ys, xs = np.nonzero(rgba[..., 3] > 128)
    step = max(1, size // 64)
    on_grid = (ys % step == 0) & (xs % step == 0)
    ys, xs = ys[on_grid], xs[on_grid]
    if not len(xs):
        return []

    lat0 = _tile_pixel_to_latlon(tx, ty, size / 2, size / 2, size)[0]
    line_ll = [_tile_pixel_to_latlon(tx, ty, x + 0.5, y + 0.5, size) for x, y in zip(xs, ys)]
    line = _to_metres(lat0, line_ll)
    known = _with_link_positions(known_panos)
    retired = np.zeros(len(line), bool)
    found = {}

    async def probe(i):
        async with sem:
            return await streetview.find_panorama_async(*line_ll[i], session, radius=_GAP_M)

    for _ in range(_MAX_PROBE_WAVES):
        if known:
            dist = cKDTree(_to_metres(lat0, known)).query(line)[0]
        else:
            dist = np.full(len(line), np.inf)
        gaps = np.nonzero((dist > _GAP_M) & ~retired)[0]
        if not len(gaps):
            break
        picks = []
        for i in gaps:
            if all(np.hypot(*(line[i] - line[j])) > 2 * _GAP_M for j in picks):
                picks.append(i)
        results = await asyncio.gather(*(probe(i) for i in picks), return_exceptions=True)
        for i, p in zip(picks, results):
            retired[i] = True
            if isinstance(p, Exception) or not _is_official(p):
                retired |= np.hypot(*(line - line[i]).T) < _GAP_M
                continue
            found[p.id] = p
            known += _with_link_positions([p])
    return list(found.values())


async def _discover_tile(session, sem, tx, ty):
    panos = await streetview.get_coverage_tile_async(tx, ty, session)
    try:
        panos += await _probe_line_gaps(session, sem, tx, ty, panos)
    except Exception as e:  # the listing alone is still a usable answer
        logger.warning(f"Coverage line probe failed on tile {tx},{ty}: {e}")
    return panos


async def _discover_tiles(tiles):
    sem = asyncio.Semaphore(_PROBE_CONCURRENCY)
    async with browser_session() as session:
        return await asyncio.gather(*(_discover_tile(session, sem, tx, ty) for tx, ty in tiles))


def _tile_neighborhood(lat, lon, radius_m=None):
    """The zoom-17 tiles around (lat, lon): 3x3, or as many rings as it
    takes to hold radius_m on every side."""
    tx, ty = wgs84_to_tile_coord(lat, lon, _TILE_ZOOM)
    rings = 1
    if radius_m is not None:
        tile_m = 40075016.7 * math.cos(math.radians(lat)) / 2 ** _TILE_ZOOM
        rings = max(1, math.ceil(radius_m / tile_m))
    for dx in range(-rings, rings + 1):
        for dy in range(-rings, rings + 1):
            yield tx + dx, ty + dy


def google_tile_panos(lat, lon, radius_m=None):
    """All official panos (listed and walked) on the tiles around (lat, lon), keyed by id.
    Tiles are cached, so neighbouring lookups fetch only what they haven't seen."""
    tiles = list(_tile_neighborhood(lat, lon, radius_m))
    missing = [t for t in tiles if t not in _tile_cache]
    if missing:
        for t, panos in zip(missing, run_async(_discover_tiles(missing))):
            if len(_tile_cache) >= _MAX_CACHED_TILES:
                _tile_cache.pop(next(iter(_tile_cache)))
            _tile_cache[t] = panos
    seen = {}
    for t in tiles:
        for p in _tile_cache[t]:
            seen[p.id] = p
    return seen
