"""Google's elevation of every Street View pano around the scene, not only
the scene's own: more places where the land's height is known (terrain.py
bends the map onto them, correction).

Google's coverage tiles (zoom TILE_ZOOM, the ones the map picker reads) list
each pano with its elevation -- no pano asked one by one: Lund's 1585
within 300 m in 0.5 s. Only the car's (and Google's own walked) panos: no
user photospheres.

Not all of them are the ground: a pano on a bridge stands over it, one in
a tunnel under it (Lake Como's lakeside road: 66 m under the hill). So
those on OSM's bridges and in its tunnels are left out (on_ground), the
rest thinned to one per CELL_M square (the median of its panos' fixes,
Google minus the map), a square whose fix is more than OUTLIER_M off its
neighbours' (within NEIGHBOUR_M) left out, and (terrain.py) one more than
MAX_FIX_M off the scene's own.
"""
import math

import numpy as np

TILE_ZOOM = 17
REACH_M = 300.0         # panos this far past the scene's cameras
CELL_M = 25.0
NEIGHBOUR_M, OUTLIER_M = 80.0, 2.5
OFF_GROUND_M = 15.0     # a pano this near a bridge's or a tunnel's way is on it (Lake Como's: 9-10 m off the line)
MAX_FIX_M = 15.0


def around(lats, lons, reach_m=REACH_M, m_per_lat=111320.0):
    """(lat, lon, elevation) arrays of the panos within reach_m of the box
    round (lats, lons); empty if the tiles cannot be had."""
    import aiohttp
    from streetlevel import streetview
    from streetview_to_3d.services.http_headers import BROWSER_HEADERS
    from streetview_to_3d.services.streetview_fetch import run_async
    lat0 = float(np.mean(lats))
    dlat, dlon = reach_m / m_per_lat, reach_m / (m_per_lat * math.cos(math.radians(lat0)))
    s, w, n, e = min(lats) - dlat, min(lons) - dlon, max(lats) + dlat, max(lons) + dlon
    z = 2 ** TILE_ZOOM
    tx = lambda lon: int((lon + 180) / 360 * z)
    ty = lambda lat: int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * z)
    cells = [(x, y) for x in range(tx(w), tx(e) + 1) for y in range(ty(n), ty(s) + 1)]

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


def on_ground(xy, elements, to_xy):
    """Whether each east/north xy (n, 2) is off OSM's bridges, tunnels and
    raised or sunken roads (elements: osm.fetch's; to_xy their geometry's)."""
    import shapely
    off = [shapely.LineString(to_xy(e["geometry"])).buffer(OFF_GROUND_M)
           for e in elements if e.get("type") == "way" and "highway" in e.get("tags", {})
           and len(e.get("geometry") or []) >= 2 and _raised(e["tags"])]
    if not off or not len(xy):
        return np.ones(len(xy), bool)
    return ~shapely.contains_xy(shapely.union_all(off), xy[:, 0], xy[:, 1])


def _raised(tags):
    return (tags.get("bridge", "no") != "no" or tags.get("tunnel", "no") != "no"
            or tags.get("covered", "no") != "no" or tags.get("layer", "0") not in ("0", ""))


def thinned(xy, fix):
    """(xy (m, 2), fix (m,)): xy's (east/north (n, 2)) fixes, one per
    CELL_M square (its panos' middle, their median fix), less the squares
    more than OUTLIER_M off the median of their neighbours' within
    NEIGHBOUR_M (a bridge's, a tunnel's)."""
    from scipy.spatial import cKDTree
    if not len(xy):
        return np.zeros((0, 2)), np.zeros(0)
    key, inv = np.unique(np.floor(xy / CELL_M), axis=0, return_inverse=True)
    inv = inv.ravel()
    at = np.array([xy[inv == k].mean(0) for k in range(len(key))])
    f = np.array([np.median(fix[inv == k]) for k in range(len(key))])
    keep = np.ones(len(at), bool)
    for k, near in enumerate(cKDTree(at).query_ball_point(at, NEIGHBOUR_M)):
        others = [j for j in near if j != k]
        if len(others) >= 2 and abs(f[k] - np.median(f[others])) > OUTLIER_M:
            keep[k] = False
    return at[keep], f[keep]
