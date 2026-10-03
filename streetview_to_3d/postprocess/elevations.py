"""Google elevations of the Street View panos around the scene (for terrain.from_panos): off
bridges and tunnels, one per CELL_M square, outliers against their neighbours dropped."""
import math

import numpy as np

TILE_ZOOM = 17
REACH_M = 300.0         # panos this far past the scene's cameras
M_PER_LAT = 111320.0
CELL_M = 25.0
NEIGHBOUR_M, OUTLIER_M = 80.0, 2.5
OFF_GROUND_M = 15.0     # a pano this near a bridge or tunnel way is on it


def around(lats, lons):
    """(lat, lon, elevation) arrays of the panos within REACH_M of the box round (lats, lons)."""
    import aiohttp
    from streetlevel import streetview
    from streetview_to_3d.postprocess.openfreemap import tile_xy
    from streetview_to_3d.services.http_headers import BROWSER_HEADERS
    from streetview_to_3d.services.streetview_fetch import run_async
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


def on_ground(xy, elements, to_xy):
    """Whether each east/north xy (n, 2) is clear of OSM's bridges, tunnels and other non-ground roads."""
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


def thinned(xy, elevation):
    """(xy (m, 2), elevation (m,)): one mean point and median elevation per CELL_M square,
    without squares more than OUTLIER_M off the plane of their neighbours within NEIGHBOUR_M."""
    from scipy.spatial import cKDTree
    if not len(xy):
        return np.zeros((0, 2)), np.zeros(0)
    key, inv = np.unique(np.floor(xy / CELL_M), axis=0, return_inverse=True)
    inv = inv.ravel()
    at = np.array([xy[inv == k].mean(0) for k in range(len(key))])
    h = np.array([np.median(elevation[inv == k]) for k in range(len(key))])
    keep = np.ones(len(at), bool)
    for k, near in enumerate(cKDTree(at).query_ball_point(at, NEIGHBOUR_M)):
        others = [j for j in near if j != k]
        if len(others) >= 3:
            rows = np.c_[np.ones(len(others)), at[others] - at[k]]
            plane = np.linalg.lstsq(rows, h[others], rcond=None)[0]
            keep[k] = abs(h[k] - plane[0]) <= OUTLIER_M
    return at[keep], h[keep]
