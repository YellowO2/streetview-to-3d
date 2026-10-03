"""Find the real Street View panoramas and links near a location, for the map picker."""
import asyncio
import io
import math

import aiohttp
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from streetlevel import streetview
from streetlevel.geo import wgs84_to_tile_coord

from streetview_to_3d.common.scene import node_key
from streetview_to_3d.common.geo import haversine_m as _haversine_m, latlon_to_local_m, local_m_to_latlon
from streetview_to_3d.common.http import BROWSER_HEADERS
from streetview_to_3d.common.streetview_fetch import fetch_panos_by_id, google_node, run_async

# Street View publishes coverage on zoom-17 Slippy Map tiles.
_TILE_ZOOM = 17

# The tile listing holds only car panos, not walked (scout) ones, so where a tile's official
# coverage lines run farther than _GAP_M from every known pano, a pano is searched for.
_GAP_M = 12.0
_PROBE_CONCURRENCY = 16
_MAX_PROBE_WAVES = 6
_MAX_CACHED_TILES = 512
_tile_cache = {}  # (tx, ty) -> [panorama]; coverage barely changes within a run
# cached like tiles, so a redrawn area only looks up what it has not seen
_MAX_CACHED_PANOS = 50000
_meta_cache = {}    # pano id -> fetch_panos_by_id's metadata (its links)
_centre_cache = {}  # (lat, lon, radius) rounded -> the pano nearest it, or None


def _metas(ids):
    """fetch_panos_by_id's metadata for ids, each fetched once (_meta_cache)."""
    missing = [i for i in dict.fromkeys(ids) if i not in _meta_cache]
    if missing:
        for i, meta in zip(missing, run_async(fetch_panos_by_id(missing))):
            if meta:                                  # a failed lookup is tried again next time
                if len(_meta_cache) >= _MAX_CACHED_PANOS:
                    _meta_cache.pop(next(iter(_meta_cache)))
                _meta_cache[i] = meta
    return [_meta_cache.get(i) for i in ids]


def _nearest_to(lat, lon, radius_m):
    """The official pano nearest (lat, lon) within radius_m, looked up once."""
    key = (round(lat, 6), round(lon, 6), round(radius_m, 1))
    if key not in _centre_cache:
        try:
            p = streetview.find_panorama(lat, lon, radius=radius_m)
        except Exception as e:
            print(f"Google search at the center failed: {e}")
            return None
        _centre_cache[key] = p if _is_official(p) else None
    return _centre_cache[key]


def circle(lat, lon, radius_m, corners=8):
    """An area as a polygon: a few (lat, lon) corners on the circle radius_m round (lat, lon),
    so dragging one moves a good part of its edge."""
    a = np.linspace(0, 2 * np.pi, corners, endpoint=False)
    return [list(local_m_to_latlon(radius_m * np.cos(t), radius_m * np.sin(t), lat, lon)) for t in a]


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
        print(f"Coverage line probe failed on tile {tx},{ty}: {e}")
    return panos


async def _discover_tiles(tiles):
    sem = asyncio.Semaphore(_PROBE_CONCURRENCY)
    async with aiohttp.ClientSession(headers=BROWSER_HEADERS) as session:
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


DEFAULT_RADIUS_M = 350
MAX_NODES = 200


def nearby_nodes(lat, lon, radius_m=DEFAULT_RADIUS_M, max_nodes=MAX_NODES):
    """(nodes, edges) within radius_m of (lat, lon): google_node dicts, nearest first (no
    date: the tile listing lacks it), and (key_a, key_b) links from the coverage graph."""
    try:
        panos = google_tile_panos(lat, lon)
    except Exception as e:
        print(f"Google coverage lookup failed: {e}")
        return [], []

    nodes = []
    for p in panos.values():
        if _haversine_m(lat, lon, p.lat, p.lon) > radius_m:
            continue
        nodes.append(google_node(p.id, p.lat, p.lon, p.heading))
    nodes.sort(key=lambda n: _haversine_m(lat, lon, n["lat"], n["lon"]))
    nodes = nodes[:max_nodes]

    kept_keys = {n["key"] for n in nodes}
    edges = set()
    for key in kept_keys:
        pano_id = key.split(":", 1)[1]  # Google ids are strings, matching `panos`' keys directly
        p = panos.get(pano_id)
        if not p:
            continue
        for link in (p.links or []):
            other_key = node_key("google", link.pano.id)
            if other_key in kept_keys and other_key != key:
                edges.add(tuple(sorted((key, other_key))))

    return nodes, sorted(edges)


# A discovered pano this close to a walked one is the same place, not a new walk.
_SAME_PLACE_M = 3.0

# A pano with water this far all round is a boat's: too little shore for DA3, never walked to.
_OPEN_WATER_M = 30.0


def on_open_water(often):
    """f(lats, lons) -> whether each is out on the water; often(lat, lon)
    -> how often water there, 0-1 (postprocess/water.occurrence_map).
    Nowhere, if the map cannot be had."""
    from streetview_to_3d.postprocess.world.water import WET

    def f(lat, lon):
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        try:
            out = often(lat, lon) >= WET
            for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                out &= often(*local_m_to_latlon(_OPEN_WATER_M * np.cos(a), _OPEN_WATER_M * np.sin(a), lat, lon)) >= WET
            return out
        except OSError as e:
            print(f"Water map lookup failed: {e}")
            return np.zeros(lat.shape, bool)
    return f


def expand_area(center_lat, center_lon, radius_m=None, max_nodes=2000, area=None):
    """Every real Street View graph within radius_m of the center, or inside area (a polygon
    [(lat, lon)] as the map's edges were dragged), as (nodes, edges) like nearby_nodes'.

    Edges are only real per-pano links. Each graph is walked on its own, wave by wave: the
    first from the pano nearest the center, then from the nearest still unreached. Panos
    past the area or out on open water are left out. Every lookup is cached.
    """
    import shapely
    if area is not None:
        here = _to_metres(center_lat, [(center_lat, center_lon)])[0]
        corners = _to_metres(center_lat, area)
        shape = shapely.Polygon(corners).buffer(0)
        shapely.prepare(shape)
        radius_m = float(np.linalg.norm(corners - here, axis=1).max())

        def within(nodes):
            xy = _to_metres(center_lat, [(n["lat"], n["lon"]) for n in nodes])
            return shapely.contains_xy(shape, xy[:, 0], xy[:, 1]) if len(nodes) else np.zeros(0, bool)
    else:
        def within(nodes):
            return np.array([_haversine_m(center_lat, center_lon, n["lat"], n["lon"]) <= radius_m
                             for n in nodes], bool)
    try:
        discovered = google_tile_panos(center_lat, center_lon, radius_m)
    except Exception as e:
        print(f"Google coverage lookup failed: {e}")
        return [], []
    # also the pano nearest the center: discovery keeps a walked path's panos only every
    # 12-24 m, so a small radius in a park can hold none
    at_center = _nearest_to(center_lat, center_lon, min(max(radius_m, 15), 50))
    if at_center is not None:
        discovered.setdefault(at_center.id, at_center)

    def dist(n):
        return _haversine_m(center_lat, center_lon, n["lat"], n["lon"])

    seeds = [google_node(p.id, p.lat, p.lon, p.heading) for p in discovered.values()]
    seeds = [n for n, ok in zip(seeds, within(seeds)) if ok]
    from streetview_to_3d.postprocess.world.water import occurrence_map
    at_sea = on_open_water(occurrence_map())
    afloat = set()  # ids of the panos left out

    def ashore(found):
        """The found nodes not out on the water, noting those that are."""
        found = [n for n in found if n["id"] not in afloat]
        if not found:
            return found
        wet = at_sea([n["lat"] for n in found], [n["lon"] for n in found])
        afloat.update(n["id"] for n, w in zip(found, wet) if w)
        return [n for n, w in zip(found, wet) if not w]
    seeds = sorted(ashore(seeds), key=dist)
    if not seeds:
        return [], []

    nodes, edges = [], []
    by_key, edge_set, visited = {}, set(), set()
    walked_ll = []  # positions of nodes walks have reached, for _SAME_PLACE_M

    def add_node(n):
        if n["key"] not in by_key:
            nodes.append(n)
            by_key[n["key"]] = n
            walked_ll.append((n["lat"], n["lon"]))

    def reached(seed):
        if seed["key"] in by_key:
            return True
        return any(_haversine_m(seed["lat"], seed["lon"], la, lo) <= _SAME_PLACE_M for la, lo in walked_ll)

    walks = 0
    for seed in seeds:
        if len(nodes) >= max_nodes:
            break
        if reached(seed):
            continue
        walks += 1
        add_node(seed)
        frontier = [seed["key"]]
        while frontier and len(nodes) < max_nodes:
            visited.update(frontier)
            metas = _metas([k.split(":", 1)[1] for k in frontier])
            next_frontier = []
            for key, meta in zip(frontier, metas):
                if not meta:
                    continue
                known = [n for n in meta["neighbors"] if node_key("google", n["id"]) in by_key]
                new = [n for n in meta["neighbors"] if n not in known]
                for n in known + ashore([n for n, ok in zip(new, within(new)) if ok]):
                    node = google_node(n["id"], n["lat"], n["lon"])
                    other_key = node["key"]
                    add_node(node)
                    fe = frozenset((key, other_key))
                    if fe not in edge_set:
                        edges.append((key, other_key))
                        edge_set.add(fe)
                    if other_key not in visited and other_key not in next_frontier:
                        next_frontier.append(other_key)
            frontier = next_frontier

    print(f"expand_area: {len(nodes)} node(s) from {walks} walk(s), {len(discovered)} discovered, "
          f"{len(afloat)} out on the water left out")
    return nodes, edges
