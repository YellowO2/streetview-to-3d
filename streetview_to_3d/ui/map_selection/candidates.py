"""Low-level fetch of real Street View panoramas near a
location. Used by the map picker and by build_street_graph/.
"""
import asyncio
import io
import math

import aiohttp
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from streetlevel import streetview
from streetlevel.geo import wgs84_to_tile_coord

from streetview_to_3d.services.geo import haversine_m as _haversine_m
from streetview_to_3d.services.http_headers import BROWSER_HEADERS
from streetview_to_3d.services.streetview_fetch import fetch_panos_by_id, run_async

# Street View publishes coverage on zoom-17 Slippy Map tiles.
_TILE_ZOOM = 17

# The coverage tile listing only holds car ("launch") panos. Google's own
# walked captures (Trekker/backpack, source "scout": parks, plazas, paths)
# are left out, though Maps draws them as blue lines. So each tile's
# official-coverage raster (the one Maps draws, image type 2 = Google's
# own, no user photospheres) is read, and only where a line runs farther
# than _GAP_M from every pano already known is a pano searched for.
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
    a = np.asarray(latlons, float).reshape(-1, 2)
    return np.c_[(a[:, 0] - lat0) * 111320.0, a[:, 1] * 111320.0 * math.cos(math.radians(lat0))]


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
    """All official Street View panos on the tiles around (lat, lon) (see
    _tile_neighborhood), keyed by id: the tile listing plus walked (scout)
    captures. Tiles are cached, so neighbouring lookups (fetch_corridor_nodes
    runs one per corridor point) only fetch the tiles they haven't seen."""
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


def node_key(source, pano_id):
    return f"{source}:{pano_id}"


DEFAULT_RADIUS_M = 350
MAX_NODES = 200


def nearby_nodes(lat, lon, radius_m=DEFAULT_RADIUS_M, max_nodes=MAX_NODES):
    """Google Street View nodes within radius_m of (lat, lon), distance-sorted,
    plus edges from Street View's own coverage graph.

    Returns (nodes, edges). Node: {key, source, id, lat, lon, heading} --
    no date (tile listing doesn't carry it; see build_graph/fetch_nodes.py
    for the full per-pano fetch that does). Edge: (key_a, key_b).
    """
    try:
        panos = google_tile_panos(lat, lon)
    except Exception as e:
        print(f"Google coverage lookup failed: {e}")
        return [], []

    nodes = []
    for p in panos.values():
        if _haversine_m(lat, lon, p.lat, p.lon) > radius_m:
            continue
        nodes.append({
            "key": node_key("google", p.id),
            "source": "google",
            "id": p.id,
            "lat": p.lat,
            "lon": p.lon,
            "heading": p.heading,
        })
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


# A discovered pano this close to one already walked is the same place
# (another capture of it), not a new group worth walking from.
_SAME_PLACE_M = 3.0

# A pano with water under it and all round it, this far every way (the JRC
# water map), is out on the water -- a boat's, as Matsushima's bay tour:
# too little shore for DA3 to see, so never walked to. One near land (a
# bridge, a pier, a path on the shore) is kept.
_OPEN_WATER_M = 30.0


def on_open_water(often):
    """f(lats, lons) -> whether each is out on the water; often(lat, lon)
    -> how often water there, 0-1 (postprocess/water.occurrence_map).
    Nowhere, if the map cannot be had."""
    from streetview_to_3d.postprocess.water import WET

    def f(lat, lon):
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        try:
            out = often(lat, lon) >= WET
            dlat = _OPEN_WATER_M / 111320.0
            dlon = dlat / np.cos(np.radians(lat))
            for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                out &= often(lat + dlat * np.sin(a), lon + dlon * np.cos(a)) >= WET
            return out
        except OSError as e:
            print(f"Water map lookup failed: {e}")
            return np.zeros(lat.shape, bool)
    return f


def expand_area(center_lat, center_lon, radius_m, max_nodes=2000):
    """Auto-discover every real Street View graph within radius_m of
    (center_lat, center_lon) -- the same real-link expansion
    map_selection/tab.py's _augment_real_links does for one clicked node,
    just driven by a BFS instead of a person clicking node by node. Feed
    the result straight in as corridor_edges, same shape a manually-built
    selection already produces.

    Edges are only ever real per-pano links (fetch_pano_by_id), never
    guessed from proximity. But the area can hold several graphs Google
    never linked -- a park's walked (scout) paths next to the car's roads,
    streets meeting only across a gap. Each is walked on its own: the first
    from the discovered pano nearest the center, then, while any discovered
    pano in radius is still unreached (and not within _SAME_PLACE_M of a
    walked one), a new walk from the nearest of those. So the result can
    be several components; they no longer need joining here, since every
    piece is placed by its own GPS (postprocess/place.py), and
    fetch_nodes.corridor_points bridges the ones that come close.

    Each walk goes wave by wave, a whole frontier fetched at once. Only
    ever expanding FROM a node still within radius_m -- anything found just
    past the boundary is kept as a leaf but never expanded further. A pano
    out on the water (on_open_water: a boat's) is never added, so a walk
    stops at the shore.

    Returns (nodes, edges) -- same shape nearby_nodes/tab.py's
    state["nodes"]/state["edges"] already use.
    """
    try:
        discovered = google_tile_panos(center_lat, center_lon, radius_m)
    except Exception as e:
        print(f"Google coverage lookup failed: {e}")
        return [], []
    # The pano nearest the center too, looked up directly: discovery keeps a
    # walked path's panos only every 12-24 m (_probe_line_gaps), so a small
    # radius inside a park can hold none of them, though one stands right
    # at the center.
    try:
        at_center = streetview.find_panorama(center_lat, center_lon, radius=min(max(radius_m, 15), 50))
    except Exception as e:
        print(f"Google search at the center failed: {e}")
        at_center = None
    if _is_official(at_center):
        discovered.setdefault(at_center.id, at_center)

    def dist(n):
        return _haversine_m(center_lat, center_lon, n["lat"], n["lon"])

    seeds = [{"key": node_key("google", p.id), "source": "google", "id": p.id,
              "lat": p.lat, "lon": p.lon, "heading": p.heading}
             for p in discovered.values()]
    seeds = [n for n in seeds if dist(n) <= radius_m]
    from streetview_to_3d.postprocess.water import occurrence_map
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
            metas = run_async(fetch_panos_by_id([k.split(":", 1)[1] for k in frontier]))
            next_frontier = []
            for key, meta in zip(frontier, metas):
                if not meta:
                    continue
                known = [n for n in meta["neighbors"] if node_key("google", n["id"]) in by_key]
                for n in known + ashore([n for n in meta["neighbors"] if n not in known]):
                    other_key = node_key("google", n["id"])
                    add_node({"key": other_key, "source": "google", "id": n["id"],
                              "lat": n["lat"], "lon": n["lon"], "heading": None})
                    fe = frozenset((key, other_key))
                    if fe not in edge_set:
                        edges.append((key, other_key))
                        edge_set.add(fe)
                    if (other_key not in visited and other_key not in next_frontier
                            and dist(by_key[other_key]) <= radius_m):
                        next_frontier.append(other_key)
            frontier = next_frontier

    print(f"expand_area: {len(nodes)} node(s) from {walks} walk(s), {len(discovered)} discovered, "
          f"{len(afloat)} out on the water left out")
    return nodes, edges
