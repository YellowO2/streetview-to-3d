"""Select every real Street View graph inside an area, walked link by link.

Each graph is walked on its own from the panos the map tiles list; panos out on open water
are left out. Every lookup is cached, so a redrawn area fetches only what it adds.
"""
import logging

import numpy as np
from streetlevel import streetview

from street_builder.map_selection.tools.google_tile_panos import google_tile_panos
from street_builder.map_selection.tools.street_view import is_official
from streetview_to_3d.common.Pano import Pano
from streetview_to_3d.panos.tools.fetch_panos_by_id import fetch_panos_by_id
from streetview_to_3d.panos.tools.google_node import google_node
from streetview_to_3d.postprocess.world.tools.occurrence_map import WET, occurrence_map
from streetview_to_3d.tools.geo import haversine_m as _haversine_m, local_m_to_latlon, to_metres
from streetview_to_3d.tools.run_async import run_async

logger = logging.getLogger(__name__)

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
            logger.warning(f"Google search at the center failed: {e}")
            return None
        _centre_cache[key] = p if is_official(p) else None
    return _centre_cache[key]


# A discovered pano this close to a walked one is the same place, not a new walk.
_SAME_PLACE_M = 3.0

# A pano with water this far all round is a boat's: too little shore for DA3, never walked to.
_OPEN_WATER_M = 30.0


def _on_open_water(often):
    """f(lats, lons) -> whether each is out on the water; often(lat, lon)
    -> how often water there, 0-1 (postprocess/world/tools/occurrence_map.py).
    Nowhere, if the map cannot be had."""

    def f(lat, lon):
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        try:
            out = often(lat, lon) >= WET
            for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                out &= often(*local_m_to_latlon(_OPEN_WATER_M * np.cos(a), _OPEN_WATER_M * np.sin(a), lat, lon)) >= WET
            return out
        except OSError as e:
            logger.warning(f"Water map lookup failed: {e}")
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
        here = to_metres(center_lat, [(center_lat, center_lon)])[0]
        corners = to_metres(center_lat, area)
        shape = shapely.Polygon(corners).buffer(0)
        shapely.prepare(shape)
        radius_m = float(np.linalg.norm(corners - here, axis=1).max())

        def within(nodes):
            xy = to_metres(center_lat, [(n["lat"], n["lon"]) for n in nodes])
            return shapely.contains_xy(shape, xy[:, 0], xy[:, 1]) if len(nodes) else np.zeros(0, bool)
    else:
        def within(nodes):
            return np.array([_haversine_m(center_lat, center_lon, n["lat"], n["lon"]) <= radius_m
                             for n in nodes], bool)
    try:
        discovered = google_tile_panos(center_lat, center_lon, radius_m)
    except Exception as e:
        logger.warning(f"Google coverage lookup failed: {e}")
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
    at_sea = _on_open_water(occurrence_map())
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
                known = [n for n in meta["neighbors"] if Pano.key_of("google", n["id"]) in by_key]
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

    logger.info(f"expand_area: {len(nodes)} node(s) from {walks} walk(s), {len(discovered)} discovered, "
                f"{len(afloat)} out on the water left out")
    return nodes, edges
