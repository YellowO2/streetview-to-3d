"""Low-level fetch of real Street View / Look Around panoramas near a
location. Used by the map picker and by street_builder/pathfinding/.
"""
from streetlevel import streetview
from streetlevel.geo import wgs84_to_tile_coord
from streetlevel.lookaround import lookaround as apple_lookaround

from services.geo import haversine_m as _haversine_m

# Both services publish coverage on zoom-17 Slippy Map tiles.
_TILE_ZOOM = 17


def _tile_neighborhood(lat, lon):
    tx, ty = wgs84_to_tile_coord(lat, lon, _TILE_ZOOM)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            yield tx + dx, ty + dy


def google_tile_panos(lat, lon):
    """All Street View panos on the 3x3 tile neighborhood around (lat, lon), keyed by id."""
    seen = {}
    for tx, ty in _tile_neighborhood(lat, lon):
        for p in streetview.get_coverage_tile(tx, ty):
            seen[p.id] = p
    return seen


def apple_tile_panos(lat, lon):
    """All Look Around panos on the 3x3 tile neighborhood around (lat, lon), keyed by id."""
    seen = {}
    for tx, ty in _tile_neighborhood(lat, lon):
        tile = apple_lookaround.get_coverage_tile(tx, ty)
        for p in tile.panos:
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
    no date (tile listing doesn't carry it; see pathfinding/fetch_nodes.py
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


# Max distance a candidate can be from a node and still count as "at" it.
# ~2.5x the measured ~10m real node spacing. Shared by
# street_builder/pathfinding/ and street_builder/reconstruct.py.
APPLE_CANDIDATE_MAX_DIST_M = 25.0
