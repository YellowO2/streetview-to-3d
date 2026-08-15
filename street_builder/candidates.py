"""Fetch real Street View / Look Around panorama nodes near a location, for the
street-builder map picker. Normalizes both sources into one shape so the map
UI and Gradio wiring don't need to know which service a node came from.

Deliberately independent of app.py (no imports from it) so this stays usable
on its own, e.g. from a future non-Gradio script. Does import from services/
for small shared utilities (geo math) that app.py also uses -- that's a
one-way dependency on plain helpers, not on app.py itself.
"""
from streetlevel import streetview
from streetlevel.geo import wgs84_to_tile_coord
from streetlevel.lookaround import lookaround as apple_lookaround

from services.geo import haversine_m as _haversine_m

# Both services publish coverage on zoom-17 Slippy Map tiles. Fetching the
# center tile plus its 8 neighbors (not just the one tile) avoids gaps right
# at a tile edge, matching the radius app.py already uses for its Apple
# neighbor lookup.
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
    """All Look Around panos on the 3x3 tile neighborhood around (lat, lon), keyed by id.

    Not used by the street-builder map picker (that's Google-only — see
    nearby_nodes below). Kept here for the later step where support panos
    get gathered automatically near each chosen Google node.
    """
    seen = {}
    for tx, ty in _tile_neighborhood(lat, lon):
        tile = apple_lookaround.get_coverage_tile(tx, ty)
        for p in tile.panos:
            seen[p.id] = p
    return seen


def _node_key(source, pano_id):
    return f"{source}:{pano_id}"


# Google alone is far sparser than Google+Apple combined (the old combined
# fetch hit 9000 raw hits / 1500 within 150m in dense downtown Singapore;
# Google alone is ~30 within 150m at the same spot), so this can afford a
# wider default radius and a much lower safety cap.
DEFAULT_RADIUS_M = 350
MAX_NODES = 200


def nearby_nodes(lat, lon, radius_m=DEFAULT_RADIUS_M, max_nodes=MAX_NODES):
    """Distance-sorted Google Street View nodes within radius_m of (lat, lon), plus the
    edges (pairs of node keys) Street View's own coverage graph links between them.

    Nodes are the street-builder chain's only source — Apple Look Around is
    gathered separately, later, as automatic support imagery per selected
    node, not as something you click on this map.

    Returns (nodes, edges). Each node: {key, source, id, lat, lon, heading}.
    Each edge: (key_a, key_b), only included when both ends survive the
    radius/cap filtering (an edge to a node outside that set isn't
    selectable, so there's nothing useful to draw it against).
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
            "key": _node_key("google", p.id),
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
            other_key = _node_key("google", link.pano.id)
            if other_key in kept_keys and other_key != key:
                edges.add(tuple(sorted((key, other_key))))

    return nodes, sorted(edges)
