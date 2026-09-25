"""Low-level fetch of real Street View / Look Around panoramas near a
location. Used by the map picker and by build_street_graph/.
"""
from streetlevel import streetview
from streetlevel.geo import wgs84_to_tile_coord
from streetlevel.lookaround import lookaround as apple_lookaround

from services.geo import haversine_m as _haversine_m
from services.streetview_fetch import fetch_pano_by_id, run_async

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


def expand_area(center_lat, center_lon, radius_m, max_nodes=2000):
    """Auto-discover the real Street View graph within radius_m of
    (center_lat, center_lon) -- the same real-link expansion
    map_selection/tab.py's _augment_real_links does for one clicked node,
    just driven by a BFS loop instead of a person clicking node by node.
    Lets a whole area (a campus, a district) be selected without clicking
    every node by hand -- feed the result straight in as corridor_edges,
    same shape a manually-built selection already produces.

    This is a PURE single-source BFS: nearby_nodes is used only to locate
    the one real pano nearest to the center point (a geometric tile scan
    can't tell us that without first knowing a real pano id to start
    from), and every other node in the result is discovered strictly by
    walking real per-pano links (fetch_pano_by_id) outward from that one
    seed. A node only ever gets ADDED because it was found as a real-link
    neighbor of an already-visited, in-radius node -- never because it
    happened to be geometrically nearby. That guarantees the whole
    output is one connected component by construction, exactly like
    a person standing at the seed and clicking outward one linked node
    at a time, radius_m capping how far they walk.

    This matters: dumping every geometrically-nearby node in as extra
    seeds (the earlier version of this function did) can silently return
    MULTIPLE disconnected components -- e.g. a center point sitting in
    the middle of a triangular block, equidistant from 3 unconnected
    streets, would have a tile scan grab panos from all 3 (all within
    radius) even though they share no real link. A pure BFS from one
    start node instead correctly returns just the one street the start
    node actually belongs to.

    Only ever expanding FROM a node that's still within radius_m --
    anything found just past the boundary is kept as a leaf in the
    result but never itself expanded further.

    Apple has no real link data (same limitation _augment_real_links
    documents) -- this only ever discovers Google coverage. Not a
    problem in practice: Apple candidates still get pulled in later,
    per corridor dot, by fetch_corridor_nodes's own radius lookup --
    exactly how a manually-clicked selection already works today, since
    clicks were always Google-link-driven too.

    Returns (nodes, edges) -- same shape nearby_nodes/tab.py's
    state["nodes"]/state["edges"] already use.
    """
    seed_nodes, _ = nearby_nodes(center_lat, center_lon, radius_m=min(radius_m, DEFAULT_RADIUS_M))
    google_seeds = [n for n in seed_nodes if n["key"].startswith("google:")]
    if not google_seeds:
        # Nothing to walk real links from -- either nothing nearby at
        # all, or only Apple coverage nearby (no real link data to BFS).
        return (seed_nodes[:1], []) if seed_nodes else ([], [])

    start_node = min(google_seeds, key=lambda n: _haversine_m(center_lat, center_lon, n["lat"], n["lon"]))

    nodes = [start_node]
    edges = []
    by_key = {start_node["key"]: start_node}
    edge_set = set()

    visited = set()
    queue = [start_node["key"]]
    while queue and len(nodes) < max_nodes:
        key = queue.pop(0)
        if key in visited:
            continue
        visited.add(key)

        pano_id = key.split(":", 1)[1]
        try:
            meta = run_async(fetch_pano_by_id(pano_id))
        except Exception as e:
            print(f"expand_area: link fetch failed for {pano_id}: {e}")
            continue
        if not meta:
            continue

        for n in meta["neighbors"]:
            other_key = node_key("google", n["id"])
            if other_key not in by_key:
                new_node = {"key": other_key, "source": "google", "id": n["id"],
                            "lat": n["lat"], "lon": n["lon"], "heading": None}
                nodes.append(new_node)
                by_key[other_key] = new_node
            fe = frozenset((key, other_key))
            if fe not in edge_set:
                edges.append((key, other_key))
                edge_set.add(fe)
            if (other_key not in visited
                    and _haversine_m(center_lat, center_lon, by_key[other_key]["lat"], by_key[other_key]["lon"]) <= radius_m):
                queue.append(other_key)

    return nodes, edges



