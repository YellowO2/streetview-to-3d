"""Gather every Google + Apple panorama along a street corridor (no GPU)."""
import asyncio

from services.geo import haversine_m
from services.streetview_fetch import fetch_pano_by_id, format_date
from street_builder.map_selection.candidates import MAX_NODES, apple_tile_panos, nearby_nodes, node_key

# Catchment radius for candidate lookup around each real selection-graph
# node. Real Street View node spacing is commonly ~10-20m, so this is
# generous enough to catch nearby Apple/Google coverage without one
# dot's search reaching into a neighboring dot's own territory.
POINT_MAX_DIST_M = 5.0


def corridor_points(edges) -> tuple[list[tuple[float, float]], dict[int, list[int]]]:
    """Real (lat, lon) dots + structural adjacency straight from the
    corridor's own already-confirmed edges -- edges: list of ((lat1,
    lon1), (lat2, lon2)) pairs, each a real, already-connected pair (not
    a single ordered polyline: the corridor can branch or loop, so edges
    aren't assumed to trace one path in list order).

    No synthetic in-between sampling -- a dot is exactly one real
    selection-graph node, not an interpolated point along a straight
    line between two of them. Two edges sharing the exact same (lat,
    lon) endpoint (a branch point, where two real clicked edges meet at
    the same node) collapse into the SAME dot, so the corridor's own
    branch structure carries through into the dot graph rather than each
    edge getting a disconnected copy of that point.

    Returns (points, adjacency). adjacency: {dot_index: [neighbor_dot_index,
    ...]} -- the corridor's own real dot-to-dot structure, independent of
    which real panos end up at either dot. This is what the pathfind
    algorithm walks dot-by-dot over (see
    street_builder/reconstruction/walk_graph.py).
    """
    points: list[tuple[float, float]] = []
    adjacency: dict[int, list[int]] = {}
    index_by_latlon: dict[tuple[float, float], int] = {}

    def dot_for(latlon):
        idx = index_by_latlon.get(latlon)
        if idx is None:
            idx = len(points)
            points.append(latlon)
            adjacency[idx] = []
            index_by_latlon[latlon] = idx
        return idx

    def connect(i, j):
        if j not in adjacency[i]:
            adjacency[i].append(j)
        if i not in adjacency[j]:
            adjacency[j].append(i)

    for (lat1, lon1), (lat2, lon2) in edges:
        connect(dot_for((lat1, lon1)), dot_for((lat2, lon2)))

    return points, adjacency


def fetch_corridor_nodes(edges, max_dist_m: float = POINT_MAX_DIST_M):
    """Every Google + Apple pano within max_dist_m of any real corridor
    node (see corridor_points).

    - For each point: nearby_nodes (Google stops) + apple_tile_panos
      (Apple), both metadata only. A newly-seen Google stop gets one extra
      fetch_pano_by_id call for its real historical dates (one graph node
      per date); already-seen stops/panos aren't re-fetched.

    Each real pano is assigned to exactly one dot -- the first (lowest-
    index) dot it's found within max_dist_m of, tracked globally via
    seen_google_ids/seen_apple_ids so a pano already claimed by an earlier
    dot never gets double-counted by a later one. The "first dot wins"
    rule resolves the rare case of a pano sitting within range of two
    real dots at once.

    Returns (buckets, points, adjacency): buckets is {point_index: [{key,
    source, id, lat, lon, date}, ...]} -- each dot's own separate set of
    panos, not one shared pool. points is the corridor's own real node
    list. adjacency is the dot-to-dot structural graph (see
    corridor_points).
    """
    points, adjacency = corridor_points(edges)

    buckets = {i: [] for i in range(len(points))}
    seen_google_ids = set()
    seen_apple_ids = set()

    for i, (lat, lon) in enumerate(points):
        try:
            google_candidates, _ = nearby_nodes(lat, lon, radius_m=max_dist_m, max_nodes=MAX_NODES)
        except Exception as e:
            print(f"Google lookup failed near ({lat}, {lon}): {e}")
            google_candidates = []
        for gc in google_candidates:
            if gc["id"] in seen_google_ids:
                continue
            seen_google_ids.add(gc["id"])
            try:
                meta = asyncio.run(fetch_pano_by_id(gc["id"]))
            except Exception as e:
                print(f"Google date lookup failed for {gc['id']}: {e}")
                continue
            if not meta:
                continue
            for entry in meta["dates"]:
                buckets[i].append({
                    "key": node_key("google", entry["id"]), "source": "google", "id": entry["id"],
                    "lat": gc["lat"], "lon": gc["lon"], "date": entry["label"],
                })

        try:
            apple_candidates = apple_tile_panos(lat, lon)
        except Exception as e:
            print(f"Apple lookup failed near ({lat}, {lon}): {e}")
            apple_candidates = {}
        for p in apple_candidates.values():
            if p.id in seen_apple_ids:
                continue
            if haversine_m(lat, lon, p.lat, p.lon) > max_dist_m:
                continue
            seen_apple_ids.add(p.id)
            buckets[i].append({
                "key": node_key("apple", p.id), "source": "apple", "id": p.id,
                "lat": p.lat, "lon": p.lon, "date": format_date(p.date),
                "_pano": p,  # kept for download_lookaround (needs the object, not just the id)
            })

    return buckets, points, adjacency
