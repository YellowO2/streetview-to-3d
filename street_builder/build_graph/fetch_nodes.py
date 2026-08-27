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

# Real selection-graph nodes within this distance of each other collapse
# into ONE dot (see corridor_points) -- real coverage is frequently
# double/triple-sampled at the same real spot (a road captured in both
# directions, a pedestrian path running alongside a road, a backpack
# capture re-walking the same stretch), which otherwise shows up as
# several near-duplicate dots each independently competing for a date/
# candidates instead of one dot with the union of everyone's real
# candidates. 8.0, not POINT_MAX_DIST_M's 5.0 -- checked real NTU data
# (tests/visualize_date_cover.py) after the first pass at 5.0: normal
# real node spacing along a single path is itself often 8-11m, so 5.0
# already correctly avoided merging genuinely distinct waypoints; 8.0
# is a deliberate small step up, not matched to the catchment radius on
# principle.
MERGE_DIST_M = 8.0


def corridor_points(edges) -> tuple[list[tuple[float, float]], dict[int, list[int]]]:
    """Real (lat, lon) dots + structural adjacency straight from the
    corridor's own already-confirmed edges -- edges: list of ((lat1,
    lon1), (lat2, lon2)) pairs, each a real, already-connected pair (not
    a single ordered polyline: the corridor can branch or loop, so edges
    aren't assumed to trace one path in list order).

    No synthetic in-between sampling -- a dot starts as exactly one real
    selection-graph node, not an interpolated point along a straight
    line between two of them. Real selection-graph nodes within
    MERGE_DIST_M of each other then collapse into the SAME dot (see
    MERGE_DIST_M's own docstring) -- transitively: A-B within range and
    B-C within range merges all three into one dot even if A-C alone
    wouldn't qualify, same as two edges sharing an exact (lat, lon)
    endpoint always did. A merged dot's own position is the centroid of
    everything folded into it. Rare, deliberately-accepted edge case: a
    long enough near-straight run of sub-threshold gaps (e.g. two
    parallel captures the whole length of a long road) can chain into
    one dot whose centroid sits further from either original end than
    MERGE_DIST_M -- POINT_MAX_DIST_M's own candidate catchment (fetched
    from the centroid) could then miss real candidates out at the ends.

    Returns (points, adjacency). adjacency: {dot_index: [neighbor_dot_index,
    ...]} -- the corridor's own real dot-to-dot structure, independent of
    which real panos end up at either dot. This is what the pathfind
    algorithm walks dot-by-dot over (see
    street_builder/reconstruction/walk_graph.py).
    """
    raw_points: list[tuple[float, float]] = []
    raw_index_by_latlon: dict[tuple[float, float], int] = {}

    def raw_index_for(latlon):
        idx = raw_index_by_latlon.get(latlon)
        if idx is None:
            idx = len(raw_points)
            raw_points.append(latlon)
            raw_index_by_latlon[latlon] = idx
        return idx

    raw_edges = [(raw_index_for((lat1, lon1)), raw_index_for((lat2, lon2))) for (lat1, lon1), (lat2, lon2) in edges]

    # Union-find: any two raw nodes within MERGE_DIST_M join the same
    # dot, transitively. O(n^2) distance checks -- fine at real-world
    # selection-graph sizes (a few thousand nodes at most).
    parent = list(range(len(raw_points)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(raw_points)):
        lat_i, lon_i = raw_points[i]
        for j in range(i + 1, len(raw_points)):
            if haversine_m(lat_i, lon_i, *raw_points[j]) <= MERGE_DIST_M:
                union(i, j)

    cluster_members: dict[int, list[int]] = {}
    for i in range(len(raw_points)):
        cluster_members.setdefault(find(i), []).append(i)

    points: list[tuple[float, float]] = []
    dot_index_by_root: dict[int, int] = {}
    for root, members in cluster_members.items():
        lat = sum(raw_points[m][0] for m in members) / len(members)
        lon = sum(raw_points[m][1] for m in members) / len(members)
        dot_index_by_root[root] = len(points)
        points.append((lat, lon))

    adjacency: dict[int, list[int]] = {i: [] for i in range(len(points))}

    def connect(i, j):
        if i == j:
            return
        if j not in adjacency[i]:
            adjacency[i].append(j)
        if i not in adjacency[j]:
            adjacency[j].append(i)

    for a, b in raw_edges:
        connect(dot_index_by_root[find(a)], dot_index_by_root[find(b)])

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
