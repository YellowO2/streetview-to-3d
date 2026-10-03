"""Gather every Google panorama along a street corridor (no GPU)."""
from streetview_to_3d.common.scene import DisjointSet
from streetview_to_3d.common.geo import haversine_m
from streetview_to_3d.common.streetview_fetch import fetch_panos_by_id, google_node, run_async

# Selected panos this close collapse into one dot: the same spot captured twice
# (both directions, a path beside a road). Normal spacing along one path is 8-11 m.
MERGE_DIST_M = 8.0

# Separate graphs closer than this get one bridge edge: about the reach of a real link.
BRIDGE_DIST_M = 15.0


def corridor_points(edges):
    """Dots and their adjacency from the corridor's real edges, [((lat, lon, id), (lat, lon, id))].

    Greedy, not transitive: each unclaimed pano seeds a dot at its own position and claims
    every unclaimed pano within MERGE_DIST_M (union-find chain-collapsed a dense 72 m stretch).
    Returns (points, adjacency, members); members: per dot, its pano ids, seed first.
    """
    nodes = {}  # pano id -> (lat, lon), in first-seen order
    for a, b in edges:
        for lat, lon, pano_id in (a, b):
            nodes.setdefault(pano_id, (lat, lon))

    points, members, dot_of = [], [], {}
    for seed, pos in nodes.items():
        if seed in dot_of:
            continue
        cluster = [m for m, p in nodes.items() if m not in dot_of and haversine_m(*pos, *p) <= MERGE_DIST_M]
        for m in cluster:
            dot_of[m] = len(points)
        points.append(pos)
        members.append(cluster)

    adjacency: dict[int, list[int]] = {i: [] for i in range(len(points))}

    def connect(i, j):
        if i == j:
            return
        if j not in adjacency[i]:
            adjacency[i].append(j)
        if i not in adjacency[j]:
            adjacency[j].append(i)

    for (_, _, a), (_, _, b) in edges:
        connect(dot_of[a], dot_of[b])

    bridge_components(points, adjacency, connect)
    return points, adjacency, members


def bridge_components(points, adjacency, connect, max_gap_m: float = BRIDGE_DIST_M):
    """Join graphs Google never linked where they come within max_gap_m, so the walk can
    try a DA3 pair test there. Kruskal over the closest dot pairs: each two groups get
    their nearest crossing and no more.
    """
    sets = DisjointSet(len(points))
    for i, ns in adjacency.items():
        for j in ns:
            sets.union(i, j)

    pairs = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if sets.find(i) != sets.find(j):
                d = haversine_m(*points[i], *points[j])
                if d <= max_gap_m:
                    pairs.append((d, i, j))
    for _, i, j in sorted(pairs):
        if sets.find(i) != sets.find(j):
            sets.union(i, j)
            connect(i, j)


def fetch_corridor_nodes(edges):
    """Every capture date of the selected panos folded into each dot, one candidate per date.

    Returns (buckets, points, adjacency, elevations): buckets is {dot: [node dicts with date,
    pitch, roll]}, elevations is metres above sea level per dot (saves placement a re-fetch).
    """
    points, adjacency, members = corridor_points(edges)
    ids = [pano_id for m in members for pano_id in m]
    meta_by_id = dict(zip(ids, run_async(fetch_panos_by_id(ids))))

    buckets = {i: [] for i in range(len(points))}
    elevations = [None] * len(points)
    for i, pano_ids in enumerate(members):
        seen = set()
        for pano_id in pano_ids:
            meta = meta_by_id.get(pano_id)
            if not meta:
                continue
            if elevations[i] is None:
                elevations[i] = meta.get("elevation")
            for entry in meta["dates"]:
                if entry["id"] in seen:
                    continue
                seen.add(entry["id"])
                # each date's own pose: older captures are separate drives
                buckets[i].append(google_node(
                    entry["id"], entry.get("lat", meta["lat"]), entry.get("lon", meta["lon"]),
                    entry.get("heading", meta.get("heading")), date=entry["label"],
                    pitch=entry.get("pitch", meta.get("pitch")), roll=entry.get("roll", meta.get("roll"))))

    return buckets, points, adjacency, elevations
