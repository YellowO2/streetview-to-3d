"""Turn fetch_nodes' candidate list into a graph (no GPU, no validation)."""
from services.geo import haversine_m
from street_builder.map_selection.candidates import APPLE_CANDIDATE_MAX_DIST_M
from street_builder.pathfinding.fetch_nodes import fetch_corridor_nodes

# Max distance between two same-date candidates to count as a valid hop.
EDGE_MAX_DIST_M = APPLE_CANDIDATE_MAX_DIST_M


def build_corridor_graph(start_lat, start_lon, end_lat, end_lon):
    """Build a candidate graph: nodes = every panorama along the corridor,
    edges = same-date pairs within EDGE_MAX_DIST_M. Edges are untested
    candidates -- walk_graph.py runs the real DA3 check.

    Returns (nodes, edges). edges: {key -> [(other_key, dist_m), ...]},
    sorted nearest-first.
    """
    nodes = fetch_corridor_nodes(start_lat, start_lon, end_lat, end_lon)

    by_date = {}
    for n in nodes:
        by_date.setdefault(n["date"], []).append(n)

    edges = {n["key"]: [] for n in nodes}
    for group in by_date.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                dist = haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])
                if dist <= EDGE_MAX_DIST_M:
                    edges[a["key"]].append((b["key"], dist))
                    edges[b["key"]].append((a["key"], dist))

    for key in edges:
        edges[key].sort(key=lambda pair: pair[1])

    return nodes, edges
