"""Turn fetch_nodes' candidate list into a graph (no GPU, no validation)."""
from services.geo import haversine_m
from street_builder.build_graph.fetch_nodes import fetch_corridor_nodes

# Max distance between two same-date candidates to count as a valid hop.
# Independent of APPLE_CANDIDATE_MAX_DIST_M (a different concept -- matching
# an Apple candidate to a Google node's position, not a graph edge cap).
EDGE_MAX_DIST_M = 18.0

# Cap per node -- Apple can be ~1.2m between frames, so one node can have
# 100+ same-date neighbors within EDGE_MAX_DIST_M. Keeps every node's own
# fan-out bounded without dropping any node from the graph.
MAX_EDGES_PER_NODE = 20


def build_corridor_graph(waypoints):
    """Build a candidate graph: nodes = every panorama along the corridor
    traced by waypoints (see fetch_corridor_nodes), edges = same-date pairs
    within EDGE_MAX_DIST_M, capped at MAX_EDGES_PER_NODE nearest per node.
    Edges are untested candidates -- walk_graph.py runs the real DA3 check.

    Returns (nodes, edges, points). edges: {key -> [(other_key, dist_m), ...]},
    sorted nearest-first. points: the interpolated spine (see fetch_corridor_nodes).
    """
    nodes, points = fetch_corridor_nodes(waypoints)

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
        edges[key] = edges[key][:MAX_EDGES_PER_NODE]

    return nodes, edges, points
