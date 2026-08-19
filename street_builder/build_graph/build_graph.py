"""Turn fetch_nodes' per-dot candidate buckets into isolated, per-date
graphs (no GPU, no validation)."""
from services.geo import haversine_m
from street_builder.build_graph.date_ranking import DATE_TOP_N, date_connects, rank_dates
from street_builder.build_graph.fetch_nodes import fetch_corridor_nodes

# Max distance between two same-date candidates to count as a valid hop.
EDGE_MAX_DIST_M = 18.0

# Cap per node -- Apple can be ~1.2m between frames, so even after
# TOP_PANOS_PER_DOT, a node can still have same-date neighbors piling in
# from several nearby dots. Keeps every node's own fan-out bounded
# without dropping any node from the graph.
MAX_EDGES_PER_NODE = 20

# Per dot, per date, how many of that date's own closest panos to keep.
# This is the actual fix for a dense capture date (Apple's ~1.2m frame
# spacing) trapping the search in one dot's worth of redundant same-spot
# candidates before it ever reaches the next dot -- at most this many
# real options exist at any dot, for any date, period, regardless of how
# dense that date's real coverage is.
TOP_PANOS_PER_DOT = 5


def _cap_bucket_for_date(bucket, date, dot_lat, dot_lon, top_n):
    """This dot's own panos of ONE date, closest-first, capped to top_n."""
    same_date = [n for n in bucket if n["date"] == date]
    same_date.sort(key=lambda n: haversine_m(dot_lat, dot_lon, n["lat"], n["lon"]))
    return same_date[:top_n]


def _build_edges(nodes):
    """Same-date pairs within EDGE_MAX_DIST_M, capped at MAX_EDGES_PER_NODE
    nearest per node. No date filtering needed -- every node given here
    already IS the same date (see build_corridor_graphs)."""
    edges = {n["key"]: [] for n in nodes}
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            dist = haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])
            if dist <= EDGE_MAX_DIST_M:
                edges[a["key"]].append((b["key"], dist))
                edges[b["key"]].append((a["key"], dist))
    for key in edges:
        edges[key].sort(key=lambda pair: pair[1])
        edges[key] = edges[key][:MAX_EDGES_PER_NODE]
    return edges


def build_corridor_graphs(corridor_edges, start_lat, start_lon, goals,
                           top_n_dates=DATE_TOP_N, top_per_dot=TOP_PANOS_PER_DOT):
    """Build up to top_n_dates ISOLATED per-date graphs along the corridor
    traced by corridor_edges (see fetch_corridor_nodes -- real, already-
    connected edges, not an assumed-linear list).

    Each date graph is built independently: every dot gets capped to that
    date's own top_per_dot closest panos before edges are built, so a
    dense capture date can never trap the search in one dot's worth of
    redundant same-spot candidates -- at most top_per_dot real options
    exist at any dot, for any date.

    Ranked best-first by coverage span (see date_ranking.rank_dates), and
    a candidate date only counts toward top_n_dates if its own edges can
    structurally reach from the start toward at least one goal (see
    date_ranking.date_connects) -- checked AFTER building that date's own
    graph, since reachability depends on real edges, not just raw dot
    coverage.

    Returns (date_graphs, points). date_graphs: [{"date": str, "nodes":
    [...], "edges": {...}}, ...], ranked best first, each graph already
    isolated to its own date. points: the interpolated spine (see
    fetch_corridor_nodes), shared across every date graph -- it's the
    date-independent coverage reference the algorithm scores progress
    against.
    """
    buckets, points = fetch_corridor_nodes(corridor_edges)
    ranked_dates = rank_dates(buckets)

    date_graphs = []
    for date in ranked_dates:
        if len(date_graphs) >= top_n_dates:
            break

        nodes = []
        for i, bucket in buckets.items():
            dot_lat, dot_lon = points[i]
            nodes.extend(_cap_bucket_for_date(bucket, date, dot_lat, dot_lon, top_per_dot))
        if len(nodes) < 2:
            continue

        edges = _build_edges(nodes)
        nodes = [n for n in nodes if edges.get(n["key"])]  # drop isolated (never testable)
        if len(nodes) < 2:
            continue

        if not date_connects(nodes, edges, start_lat, start_lon, goals):
            continue

        date_graphs.append({"date": date, "nodes": nodes, "edges": edges})

    return date_graphs, points
