"""Turn fetch_nodes' per-dot candidate buckets into isolated, per-date
graphs (no GPU, no validation)."""
from services.geo import haversine_m
from street_builder.build_graph.date_ranking import DATE_TOP_N, date_connects, rank_dates
from street_builder.build_graph.fetch_nodes import fetch_corridor_nodes

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


def build_corridor_graphs(corridor_edges, start_lat, start_lon, goals,
                           top_n_dates=DATE_TOP_N, top_per_dot=TOP_PANOS_PER_DOT):
    """Build up to top_n_dates ISOLATED per-date graphs along the corridor
    traced by corridor_edges (see fetch_corridor_nodes -- real, already-
    connected edges, not an assumed-linear list).

    Each date graph is built independently: every dot gets capped to that
    date's own top_per_dot closest panos, so a dense capture date can
    never trap the search in one dot's worth of redundant same-spot
    candidates -- at most top_per_dot real options exist at any dot, for
    any date.

    Unlike the client-graph edges the map selector gives us, there's no
    real pano-to-pano edge list here anymore -- the pathfind algorithm
    (street_builder/reconstruction/walk_graph.py) walks the shared
    dot-to-dot adjacency directly (dot i to dot i+1, branching wherever
    the corridor itself branches), only checking real pano distance
    dynamically for its skip-one-empty-dot fallback.

    Ranked best-first by coverage span (see date_ranking.rank_dates), and
    a candidate date only counts toward top_n_dates if its own dots can
    structurally reach from the start toward at least one goal (see
    date_ranking.date_connects) -- checked AFTER capping, since
    reachability depends on which dots actually end up with candidates.

    Returns (date_graphs, points, adjacency). date_graphs: [{"date": str,
    "dot_candidates": {dot_index: [panos]}}, ...], ranked best first, each
    graph already isolated to its own date and containing only its own
    non-empty dots. points/adjacency: shared across every date graph --
    the corridor's own dot positions and structure (see
    fetch_nodes.interpolate_points).
    """
    buckets, points, adjacency = fetch_corridor_nodes(corridor_edges)
    ranked_dates = rank_dates(buckets)

    date_graphs = []
    for date in ranked_dates:
        if len(date_graphs) >= top_n_dates:
            break

        dot_candidates = {}
        for i, bucket in buckets.items():
            dot_lat, dot_lon = points[i]
            capped = _cap_bucket_for_date(bucket, date, dot_lat, dot_lon, top_per_dot)
            if capped:
                dot_candidates[i] = capped
        if not dot_candidates:
            continue

        if not date_connects(dot_candidates, adjacency, points, start_lat, start_lon, goals):
            continue

        date_graphs.append({"date": date, "dot_candidates": dot_candidates})

    return date_graphs, points, adjacency
