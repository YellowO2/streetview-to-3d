"""Build one graph of the corridor's dots per worthwhile capture date."""
from streetview_to_3d.gather_panoramas.tools.dates import TOP_PANOS_PER_DOT, cap_bucket_for_date, date_connects, rank_dates

# How many per-date graphs to build; a date covering little can still patch.
DATE_TOP_N = 5


def date_graphs(nodes, start, goals, top_n_dates=DATE_TOP_N, top_per_dot=TOP_PANOS_PER_DOT):
    """Up to top_n_dates per-date graphs along the corridor, best first (rank_dates).

    nodes: the corridor's (buckets, points, adjacency, elevations), from fetch_corridor_nodes;
    start: (lat, lon); goals: [(lat, lon)], the other selected nodes. Each dot keeps that
    date's top_per_dot closest panos; a date counts only if its dots reach from the start
    toward a goal (date_connects), checked after capping. Returns {date_graphs, points,
    adjacency, elevations}; date_graphs is [{"date", "dot_candidates": {dot: [panos]}}],
    points/adjacency are shared by all dates. Raises ValueError when there is nothing to walk.
    """
    if not goals:
        raise ValueError("Need at least one goal (a second selected node).")
    start_lat, start_lon = start
    buckets, points, adjacency, elevations = nodes
    ranked_dates = rank_dates(buckets)

    graphs = []
    for date in ranked_dates:
        if len(graphs) >= top_n_dates:
            break

        dot_candidates = {}
        for i, bucket in buckets.items():
            dot_lat, dot_lon = points[i]
            capped = cap_bucket_for_date(bucket, date, dot_lat, dot_lon, top_per_dot)
            if capped:
                dot_candidates[i] = capped
        if not dot_candidates:
            continue

        if not date_connects(dot_candidates, adjacency, points, start_lat, start_lon, goals):
            continue

        graphs.append({"date": date, "dot_candidates": dot_candidates})

    if not graphs:
        raise ValueError("No date reaches from the start toward any goal -- not enough connected candidates.")
    return {"date_graphs": graphs, "points": points, "adjacency": adjacency, "elevations": elevations}
