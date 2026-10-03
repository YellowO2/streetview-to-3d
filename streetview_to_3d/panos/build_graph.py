"""Turn fetch_nodes' per-dot candidate buckets into isolated per-date graphs (no GPU)."""
from streetview_to_3d.common.geo import haversine_m
from streetview_to_3d.panos.date_ranking import DATE_TOP_N, date_connects, rank_dates
from streetview_to_3d.panos.fetch_nodes import fetch_corridor_nodes

# Per dot, per date, how many of the closest panos to keep (and so download and rate):
# enough for a fallback when the closest is bad; more is the same spot again.
TOP_PANOS_PER_DOT = 3


def cap_bucket_for_date(bucket, date, dot_lat, dot_lon, top_n):
    """This dot's own panos of ONE date, closest-first, capped to top_n."""
    same_date = [n for n in bucket if n["date"] == date]
    same_date.sort(key=lambda n: haversine_m(dot_lat, dot_lon, n["lat"], n["lon"]))
    return same_date[:top_n]


def build_corridor_graphs(corridor_edges, start_lat, start_lon, goals,
                           top_n_dates=DATE_TOP_N, top_per_dot=TOP_PANOS_PER_DOT):
    """Up to top_n_dates per-date graphs along the corridor, best first (rank_dates).

    Each dot keeps that date's top_per_dot closest panos; a date counts only if its
    dots reach from the start toward a goal (date_connects), checked after capping.
    Returns (date_graphs, points, adjacency, elevations); date_graphs is
    [{"date", "dot_candidates": {dot: [panos]}}], points/adjacency are shared by all dates.
    """
    buckets, points, adjacency, elevations = fetch_corridor_nodes(corridor_edges)
    ranked_dates = rank_dates(buckets)

    date_graphs = []
    for date in ranked_dates:
        if len(date_graphs) >= top_n_dates:
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

        date_graphs.append({"date": date, "dot_candidates": dot_candidates})

    return date_graphs, points, adjacency, elevations
