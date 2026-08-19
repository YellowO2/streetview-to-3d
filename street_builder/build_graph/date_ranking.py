"""Given the gathered per-dot candidate buckets, decide which dates are
worth building a real graph for -- coverage ranking + structural
reachability, no GPU, no DA3. This is a "build the candidate pool"
concern, not "solve the graph" (see
street_builder/reconstruction/walk_graph.py for the actual algorithm).
"""
from services.geo import haversine_m

# Dates kept, ranked by coverage span. Single source of truth for how many
# isolated per-date graphs build_corridor_graphs ever builds.
DATE_TOP_N = 5

# Mirrors the pathfind algorithm's own start_zone_m/point_cover_tolerance_m
# defaults -- used here only to pre-check whether a date's own same-date
# edges can even reach from a start-zone root to a goal-zone node, before
# spending a download (let alone a GPU test) on it.
START_ZONE_M = 5.0
GOAL_TOLERANCE_M = 15.0


def _date_recency_key(date_str):
    """date_str is format_date's output: "YYYY-MM" or "YYYY-MM-DD", zero-
    padded so plain string comparison already sorts chronologically.
    "unknown date" (format_date's fallback for a missing capture date)
    isn't comparable to those -- sorts as oldest/worst rather than
    crashing or landing in the middle by accident."""
    return "" if date_str == "unknown date" else date_str


def rank_dates(buckets: dict[int, list[dict]]) -> list[str]:
    """Every date present in ANY dot's bucket, ranked best-first by span
    (earliest to latest dot it has a pano in -- does coverage reach start
    to end), then total dot count, then recency (newer wins) as the final
    tiebreaker -- without it, ties fall back to insertion order, which
    happens to always favor Google over Apple since fetch_corridor_nodes
    fetches Google first for every dot, regardless of which source's
    coverage is actually better.

    Computed directly from the buckets (no edges needed) -- "which dots
    have a pano of this date" is exactly what a bucket already tells us.
    Returns ALL dates ranked, not just the top N -- callers building
    actual graphs stop once they have enough VALID ones (see
    build_graph.build_corridor_graphs), since a date can still fail the
    separate reachability check after this ranking.
    """
    covered_by_date: dict[str, set[int]] = {}
    for dot_index, bucket in buckets.items():
        for n in bucket:
            covered_by_date.setdefault(n["date"], set()).add(dot_index)

    scored = []
    for date, covered in covered_by_date.items():
        span = max(covered) - min(covered)
        scored.append((date, span, len(covered)))
    scored.sort(key=lambda t: (t[1], t[2], _date_recency_key(t[0])), reverse=True)
    return [date for date, _, _ in scored]


def date_connects(nodes, edges, start_lat, start_lon, goals):
    """Whether this date's own same-date edges can reach from a start-zone
    root to at least one goal-zone at all -- a structural check (graph
    reachability), not a real DA3 test. Dates that fail this can't connect
    anything no matter what, so there's no point downloading or GPU-testing
    them. Doesn't need to reach EVERY goal to be worth trying -- the
    algorithm itself handles a date covering only some of them.

    nodes/edges: ALREADY isolated to one date (see
    build_graph.build_corridor_graphs) -- no date field to check here,
    every node given IS that date."""
    by_key = {n["key"]: n for n in nodes}
    roots = [k for k, n in by_key.items()
             if haversine_m(n["lat"], n["lon"], start_lat, start_lon) <= START_ZONE_M]
    if not roots:
        return False
    seen = set(roots)
    stack = list(roots)
    while stack:
        key = stack.pop()
        n = by_key[key]
        if any(haversine_m(n["lat"], n["lon"], g[0], g[1]) <= GOAL_TOLERANCE_M for g in goals):
            return True
        for other_key, _ in edges.get(key, []):
            if other_key in by_key and other_key not in seen:
                seen.add(other_key)
                stack.append(other_key)
    return False
