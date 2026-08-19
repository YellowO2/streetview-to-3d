"""Given the gathered candidate pool, decide which dates are even worth
handing to the pathfind algorithm -- structural reachability + coverage
ranking, no GPU, no DA3. This is a "build the candidate pool" concern,
not "solve the graph" (see street_builder/reconstruction/walk_graph.py
for the actual algorithm).
"""
from services.geo import haversine_m
from street_builder.build_graph.fetch_nodes import POINT_MAX_DIST_M

# Dates kept, ranked by coverage span. Single source of truth: also passed
# as top_n_dates to the GPU call, so download count == dates actually used.
DATE_TOP_N = 5

# Mirrors the pathfind algorithm's own start_zone_m/goal_tolerance_m
# defaults -- used here only to pre-check whether a date's own same-date
# edges can even reach from a start-zone root to a goal-zone node, before
# spending a download (let alone a GPU test) on it.
START_ZONE_M = 5.0
GOAL_TOLERANCE_M = 15.0


def _covered_indices(candidates, points, max_dist_m):
    """Indices of sample points with >=1 candidate within max_dist_m."""
    covered = set()
    for i, (lat, lon) in enumerate(points):
        if any(haversine_m(lat, lon, c["lat"], c["lon"]) <= max_dist_m for c in candidates):
            covered.add(i)
    return covered


def _date_connects(date_nodes, edges, start_lat, start_lon, goals):
    """Whether this date's own same-date edges can reach from a start-zone
    root to at least one goal-zone at all -- a structural check (graph
    reachability), not a real DA3 test. Dates that fail this can't connect
    anything no matter what, so there's no point downloading or GPU-testing
    them. Doesn't need to reach EVERY goal to be worth trying -- the
    multi-goal search itself handles a date covering only some of them."""
    by_key = {n["key"]: n for n in date_nodes}
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


def _date_recency_key(date_str):
    """date_str is format_date's output: "YYYY-MM" or "YYYY-MM-DD", zero-
    padded so plain string comparison already sorts chronologically.
    "unknown date" (format_date's fallback for a missing capture date)
    isn't comparable to those -- sorts as oldest/worst rather than
    crashing or landing in the middle by accident."""
    return "" if date_str == "unknown date" else date_str


def _rank_dates(by_date, points, max_dist_m, top_n):
    """Dates ranked by span (earliest to latest covered point -- does
    coverage reach start to end), then total coverage count, then recency
    (newer wins) as the final tiebreaker -- without it, ties fall back to
    insertion order, which happens to always favor Google over Apple since
    fetch_corridor_nodes fetches Google first for every corridor point,
    regardless of which source's coverage is actually better."""
    scored = []
    for date, candidates in by_date.items():
        covered = _covered_indices(candidates, points, max_dist_m)
        span = (max(covered) - min(covered)) if covered else 0
        scored.append((date, span, len(covered)))
    scored.sort(key=lambda t: (t[1], t[2], _date_recency_key(t[0])), reverse=True)
    return [date for date, _, _ in scored[:top_n]]


def local_batch(nodes, edges, points, start_lat, start_lon, goals):
    """Keep every already-gathered candidate whose date both (a) structurally
    connects start-zone to at least one goal-zone via its own same-date
    edges, and (b) ranks in the top-N by coverage span among the dates that
    pass (a). This is the whole download batch (single GPU call downstream,
    see street_builder/main.py for why there's no second-pass fallback).

    Returns (keys, top_dates) -- keys as a list (not a set) specifically so
    the caller can preserve top_dates' rank order downstream. A set's
    iteration order depends on Python's per-process string hash seed, which
    was silently discarding this ranking (and making the whole pathfind
    result non-reproducible run to run) even though top_dates itself is a
    real, deterministic ranking."""
    by_date = {}
    for n in nodes:
        by_date.setdefault(n["date"], []).append(n)

    connectable = {date: ns for date, ns in by_date.items()
                   if _date_connects(ns, edges, start_lat, start_lon, goals)}
    dropped = len(by_date) - len(connectable)
    if dropped:
        print(f"Dropped {dropped}/{len(by_date)} dates: no same-date edge path from start toward any goal.")

    top_dates = _rank_dates(connectable, points, POINT_MAX_DIST_M, DATE_TOP_N)
    print(f"Top dates by coverage span: {top_dates}")

    keys = [n["key"] for n in nodes if n["date"] in top_dates]
    return keys, top_dates
