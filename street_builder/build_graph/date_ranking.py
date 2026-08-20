"""Given the gathered per-dot candidate buckets, decide which dates are
worth building a real graph for -- coverage ranking + structural
reachability, no GPU, no DA3. This is a "build the candidate pool"
concern, not "solve the graph" (see
street_builder/reconstruction/walk_graph.py for the actual algorithm).
"""
from services.geo import haversine_m

# Dates kept, ranked by coverage span. Single source of truth for how many
# isolated per-date graphs build_corridor_graphs ever builds. Capped at 3
# (not 5) to save compute -- each extra date graph costs a full download
# batch plus its own share of the pathfind search's GPU time budget, and
# the top-ranked dates already capture the corridor's best coverage.
DATE_TOP_N = 3

# Mirrors the pathfind algorithm's own start_zone_m/point_cover_tolerance_m
# defaults -- used here only to pre-check whether a date's own dots can
# even reach from a start-zone dot to a goal-zone dot, before spending a
# download (let alone a GPU test) on it.
START_ZONE_M = 5.0
GOAL_TOLERANCE_M = 15.0

# A candidate real edge between two dots' own panos is only plausible if
# they're within this real distance -- used here (and by the algorithm's
# own skip-one-dot fallback, see walk_graph.py) to decide whether hopping
# PAST an empty dot to the one after it is even structurally reasonable.
EDGE_MAX_DIST_M = 18.0


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


def date_connects(dot_candidates, adjacency, points, start_lat, start_lon, goals):
    """Whether this date's own dots can structurally reach from near the
    start toward at least one goal, walking dot-to-dot -- a direct move to
    an adjacent dot that has a candidate, or a flood past however many
    empty dots in a row (structurally, through adjacency), as long as
    that stays within real EDGE_MAX_DIST_M of where the flood started.
    Any gap width, not a fixed hop count -- the real constraint was
    always "close enough to plausibly connect," never "exactly one empty
    dot in between." Mirrors the algorithm's own movement rule exactly
    (see walk_graph.py's visit()) -- not a real DA3 test, just "could
    this date's coverage even physically connect," so a date that fails
    here truly can't work no matter what gets tested. Doesn't need to
    reach EVERY goal to be worth trying -- the algorithm itself handles a
    date covering only some of them.

    dot_candidates: {dot_index: [panos]} for non-empty dots of this date
    ONLY (see build_graph.build_corridor_graphs). adjacency: the
    structural dot-to-dot graph (see fetch_nodes.interpolate_points).
    points: every dot's real (lat, lon), for the flood distance check.
    """
    non_empty = set(dot_candidates.keys())
    if not non_empty:
        return False

    starts = [i for i in non_empty
              if haversine_m(points[i][0], points[i][1], start_lat, start_lon) <= START_ZONE_M]
    if not starts:
        # Nothing of this date sits exactly in the start zone -- fall back
        # to whichever non-empty dot is closest, mirroring how the
        # algorithm itself has to bootstrap from SOMEWHERE nearby.
        starts = [min(non_empty, key=lambda i: haversine_m(points[i][0], points[i][1], start_lat, start_lon))]

    seen = set(starts)
    stack = list(starts)
    while stack:
        i = stack.pop()
        lat_i, lon_i = points[i]
        if any(haversine_m(lat_i, lon_i, g[0], g[1]) <= GOAL_TOLERANCE_M for g in goals):
            return True
        for j in adjacency.get(i, []):
            if j in seen:
                continue
            if j in non_empty:
                seen.add(j)
                stack.append(j)
                continue
            # j is empty -- flood through it (and further empty dots)
            # within real EDGE_MAX_DIST_M of i, same rule as the
            # algorithm's own flood fallback.
            sub_seen = {i, j}
            sub_frontier = [j]
            while sub_frontier:
                d = sub_frontier.pop()
                if haversine_m(lat_i, lon_i, points[d][0], points[d][1]) > EDGE_MAX_DIST_M:
                    continue
                if d in non_empty:
                    if d not in seen:
                        seen.add(d)
                        stack.append(d)
                    continue
                for k in adjacency.get(d, []):
                    if k not in sub_seen:
                        sub_seen.add(k)
                        sub_frontier.append(k)
    return False
