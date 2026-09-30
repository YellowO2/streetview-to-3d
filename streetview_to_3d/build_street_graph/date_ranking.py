"""Given the gathered per-dot candidate buckets, decide which dates are
worth building a real graph for -- coverage ranking + structural
reachability, no GPU, no DA3. This is a "build the candidate pool"
concern, not "solve the graph" (see
reconstruct/walk_graph.py for the actual algorithm).
"""
from streetview_to_3d.services.geo import haversine_m

# Dates kept, ranked by coverage span. Single source of truth for how many
# isolated per-date graphs build_corridor_graphs ever builds. 5 now that a
# date only costs its sampled panos up front (walk_graph._sample_dates)
# and the walk skips dates that sample badly or aren't needed for
# patching -- a date that covers little can still patch a weak stretch.
DATE_TOP_N = 5

# Mirrors the pathfind algorithm's own start_zone_m/point_cover_tolerance_m
# defaults -- used here only to pre-check whether a date's own dots can
# even reach from a start-zone dot to a goal-zone dot, before spending a
# download (let alone a GPU test) on it.
START_ZONE_M = 5.0
GOAL_TOLERANCE_M = 15.0

def date_recency_key(date_str):
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
    tiebreaker -- without it, ties fall back to insertion order.

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
    scored.sort(key=lambda t: (t[1], t[2], date_recency_key(t[0])), reverse=True)
    return [date for date, _, _ in scored]


def _components(adjacency, dots):
    """The connected groups of `dots`, walking adjacency."""
    left, groups = set(dots), []
    while left:
        stack = [left.pop()]
        group = set(stack)
        while stack:
            for j in adjacency.get(stack.pop(), []):
                if j in left:
                    left.discard(j)
                    group.add(j)
                    stack.append(j)
        groups.append(group)
    return groups


def date_connects(dot_candidates, adjacency, points, start_lat, start_lon, goals):
    """Whether this date's own dots can structurally reach from near the
    start toward at least one goal, walking dot-to-dot through ONLY
    non-empty, directly-adjacent dots -- no flood past an empty dot,
    mirroring the algorithm's own movement rule exactly (see
    walk_graph.py's visit()): a dot is a real selection-graph node, so an
    empty structural neighbor is a genuine dead end for that date, not
    skipped past. Not a real DA3 test, just "could this date's coverage
    even physically connect," so a date that fails here truly can't work
    no matter what gets tested. Doesn't need to reach EVERY goal to be
    worth trying -- the algorithm itself handles a date covering only
    some of them.

    The corridor can be several graphs (see candidates.expand_area), and
    the walk restarts in each (walk_graph's pick_seed). So each graph is
    checked on its own, from its own dots nearest the start: a date that
    only covers a park's paths away from the start still counts.

    dot_candidates: {dot_index: [panos]} for non-empty dots of this date
    ONLY (see build_graph.build_corridor_graphs). adjacency: the
    structural dot-to-dot graph (see fetch_nodes.corridor_points).
    points: every dot's real (lat, lon), for the start-zone check.
    """
    non_empty = set(dot_candidates.keys())
    if not non_empty:
        return False

    def to_start(i):
        return haversine_m(points[i][0], points[i][1], start_lat, start_lon)

    for group in _components(adjacency, range(len(points))):
        mine = group & non_empty
        if not mine:
            continue
        starts = [i for i in mine if to_start(i) <= START_ZONE_M]
        if not starts:
            # Nothing of this date sits exactly in the start zone -- fall
            # back to this graph's non-empty dot closest to the start,
            # mirroring how the algorithm itself has to bootstrap from
            # SOMEWHERE nearby.
            starts = [min(mine, key=to_start)]

        seen = set(starts)
        stack = list(starts)
        while stack:
            i = stack.pop()
            lat_i, lon_i = points[i]
            if any(haversine_m(lat_i, lon_i, g[0], g[1]) <= GOAL_TOLERANCE_M for g in goals):
                return True
            for j in adjacency.get(i, []):
                if j in seen or j not in non_empty:
                    continue
                seen.add(j)
                stack.append(j)
    return False
