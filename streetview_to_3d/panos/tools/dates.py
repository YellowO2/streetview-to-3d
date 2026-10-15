"""Rank capture dates, and check what one date covers (no GPU).

rank_dates orders dates by the corridor each newly covers (date_recency_key breaks ties),
cap_bucket_for_date keeps a dot's closest panos of one date, date_connects checks a date's
dots reach from the start to a goal.
"""
from streetview_to_3d.common.DisjointSet import DisjointSet
from streetview_to_3d.tools.geo import haversine_m

# Per dot, per date, how many of the closest panos to keep (and so download and rate):
# enough for a fallback when the closest is bad; more is the same spot again.
TOP_PANOS_PER_DOT = 3

# A dot this close to the start is a start; one this close to a goal reaches it.
START_ZONE_M = 5.0
GOAL_TOLERANCE_M = 15.0


def date_recency_key(date_str):
    """Sort key for fetch_pano_by_id's "YYYY-MM[-DD]" dates: "unknown date" sorts oldest."""
    return "" if date_str == "unknown date" else date_str


def rank_dates(buckets: dict[int, list[dict]]) -> list[str]:
    """Every date in any bucket, best first: most dots no earlier date covers, then most
    dots, then newest. New dots first, so an area's walked paths get a date even when
    its roads were driven many more times. Returns all dates; callers stop once enough pass.
    """
    covered_by_date: dict[str, set[int]] = {}
    for dot_index, bucket in buckets.items():
        for n in bucket:
            covered_by_date.setdefault(n["date"], set()).add(dot_index)

    ranked, covered = [], set()
    left = dict(covered_by_date)
    while left:
        date = max(left, key=lambda d: (len(left[d] - covered), len(left[d]), date_recency_key(d)))
        covered |= left.pop(date)
        ranked.append(date)
    return ranked


def cap_bucket_for_date(bucket, date, dot_lat, dot_lon, top_n):
    """This dot's own panos of ONE date, closest-first, capped to top_n."""
    same_date = [n for n in bucket if n["date"] == date]
    same_date.sort(key=lambda n: haversine_m(dot_lat, dot_lon, n["lat"], n["lon"]))
    return same_date[:top_n]


def _components(adjacency, n):
    """The connected groups of dots 0..n-1, walking adjacency."""
    sets = DisjointSet(n)
    for i, ns in adjacency.items():
        for j in ns:
            sets.union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(sets.find(i), set()).add(i)
    return list(groups.values())


def date_connects(dot_candidates, adjacency, points, start_lat, start_lon, goals):
    """Whether this date's dots reach from near the start toward at least one goal,
    moving only between adjacent dots that have candidates (as the walk does).

    Each separate graph of the corridor is checked from its own dots nearest the start,
    since the walk restarts in each. dot_candidates: {dot: [panos]}, non-empty dots only.
    """
    non_empty = set(dot_candidates.keys())
    if not non_empty:
        return False

    def to_start(i):
        return haversine_m(points[i][0], points[i][1], start_lat, start_lon)

    for group in _components(adjacency, len(points)):
        mine = group & non_empty
        if not mine:
            continue
        starts = [i for i in mine if to_start(i) <= START_ZONE_M]
        if not starts:
            # none in the start zone: start from the closest
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
