"""Check that a capture date's dots reach from the start to a goal."""
from streetview_to_3d.common.DisjointSet import DisjointSet
from streetview_to_3d.common.geo import haversine_m

# A dot this close to the start is a start; one this close to a goal reaches it.
START_ZONE_M = 5.0
GOAL_TOLERANCE_M = 15.0


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
