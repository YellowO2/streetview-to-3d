"""Shared geo helpers -- used by app.py's single-pano flow and street_builder/."""
import math
import re


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def order_points_by_chain(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Reorder (lat, lon) points by real spatial adjacency instead of
    whatever order they were given in -- so a route traced out of click
    order still gets its correct two endpoints and a straight, non-backtracking
    shape between them.

    Builds a minimum spanning tree over the points (Prim's, O(n^2), fine for
    a handful of clicks), then returns its diameter path -- the longest
    path between two leaves. For a simple, non-branching line of points
    like a traced street, that diameter path is exactly the real route
    order; any leftover points not on it (only possible if the clicks
    branch, which a street selection shouldn't) are dropped.
    """
    n = len(points)
    if n <= 2:
        return list(points)

    def dist(i, j):
        return haversine_m(points[i][0], points[i][1], points[j][0], points[j][1])

    in_tree = [False] * n
    in_tree[0] = True
    adj = {i: [] for i in range(n)}
    dist_to_tree = [dist(0, i) for i in range(n)]
    nearest_in_tree = [0] * n
    for _ in range(n - 1):
        best = min((i for i in range(n) if not in_tree[i]), key=lambda i: dist_to_tree[i])
        in_tree[best] = True
        adj[best].append(nearest_in_tree[best])
        adj[nearest_in_tree[best]].append(best)
        for i in range(n):
            if not in_tree[i]:
                d = dist(best, i)
                if d < dist_to_tree[i]:
                    dist_to_tree[i] = d
                    nearest_in_tree[i] = best

    def farthest_from(start):
        dist_from = {start: 0.0}
        parent = {start: None}
        stack = [start]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in dist_from:
                    dist_from[v] = dist_from[u] + dist(u, v)
                    parent[v] = u
                    stack.append(v)
        far = max(dist_from, key=dist_from.get)
        return far, parent

    a, _ = farthest_from(0)
    b, parent = farthest_from(a)

    path = []
    node = b
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()

    return [points[i] for i in path]


def extract_lat_lon(raw: str):
    """Parse a Google Maps URL (.../@lat,lon,...) or a plain "lat,lon" string."""
    raw = raw.strip()
    m = re.search(r"@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)", raw)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)$", raw)
    if m:
        return float(m.group(1)), float(m.group(2))
    raise ValueError("Use a Google Maps URL with /@lat,lon or paste lat,lon directly.")
