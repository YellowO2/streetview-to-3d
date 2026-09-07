"""The route the camera vehicle drove, through every piece in a run.

Several things downstream need one direction of travel agreed across the
whole run rather than per piece. "Left kerb" is the obvious one: each
piece's centreline is traced in whichever direction the tracing happened
to start, so on its own "left" means nothing, and matching one piece's
left kerb to another's is a coin flip until both are oriented the same
way.

The camera positions give that direction for free, and they give it
globally: the vehicle drove the route once, in one order. Ordering them
is the only work, and they are scattered points rather than a sequence --
so they are threaded onto a minimum spanning tree and the longest path
through it taken, which is the route from one end to the other.
"""
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

SMOOTH_PER_CAMERA = 2.0
STEP_M = 0.5


def route_curve(camera_xz, step=STEP_M):
    """A smooth curve through scattered camera positions, in travel order.

    The order is the longest path through the cameras' minimum spanning
    tree. A route doubles back on itself only rarely, so its two ends are
    the two points furthest apart along the tree, and everything else
    falls between them.
    """
    points = np.asarray(camera_xz, float)
    if len(points) < 2:
        return points
    tree = minimum_spanning_tree(squareform(pdist(points))).toarray()
    tree = tree + tree.T
    far = dijkstra(tree, indices=0)
    u = int(np.argmax(np.where(np.isfinite(far), far, -1)))
    dist, pred = dijkstra(tree, indices=u, return_predecessors=True)
    v = int(np.argmax(np.where(np.isfinite(dist), dist, -1)))

    order, cur = [], v
    while cur >= 0:
        order.append(cur)
        if cur == u:
            break
        cur = pred[cur]
    p = points[order[::-1]]
    if len(p) < 4:
        return p
    length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    tck, _ = splprep([p[:, 0], p[:, 1]], s=SMOOTH_PER_CAMERA * len(p),
                     k=min(3, len(p) - 1))
    x, y = splev(np.linspace(0, 1, max(int(length / step), 20)), tck)
    return np.column_stack([x, y])


class RouteFrame:
    """Distance along the route and offset to its left, for any point.

    This is the coordinate system the whole alignment works in. Curves are
    built in along-route order rather than by connectivity, which is what
    lets one curve span the gap between two pieces instead of stopping at
    the edge of each.
    """

    def __init__(self, route):
        route = np.asarray(route, float)
        d = np.diff(route, axis=0)
        self.tangent = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        self.mid = route[:-1] + d / 2
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(d, axis=1))][:-1]
        self.length = float(self.s[-1] + np.linalg.norm(d[-1]))
        self.route = route
        self._tree = cKDTree(self.mid)

    def project(self, pts):
        """(distance along the route, signed offset to its left)."""
        _, k = self._tree.query(np.asarray(pts, float))
        rel = pts - self.mid[k]
        t = self.tangent[k]
        along = self.s[k] + np.einsum("ij,ij->i", rel, t)
        left = t[:, 0] * rel[:, 1] - t[:, 1] * rel[:, 0]
        return along, left

    def orient(self, curve):
        """Turn a curve to run the way the vehicle drove.

        A piece whose centreline was traced backwards has its left and
        right kerbs swapped, and is then fitted to the wrong one.
        """
        curve = np.asarray(curve, float)
        d = curve[-1] - curve[0]
        if np.linalg.norm(d) < 1e-9:
            return curve
        _, k = self._tree.query(curve.mean(0))
        return curve[::-1] if float(d @ self.tangent[k]) < 0 else curve
