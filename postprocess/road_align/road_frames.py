"""Which roads a piece lies along, and a frame for each of them.

One frame per road, not one for the whole run. A single frame threaded
through every camera in a run describes a corridor adequately and a campus
not at all: most pieces then get measured against a curve belonging to a
different street.

The graph already knows where the roads are, so the frame comes from the
road rather than from the cameras. `corridors.roads` gives the inventory
and each road gets its own frame, so a piece straddling a junction is not
a special case -- it simply appears on two roads.

Road polylines and camera positions are both in the scene's metre frame
(`gps_fit.fit.real_en`), so they are directly comparable.
"""

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

from postprocess.corridors import _dots, _metres, roads

# A camera this close to a road polyline is standing on that road. Google's
# dots sit on the driven line and a panorama is captured from it, so the
# distance is small; 12 m is wide enough for a dual carriageway's far side
# and narrow enough not to claim the street one block over.
NEAR_M = 12.0
# How far either side of a road its own surface is taken to extend. This is
# what keeps a piece describing ONE road: at 15 m a side road stayed inside
# the corridor, so the mask came out T-shaped, road_cells traced a boundary
# around the branch mouth, and that mouth was fitted as a kerb (piece_9's
# right "kerb" was an S-bend through a junction opening). Narrow enough to
# hold one carriageway and its verges, and to leave a branch outside.
HALF_WIDTH_M = 8.0
STEP_M = 0.5
SMOOTH_PER_DOT = 2.0
MIN_DOTS = 4


def smooth(polyline, step=STEP_M):
    """A dot polyline resampled fine enough to be a frame.

    Graph dots are ~10 m apart, and RouteFrame finds the nearest SEGMENT
    midpoint -- at that spacing the along/left it returns is quantised to
    the segment. Smoothing also steadies the tangent, which decides which
    side is left and is a derivative of noisy GPS.
    """
    p = np.asarray(polyline, float)
    # a road chained into a ring returns to its first dot, and splprep
    # rejects a repeated point
    p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]]
    if len(p) < MIN_DOTS:
        return p
    length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    tck, _ = splprep([p[:, 0], p[:, 1]], s=SMOOTH_PER_DOT * len(p),
                     k=min(3, len(p) - 1))
    x, y = splev(np.linspace(0, 1, max(int(length / step), 20)), tck)
    return np.column_stack([x, y])


def build(cams, graph, near_m=NEAR_M):
    """(curves, frames, on) for a set of pieces.

    curves  {road id: (N,2) smoothed polyline}
    frames  {road id: RouteFrame}
    on      {piece: [road ids]}, the roads the piece has a camera on. The
            line comes from Google's graph, not from the other pieces, so a
            piece alone on a road can still be seated against it.
    """
    xy = np.array(_metres(_dots(graph)))
    curves = {}
    for rid, walk in enumerate(roads(graph)):
        c = smooth(xy[walk])
        if len(c) >= MIN_DOTS:
            curves[rid] = c

    trees = {rid: cKDTree(c) for rid, c in curves.items()}
    on = {i: [rid for rid, t in trees.items()
              if (t.query(c)[0] <= near_m).any()] for i, c in cams.items()}

    used = {r for rids in on.values() for r in rids}
    curves = {r: c for r, c in curves.items() if r in used}
    return curves, {r: RouteFrame(c) for r, c in curves.items()}, on


def clip(piece, curve, half_width=HALF_WIDTH_M):
    """The part of a piece's cloud lying along one road.

    Without this a piece at a junction rasterises as one blob covering
    both streets, and its medial axis -- the centreline the kerbs are
    split by -- belongs to neither of them.
    """
    xz, y, cols = piece
    keep = cKDTree(curve).query(xz)[0] <= half_width
    return (xz[keep], y[keep], cols[keep]), keep


def near_cams(cams, curve, near_m=NEAR_M):
    """The piece's cameras that stand on this road.

    road_cells uses the cameras to pick which grey surface is the road,
    so handing it only the ones on THIS road is what makes it choose this
    road's surface rather than the whole junction.
    """
    d = cKDTree(curve).query(cams)[0]
    return cams[d <= near_m] if (d <= near_m).any() else cams


class RouteFrame:
    """Distance along a road and offset to its left, for any point.

    Everything downstream needs one agreed direction of travel: "left" is
    meaningless until both a road and a piece are oriented the same way.
    A road's own polyline supplies it.
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
