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


def _tangents(p, headings):
    """A unit direction at every dot, from the panorama's own heading.

    Heading is the source's absolute measurement of which way the camera
    faced, so unlike a difference between dot positions it does not inherit
    GPS noise, and it survives having only two dots to work with. Measured
    against the driven direction on real data it agrees to about 2.5 deg.

    A panorama says which way it faced, not which way the road is walked,
    so each heading is flipped to follow the walk. Where one is missing the
    neighbouring dots supply the direction instead.
    """
    step = np.gradient(p, axis=0)
    step /= np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-9)
    if headings is None:
        return step
    h = np.asarray(headings, float)
    t = np.column_stack([np.sin(h), np.cos(h)])          # bearing -> (east, north)
    t[np.isnan(h)] = step[np.isnan(h)]
    back = (t * step).sum(1) < 0
    t[back] *= -1
    return t


def smooth(polyline, headings=None, step=STEP_M):
    """A dot polyline resampled fine enough to be a frame.

    Graph dots are ~10 m apart, and RouteFrame finds the nearest SEGMENT
    midpoint -- at that spacing the along/left it returns is quantised to
    the segment.

    The curve passes through every dot with that dot's own heading as its
    direction (a cubic Hermite per segment). Taking direction from heading
    rather than from the dot spacing is what lets a two-dot road still
    describe a curve instead of a chord.
    """
    p = np.asarray(polyline, float)
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]
    p = p[keep]
    if headings is not None:
        headings = np.asarray(headings, float)[keep]
    if len(p) < 2:
        return p

    t = _tangents(p, headings)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    out = []
    for i, length in enumerate(seg):
        u = np.linspace(0, 1, max(int(length / step), 2), endpoint=False)[:, None]
        # Hermite basis, tangents scaled to the segment so the curve bulges
        # in proportion to how far it has to travel
        h00 = 2*u**3 - 3*u**2 + 1
        h10 = u**3 - 2*u**2 + u
        h01 = -2*u**3 + 3*u**2
        h11 = u**3 - u**2
        out.append(h00 * p[i] + h10 * length * t[i]
                   + h01 * p[i+1] + h11 * length * t[i+1])
    out.append(p[-1][None, :])
    return np.vstack(out)


def build(cams, graph, near_m=NEAR_M):
    """(curves, frames, on) for a set of pieces.

    curves  {road id: (N,2) smoothed polyline}
    frames  {road id: RouteFrame}
    on      {piece: [road ids]}, the roads the piece has a camera on. The
            line comes from Google's graph, not from the other pieces, so a
            piece alone on a road can still be seated against it.
    """
    xy = np.array(_metres(_dots(graph)))
    head = np.array([np.nan if n.pano.heading is None else n.pano.heading
                     for n in graph.nodes])
    curves = {}
    for rid, walk in enumerate(roads(graph)):
        c = smooth(xy[walk], head[walk])
        if len(c) >= 2:
            curves[rid] = c

    trees = {rid: cKDTree(c) for rid, c in curves.items()}
    on = {i: [rid for rid, t in trees.items()
              if (t.query(c)[0] <= near_m).any()] for i, c in cams.items()}

    used = {r for rids in on.values() for r in rids}
    curves = {r: c for r, c in curves.items() if r in used}
    return curves, {r: RouteFrame(c) for r, c in curves.items()}, on


def belongs(cams, curves, on):
    """{piece: the one road it was driven along}.

    `on` lists every road within reach of any of a piece's cameras, which
    near a junction is most of them. A piece was driven along exactly one,
    and only that road's line describes where it was -- measuring a piece
    against a street it was never on produces a confident, meaningless
    answer.
    """
    return {i: min(on[i], key=lambda r: cKDTree(curves[r]).query(c)[0].mean())
            for i, c in cams.items() if on.get(i)}


def clip(piece, curve, half_width=HALF_WIDTH_M):
    """The part of a piece's cloud lying along one road.

    Without this a piece at a junction rasterises as one blob covering
    both streets, and its medial axis -- the centreline the kerbs are
    split by -- belongs to neither of them.
    """
    xz, y, cols = piece
    keep = cKDTree(curve).query(xz)[0] <= half_width
    return (xz[keep], y[keep], cols[keep]), keep




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

    def normal(self, pts):
        """Unit vector pointing to the route's left, at each point."""
        _, k = self._tree.query(np.atleast_2d(np.asarray(pts, float)))
        t = self.tangent[k]
        return np.column_stack([-t[:, 1], t[:, 0]])
