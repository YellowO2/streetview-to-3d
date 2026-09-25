"""Real ground height for a place, from the street graph's own dots.

Nothing in a reconstruction says how high a piece sits: the GPS fit solves
east/north only, so a piece's height is DA3's own local height with no
datum at all. Fitting a surface through the pieces' own heights instead is
circular, and on a hill it squeezes the relief out of the scene while every
individual number still looks plausible.

Every pano lookup returns an `elevation`, so each node already carries the
ground height at its own spot from when the area was built -- an outside
measurement of the same ground, and it is there whether or not DA3 managed
to reconstruct anything at that node.

The samples all sit ALONG roads, because that is where panoramas are, so
that is the only direction they say anything about. Height is therefore
fitted per road as a function of distance along it, and taken as level
across it -- which is not a numerical dodge but how roads are built. A
2-D surface over the same points has nothing holding it down sideways:
fitted here it passed through all eight exactly and still swung 25 m off
to the side, and pieces were then sheared by 27 degrees to chase it.

    SIGN. Google's elevation is metres above sea level, Y-UP. This repo is
    Y-DOWN (see the README). Elevations are negated on the way in, here, so
    that everything downstream stays in the one convention. Skipping this
    turns a hill into a hole.
"""
import numpy as np
from scipy.spatial import cKDTree

from streetview_to_3d.postprocess.gps_fit.fit import real_en

MAX_DEGREE = 3           # a road's own rise and fall, never more than this
PER_COEFFICIENT = 4      # samples each degree has to earn


class Ground:
    """Height at any point, from one height profile per road.

    A point is answered by whichever road's centre line is nearest: at a
    junction the shared dot belongs to both roads, so both profiles pass
    through its elevation and they agree exactly where they meet.

    The one case this cannot express is a road crossing another at a
    different height -- an overpass occupies the same place as what it
    spans, and a nearest-line lookup has no way to tell them apart.
    """

    def __init__(self, profiles, frames):
        self._profiles = profiles
        self._frames = [frames[r] for r in profiles]
        self._roads = list(profiles)
        self._tree = cKDTree(np.vstack([frames[r].mid for r in profiles]))
        self._owner = np.concatenate([[i] * len(frames[r].mid)
                                      for i, r in enumerate(profiles)])

    def __call__(self, xz):
        xz = np.asarray(xz, float)
        out = np.empty(len(xz))
        near = self._owner[self._tree.query(xz)[1]]
        for i, road in enumerate(self._roads):
            m = near == i
            if m.any():
                along, _ = self._frames[i].project(xz[m])
                out[m] = self._profiles[road](along)
        return out


def _profile(along, height):
    """Height as a function of distance along one road.

    The degree is what the samples can hold still: a straight line needs
    two, and every further bend has to earn PER_COEFFICIENT more. With one
    sample the road is simply held level at that height.
    """
    if len(along) == 1:
        h = float(height[0])
        return lambda s: np.full(len(np.atleast_1d(s)), h)
    degree = min(MAX_DEGREE, max(1, len(along) // PER_COEFFICIENT))
    coef = np.polyfit(along, height, min(degree, len(along) - 1))
    return lambda s: np.polyval(coef, s)


def surface(scene, curves, frames, on):
    """(Ground, median residual, samples used) for a scene's roads.

    curves/frames/on come from road_frames.build -- the same road lines the
    horizontal stage seated pieces onto, so both stages measure against one
    description of the street.
    """
    known = [(np.array(real_en(n.pano.lat, n.pano.lon)), -n.pano.elevation)
             for n in scene.nodes if n.pano.elevation is not None]
    if not known:
        raise ValueError("no node in this area has a ground height")
    E = np.array([e for e, _ in known])
    H = np.array([h for _, h in known])               # already Y-DOWN

    profiles, used, residuals = {}, set(), []
    for road, frame in frames.items():
        along, left = frame.project(E)
        mine = np.abs(left) <= NEAR_ROAD_M
        if not mine.any():
            continue
        f = _profile(along[mine], H[mine])
        profiles[road] = f
        used |= set(np.flatnonzero(mine).tolist())
        residuals.append(np.abs(H[mine] - f(along[mine])))

    if not profiles:
        raise ValueError("no road in this area has a node with a ground height")
    resid = float(np.median(np.concatenate(residuals)))
    return Ground(profiles, frames), resid, len(used)


NEAR_ROAD_M = 15.0       # a node further out than this is not on that road


def seat_on(road_by_piece, ground):
    """{piece: 4x4}, {piece: (height, tilt_deg)} onto a known ground.

    One height offset and one tilt per piece: the piece's own shape is left
    alone, because DA3's local geometry is far better than a handful of
    sparse elevation samples.
    """
    out, report = {}, {}
    for i, p in road_by_piece.items():
        if len(p) < 3:
            out[i] = np.eye(4)
            continue
        gap = p[:, 1] - ground(p[:, [0, 2]])
        cx, cz = p[:, 0].mean(), p[:, 2].mean()
        A = np.column_stack([np.ones(len(p)), p[:, 0] - cx, p[:, 2] - cz])
        a, b, c = np.linalg.lstsq(A, gap, rcond=None)[0]
        T = np.eye(4)
        T[1, 0], T[1, 2] = -b, -c
        T[1, 3] = -a + b * cx + c * cz
        out[i] = T
        report[i] = (float(-a), float(np.degrees(np.arctan(np.hypot(b, c)))))
    return out, report
