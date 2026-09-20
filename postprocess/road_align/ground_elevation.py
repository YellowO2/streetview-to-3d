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

    SIGN. Google's elevation is metres above sea level, Y-UP. This repo is
    Y-DOWN (see the README). Elevations are negated on the way in, here, so
    that everything downstream stays in the one convention. Skipping this
    turns a hill into a hole.
"""
import numpy as np

from postprocess.gps_fit.fit import real_en

SURFACE_DEGREE = 3       # terrain is a smooth landform; the ROAD over it is not


def surface(scene, degree=SURFACE_DEGREE):
    """A smooth ground surface over an area, in THIS repo's Y-down metres.

    Returns f(xz) -> height, the fit's own residual so a caller can see
    whether the landform is smooth enough to be described this way, and how
    many nodes it was built from.
    """
    from postprocess.road_align.align_slope_of_pieces import _design, _robust

    known = [(n.pano.lat, n.pano.lon, n.pano.elevation) for n in scene.nodes
             if n.pano.elevation is not None]
    if len(known) < 4:
        raise ValueError(
            f"only {len(known)} node(s) in this area have a ground height -- "
            "not enough to fit a surface through")
    E = np.array([real_en(lat, lon) for lat, lon, _ in known])
    H = -np.array([e for _, _, e in known])           # Y-UP -> Y-DOWN
    ptp = lambda a: float(a.max() - a.min())
    ctr = E.mean(0)
    scale = max(ptp(E[:, 0]), ptp(E[:, 1]), 1.0)
    coef = _robust(_design(E, ctr, scale, degree=degree), H)

    def f(xz):
        return _design(np.asarray(xz, float), ctr, scale, degree=degree) @ coef

    return f, float(np.median(np.abs(H - f(E)))), len(known)


def seat_on(road_by_piece, ground):
    """{piece: 4x4}, {piece: (height, tilt_deg)} onto a known ground surface.

    One height offset and one tilt per piece, exactly as before -- but
    measured against an outside surface rather than one built from the
    pieces themselves, so a piece that really is higher stays higher.
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
