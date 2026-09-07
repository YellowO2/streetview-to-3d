"""Real ground height for a place, from Google's per-panorama elevation.

Nothing in a reconstruction says how high a piece sits: the GPS fit solves
east/north only, so a piece's height is DA3's own local height with no
datum at all. The vertical stage used to invent one by fitting a surface
through the pieces' own heights and moving each piece onto it -- which is
circular, and on a hill it squeezed 40 m of real relief out of the scene
(one piece pushed down 42 m, its neighbour up 25 m) while every individual
number still looked plausible.

Google returns an `elevation` per panorama. That is an outside measurement
of the same ground, so it can supply the datum the reconstruction lacks.

    SIGN. Google's elevation is metres above sea level, Y-UP. This repo is
    Y-DOWN (see the README). Elevations are negated on the way in, here, so
    that everything downstream stays in the one convention. Skipping this
    turns a hill into a hole.
"""
import json
import os

import numpy as np

from postprocess.gps_fit.fit import real_en

CACHE = "pano_elevation.json"
SURFACE_DEGREE = 3       # terrain is a smooth landform; the ROAD over it is not


def fetch(keys, latlons, cache_dir, log=print):
    """{key: elevation}, fetching only what the cache is missing.

    Keyed by panorama, but the exact panorama is often gone from Street
    View, so a lookup by position is accepted instead -- the ground at a
    spot does not depend on which day it was photographed.
    """
    from streetlevel import streetview
    path = os.path.join(cache_dir, CACHE)
    cache = json.load(open(path)) if os.path.exists(path) else {}
    todo = [(k, latlons[k]) for k in keys if k not in cache]
    if todo:
        log(f"fetching elevation for {len(todo)} panorama(s)")
    for n, (k, (lat, lon)) in enumerate(todo):
        p = None
        try:
            p = streetview.find_panorama_by_id(k.split(":", 1)[1])
        except Exception:
            pass
        if p is None:
            try:
                p = streetview.find_panorama(lat, lon)
            except Exception:
                pass
        cache[k] = None if p is None else p.elevation
        if (n + 1) % 25 == 0:
            json.dump(cache, open(path, "w"))
    json.dump(cache, open(path, "w"))
    return {k: v for k, v in cache.items() if v is not None}


def surface(elevations, latlons, degree=SURFACE_DEGREE):
    """A smooth ground surface, in THIS repo's Y-down metres.

    Returns f(xz) -> height, and the fit's own residual so a caller can see
    whether the landform is smooth enough to be described this way.
    """
    from postprocess.road_align.align_slope_of_pieces import _design, _robust

    keys = sorted(elevations)
    E = np.array([real_en(*latlons[k]) for k in keys])
    H = -np.array([elevations[k] for k in keys])      # Y-UP -> Y-DOWN
    ptp = lambda a: float(a.max() - a.min())
    ctr = E.mean(0)
    scale = max(ptp(E[:, 0]), ptp(E[:, 1]), 1.0)
    coef = _robust(_design(E, ctr, scale, degree=degree), H)

    def f(xz):
        return _design(np.asarray(xz, float), ctr, scale, degree=degree) @ coef

    return f, float(np.median(np.abs(H - f(E))))


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
