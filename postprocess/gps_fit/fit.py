"""Single shared home for the GPS-fitting math every gps_*/piece_*/
export_*_test script in this directory needs -- fit_similarity_2d was
copy-pasted into five separate files before this existed (a real source
of bugs: e.g. two scripts picking a DIFFERENT local-meters origin for
the same node produces "real_en" values that aren't actually comparable,
even though both look like valid coordinates). Import from here instead
of redefining it.

GLOBAL_ORIGIN is fixed (NTU's own dot 0) specifically so that EVERY
script's real_en values live in the same shared local-meters frame,
comparable across scripts/sessions/runs -- never compute your own
per-piece or per-group origin for anything meant to be plotted or
compared against other data.
"""
from paths import FETCHED_GRAPH, NTU_DIR
import json
import os

import numpy as np

from services.geo import latlon_to_local_m


with open(FETCHED_GRAPH) as _f:
    _points = json.load(_f)["points"]
GLOBAL_ORIGIN_LAT, GLOBAL_ORIGIN_LON = _points[0]


def real_en(lat, lon):
    """(lat, lon) -> local ENU meters relative to the ONE shared
    GLOBAL_ORIGIN -- use this everywhere instead of calling
    latlon_to_local_m directly with a locally-chosen origin, so every
    script's output lives in the same comparable frame."""
    return latlon_to_local_m(lat, lon, GLOBAL_ORIGIN_LAT, GLOBAL_ORIGIN_LON)


def fit_similarity_2d(src, dst):
    """Least-squares similarity transform (rotation + uniform scale +
    translation) mapping src -> dst, via SVD (Umeyama/Kabsch with
    reflection correction). Returns (R, scale, t) such that
    scale * (src @ R.T) + t ~= dst."""
    n = len(src)
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    H = (src_c.T @ dst_c) / n
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T
    var_src = (src_c ** 2).sum(axis=1).mean()
    scale = float(np.trace(np.diag(S) @ D) / var_src)
    t = dst_mean - scale * (R @ src_mean)
    return R, scale, t


def fit_nodes(nodes, da3_field="da3_xz", lat_field="lat", lon_field="lon"):
    """Fits one group's own nodes to GPS, all in the SAME shared
    GLOBAL_ORIGIN frame. nodes: list of dicts, each with da3_field (a
    [x, z] pair) and lat_field/lon_field. Returns (R, scale, t,
    real_en_arr, fitted_en_arr, residuals_arr) -- residuals are
    origin-invariant (real-world distances, not affected by which
    origin was chosen), but real_en/fitted_en are only meaningful
    compared against ANOTHER group's if both used this same function."""
    src = np.array([n[da3_field] for n in nodes])
    dst = np.array([real_en(n[lat_field], n[lon_field]) for n in nodes])
    R, scale, t = fit_similarity_2d(src, dst)
    fitted = scale * (src @ R.T) + t
    residuals = np.linalg.norm(fitted - dst, axis=1)
    return R, scale, t, dst, fitted, residuals
