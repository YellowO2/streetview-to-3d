"""The GPS-fitting math shared across postprocess, and the one metre frame
every module measures in.

Positions are metres east/north of an ORIGIN that belongs to the AREA, not
to any one run: every solved transform is stored relative to it, so
deriving it from whichever pieces happen to be loaded would silently
invalidate every transform already saved. A reconstruction writes the area's
centre in its scene.json, and use_origin(*scene.origin) sets it before
anything is measured.
"""
import numpy as np

from services.geo import latlon_to_local_m

_origin = None


def use_origin(lat, lon):
    """Set the frame: the area centre every position is measured from."""
    global _origin
    _origin = (float(lat), float(lon))
    return _origin


def origin():
    if _origin is None:
        raise RuntimeError("no area origin set -- call use_origin(*scene.origin)")
    return _origin


def real_en(lat, lon):
    """(lat, lon) -> metres east/north of the area origin."""
    return latlon_to_local_m(lat, lon, *origin())


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


