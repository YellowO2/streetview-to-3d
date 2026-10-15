"""Correct satellite colours to look like the panos' own."""
import numpy as np

from streetview_to_3d.postprocess.SceneGround import CELL_M

GREY_C, GREY_N = 8.0, 200   # satellite tint learnt from ground under this Lab chroma, at least this many squares
LIGHT_RANGE = (0.7, 1.6)    # lightness contrast scale limits
BOOST_MAX, VIVID_C = 2.5, 60.0  # max chroma boost; no boost at this Lab chroma


def _lab(rgb):
    """sRGB (n, 3) 0-1 -> CIE Lab (D65)."""
    c = np.clip(rgb, 0, 1)
    xyz = np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4) @ _XYZ.T / _WHITE
    f = np.where(xyz > .008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.c_[116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])]


def _rgb(lab_):
    """CIE Lab (n, 3) -> sRGB 0-1, clipped."""
    fy = (lab_[:, 0] + 16) / 116
    f = np.c_[fy + lab_[:, 1] / 500, fy, fy - lab_[:, 2] / 200]
    lin = np.where(f > .2069, f ** 3, (f - 16 / 116) / 7.787) * _WHITE @ np.linalg.inv(_XYZ).T
    return np.clip(np.where(lin <= .0031308, lin * 12.92, 1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - .055), 0, 1)


_XYZ = np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]])
_WHITE = np.array([.9505, 1, 1.089])


def unhazed(ground, satellite):
    """f(colours) -> colours: satellite colours corrected to look like the panos'.

    Learnt in Lab where both see the scene's ground (ground: SceneGround;
    satellite(east/north) -> (n, 3), NaN where missing): the colour cast from
    grey ground, lightness matched but never darkened, and chroma boosted up
    to BOOST_MAX (less for vivid colours). Hue is not changed. Identity if
    there is too little to learn from."""
    have = np.argwhere(np.isfinite(ground.height)) if ground.height.size else np.zeros((0, 2), int)
    if len(have) < GREY_N:
        return lambda cols: cols
    pano = ground.colour[tuple(have.T)]
    sat = satellite((have + ground.lo + .5) * CELL_M)
    ok = np.isfinite(sat).all(1) & np.isfinite(pano).all(1)
    if ok.sum() < GREY_N:
        return lambda cols: cols
    P, S = _lab(pano[ok]), _lab(sat[ok])
    grey = np.hypot(P[:, 1], P[:, 2]) < GREY_C
    tint = (P[grey, 1:] - S[grey, 1:]).mean(0) if grey.sum() >= GREY_N else np.zeros(2)
    S[:, 1:] += tint
    k = np.clip(P[:, 0].std() / max(S[:, 0].std(), 1e-6), *LIGHT_RANGE)
    o = P[:, 0].mean() - k * S[:, 0].mean()
    boost = np.clip(np.hypot(P[:, 1], P[:, 2]).mean() / max(np.hypot(S[:, 1], S[:, 2]).mean(), 1e-6), 1, BOOST_MAX)

    def fix(cols):
        cols = np.asarray(cols, float)
        out = cols.copy()
        good = np.isfinite(cols).all(1)
        L = _lab(cols[good])
        ab = L[:, 1:] + tint
        c = np.hypot(ab[:, 0], ab[:, 1])
        ab *= (boost / (1 + (boost - 1) * np.minimum(c / VIVID_C, 1)))[:, None]
        out[good] = _rgb(np.c_[np.clip(np.maximum(k * L[:, 0] + o, L[:, 0]), 0, 100), ab])
        return out
    return fix
