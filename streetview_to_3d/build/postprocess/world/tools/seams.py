"""How the map meets the scene: blends, map point sizes and colour fixes.

tint and meet blend colour and height at the scene ground's edge, beneath keeps map ground
under the scene's own, toward turns map points near DA3's into them, spacing_at sizes map
points by their distance from DA3's and gap_at by their distance from a camera, unhazed
corrects satellite colours, ramp eases between.
"""
import numpy as np

from streetview_to_3d.build.postprocess.SceneGround import CELL_M

# The one rule for a map point's size: DA3's points are the smallest in the world; a map point is
# as big on them, and bigger the further from them. Its spacing always follows its size.
DA3_M = 0.17              # DA3's points as drawn (the viewer's default look: 0.1 m x 1.4 x 1.2)
GROW = 0.0154             # a map point's size grows this much per metre from DA3's points
RATIO = 2.2               # a point's size, of its spacing: what leaves no gaps (the viewer's blocks.RATIO)
# A building's point goes only where DA3 has no point right there:
COVER = 0.75              # a map point with a DA3 point within COVER x its spacing is DA3's
# Where the map meets the scene's own points: its ground runs on a little way under the scene's,
# hiding that ground's ragged rim, and goes where the scene's covers (the land and Google's tiles
# alike).
UNDER_M = 0.1             # map ground kept this far under the scene's own
OVERLAP_M = 1.0           # ... for this far in from its edge; deeper in it is dropped
BLEND_M = 3.0             # map points this near DA3's turn into them (toward)
GAP0_M, GAP_PER = 0.05, 0.018   # gap_at: spacing at a camera, growth per metre
# Satellite colours made to look like the panos' (unhazed):
GREY_C, GREY_N = 8.0, 200   # satellite tint learnt from ground under this Lab chroma, at least this many squares
LIGHT_RANGE = (0.7, 1.6)    # lightness contrast scale limits
BOOST_MAX, VIVID_C = 2.5, 60.0  # max chroma boost; no boost at this Lab chroma


def ramp(t):
    """0 below 0, 1 above 1, smooth between."""
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def tint(cols, near, dist, cut_m, band_m, strength):
    """cols mixed toward near by strength at cut_m, fading back to their own over band_m."""
    w = strength * (1 - ramp((dist - cut_m) / band_m))
    w = np.where(np.isnan(near).any(1), 0.0, w)[:, None]
    return cols * (1 - w) + np.nan_to_num(near) * w


def meet(h, ground, dist, band_m, max_m=np.inf):
    """Heights h pulled onto the scene's ground: fully at its edge, not at all band_m out,
    and not where they differ by more than max_m."""
    w = 1 - ramp(dist / band_m)
    w = np.where(np.isnan(ground) | (np.abs(np.nan_to_num(ground) - h) > max_m), 0.0, w)
    return h + (np.nan_to_num(ground) - h) * w


def size_at(d):
    """A map point's size (m) d metres from DA3's points."""
    return DA3_M + GROW * np.maximum(np.asarray(d, float), 0)


def spacing_at(d):
    """A map point's spacing (m) d metres from DA3's points: its size's."""
    return size_at(d) / RATIO


def gap_at(cam_d):
    """Map point spacing cam_d metres from the nearest camera: GAP0_M plus GAP_PER per metre."""
    return GAP0_M + GAP_PER * np.asarray(cam_d)


def toward(pts, cols, tree, da3_cols):
    """(cols, near) for map points approaching DA3's points: near is 1 on a DA3 point and 0 at
    BLEND_M or more, and colours mix toward the nearest DA3 point's by it. tree: a cKDTree of
    DA3's points, da3_cols their colours."""
    if tree is None or not len(pts):
        return cols, np.zeros(len(pts))
    d, k = tree.query(pts, distance_upper_bound=BLEND_M, workers=-1)
    near = 1 - ramp(d / BLEND_M)                               # d is inf past BLEND_M
    return cols + (da3_cols[np.minimum(k, len(da3_cols) - 1)] - cols) * near[:, None], near


def beneath(h, ground, inside, overlap=OVERLAP_M):
    """(h, keep) for map ground at heights h where the scene may have ground of its own (ground:
    its height there; inside: metres in from its edge, 0 where it has none): kept UNDER_M
    beneath it within overlap of the edge, dropped deeper in."""
    return np.where(inside > 0, np.minimum(h, ground - UNDER_M), h), inside <= overlap


def lab(rgb):
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
    P, S = lab(pano[ok]), lab(sat[ok])
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
        L = lab(cols[good])
        ab = L[:, 1:] + tint
        c = np.hypot(ab[:, 0], ab[:, 1])
        ab *= (boost / (1 + (boost - 1) * np.minimum(c / VIVID_C, 1)))[:, None]
        out[good] = _rgb(np.c_[np.clip(np.maximum(k * L[:, 0] + o, L[:, 0]), 0, 100), ab])
        return out
    return fix
