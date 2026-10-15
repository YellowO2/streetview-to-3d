"""Turn map points near DA3's points into them, in colour."""
import numpy as np

from streetview_to_3d.postprocess.ramp import ramp

BLEND_M = 3.0             # map points this near DA3's turn into them


def toward(pts, cols, tree, da3_cols):
    """(cols, near) for map points approaching DA3's points: near is 1 on a DA3 point and 0 at
    BLEND_M or more, and colours mix toward the nearest DA3 point's by it. tree: a cKDTree of
    DA3's points, da3_cols their colours."""
    if tree is None or not len(pts):
        return cols, np.zeros(len(pts))
    d, k = tree.query(pts, distance_upper_bound=BLEND_M, workers=-1)
    near = 1 - ramp(d / BLEND_M)                               # d is inf past BLEND_M
    return cols + (da3_cols[np.minimum(k, len(da3_cols) - 1)] - cols) * near[:, None], near
