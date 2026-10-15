"""Mix map colours toward the scene's at the edge of its ground."""
import numpy as np

from streetview_to_3d.postprocess.ramp import ramp


def tint(cols, near, dist, cut_m, band_m, strength):
    """cols mixed toward near by strength at cut_m, fading back to their own over band_m."""
    w = strength * (1 - ramp((dist - cut_m) / band_m))
    w = np.where(np.isnan(near).any(1), 0.0, w)[:, None]
    return cols * (1 - w) + np.nan_to_num(near) * w
