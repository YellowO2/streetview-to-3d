"""Pull map heights onto the scene's ground at its edge."""
import numpy as np

from streetview_to_3d.postprocess.ramp import ramp


def meet(h, ground, dist, band_m, max_m=np.inf):
    """Heights h pulled onto the scene's ground: fully at its edge, not at all band_m out,
    and not where they differ by more than max_m."""
    w = 1 - ramp(dist / band_m)
    w = np.where(np.isnan(ground) | (np.abs(np.nan_to_num(ground) - h) > max_m), 0.0, w)
    return h + (np.nan_to_num(ground) - h) * w
