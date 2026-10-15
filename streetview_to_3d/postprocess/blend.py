"""Weigh a camera by how much further it is than the nearest one."""
import numpy as np

BLEND_M = 2.0             # a camera this much further than the nearest counts 1/e as much


def blend(dist, nearest):
    """Weight of a camera dist away where the nearest is `nearest` away: 1 for the nearest,
    1/e one BLEND_M further, 0 for inf. Used for both ground height and colour."""
    near = np.where(np.isfinite(nearest), nearest, 0.0)
    return np.where(np.isfinite(dist), np.exp(-(np.where(np.isfinite(dist), dist, 0.0) - near) / BLEND_M), 0.0)
