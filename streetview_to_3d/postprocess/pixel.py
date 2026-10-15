"""Find which of depth_buffer's pixels a view coordinate falls in."""
import numpy as np

from streetview_to_3d.postprocess.depth_buffer import ZB_W


def pixel(u, v):
    """Index into depth_buffer's pixels of view coordinates (u, v)."""
    h, w = ZB_W // 2, ZB_W
    return np.clip((v * h).astype(int), 0, h - 1) * w + (u * w).astype(int) % w
