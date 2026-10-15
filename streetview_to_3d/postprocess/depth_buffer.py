"""Find the nearest point a camera sees in each pixel of a small view."""
import numpy as np

ZB_W = 512


def depth_buffer(cam, pts):
    """Nearest distance per pixel of a ZB_W-wide view, each point covering its 3x3 neighbours."""
    h, w = ZB_W // 2, ZB_W
    near = np.full(h * w, np.inf)
    if len(pts):
        u, v, r, _ = cam.look(pts)
        iu, iv = (u * w).astype(int), (v * h).astype(int)
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                np.minimum.at(near, np.clip(iv + dv, 0, h - 1) * w + (iu + du) % w, r)
    return near
