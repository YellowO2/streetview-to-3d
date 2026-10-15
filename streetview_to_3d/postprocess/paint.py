"""Colour points from the panos that see them cleanly.

Colour for the fill's ground from the panos, mixing every camera that sees a point cleanly:
nothing of DA3's in front (depth_buffer), not masked or blurred, under NADIR_DEG, within MAX_M."""
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from streetview_to_3d.postprocess.blend import blend
from streetview_to_3d.postprocess.depth_buffer import depth_buffer
from streetview_to_3d.postprocess.pano_at import pano_at
from streetview_to_3d.postprocess.pixel import pixel

NADIR_DEG = 70
MAX_M = 25.0
BLEND_MIN = 0.01    # a camera counting less than this is not sampled at all


def paint(points, occluders, cameras, photos, max_m=MAX_M):
    """(colours, index of the nearest camera that painted each point, or -1).

    occluders: DA3's points; photos[k]: (image path, drop mask) or None.
    Only points and occluders within max_m of a camera are looked at."""
    n = len(points)
    to_paint = cKDTree(points) if n else None
    hiding = cKDTree(occluders) if len(occluders) else None
    seen = [None] * len(cameras)                      # each camera's: (points, how far, u, v)
    best = np.full(n, np.inf)
    who = np.full(n, -1)
    for k, (cam, ph) in enumerate(zip(cameras, photos)):
        if ph is None or to_paint is None:
            continue
        idx = np.asarray(to_paint.query_ball_point(cam.centre, max_m), int)
        if not len(idx):
            continue
        o = np.asarray(hiding.query_ball_point(cam.centre, max_m + 0.1) if hiding is not None else [], int)
        near = depth_buffer(cam, occluders[o])
        u, v, r, below = cam.look(points[idx])
        ok = (r <= near[pixel(u, v)] + 0.1) & (r < max_m) & (below < NADIR_DEG) & ~pano_at(ph[1], u, v)
        idx, r, u, v = idx[ok], r[ok], u[ok], v[ok]
        seen[k] = idx, r, u, v
        closer = r < best[idx]                        # nearest camera wins; first of equals
        best[idx[closer]], who[idx[closer]] = r[closer], k

    # mix every camera that can colour a point, weighted by blend
    colours, total = np.zeros((n, 3)), np.zeros(n)
    for k, ph in enumerate(photos):
        if seen[k] is None:
            continue
        idx, r, u, v = seen[k]
        wt = blend(r, best[idx])
        keep = wt > BLEND_MIN
        if keep.any():
            img = np.asarray(Image.open(ph[0]).convert("RGB"))
            colours[idx[keep]] += wt[keep, None] * pano_at(img, u[keep], v[keep]) / 255.0
            total[idx[keep]] += wt[keep]
    colours /= np.maximum(total, 1e-9)[:, None]
    return colours, who
