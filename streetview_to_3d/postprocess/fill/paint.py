"""Colour for the fill's ground from the panos, mixing every camera that sees a point cleanly:
nothing of DA3's in front (depth_buffer), not masked or blurred, under NADIR_DEG, within MAX_M."""
import numpy as np
from PIL import Image
from scipy.ndimage import label, uniform_filter
from scipy.spatial import cKDTree

from streetview_to_3d.postprocess.fill.ground import blend

NADIR_DEG = 70
MAX_M = 25.0
ZB_W = 512
BLEND_MIN = 0.01    # a camera counting less than this is not sampled at all
BLUR_DETAIL, BLUR_WIN = 0.5, 15     # under this grey-level change per pixel, over BLUR_WIN px: blur
BLUR_GROW_DEG, SEAM_DEG = 6, 2


class Camera:
    """A placed node's camera: world points to its photo's (u, v), distance
    in metres and degrees below the horizon."""

    def __init__(self, node):
        T = np.asarray(node.transform, float)
        self.scale = np.cbrt(np.linalg.det(T[:3, :3]))
        self.R, self.t = T[:3, :3] / self.scale, T[:3, 3]
        self.position = np.asarray(node.position, float)
        self.rotation = np.asarray(node.rotation, float)
        self.centre = T[:3, :3] @ self.position + self.t

    def look(self, x):
        d = (((x - self.t) @ self.R) / self.scale - self.position) @ self.rotation.T
        r = np.linalg.norm(d, axis=1)
        s = np.clip(d[:, 1] / np.maximum(r, 1e-9), -1, 1)
        below = np.degrees(np.arcsin(s))
        return np.arctan2(d[:, 0], d[:, 2]) / (2 * np.pi) + .5, below / 180 + .5, r * self.scale, below


def blurred(path):
    """Mask of the smooth blur hiding the capture rig below the horizon in many Google panos.

    Low-detail regions (BLUR_DETAIL) connected to the bottom count, plus
    everything below them in each column, grown BLUR_GROW_DEG upward. The
    bottom SEAM_DEG rows are a hard seam and are skipped."""
    g = np.asarray(Image.open(path).convert("L"), float)
    h = g.shape[0]
    d = np.zeros_like(g)
    d[:, :-1] += np.abs(np.diff(g, axis=1))
    d[:-1] += np.abs(np.diff(g, axis=0))
    m = uniform_filter(d[:h - int(SEAM_DEG / 180 * h)], BLUR_WIN, mode=["nearest", "wrap"]) < BLUR_DETAIL
    m[:h // 2] = False
    lab, _ = label(m)
    m = np.isin(lab, np.unique(lab[-1][lab[-1] > 0]))
    top = np.where(m.any(0), m.argmax(0), h)
    # the blur reaches each column's bottom, so growing it moves its top up
    top = np.where(top < h, top - int(BLUR_GROW_DEG / 180 * h), h)
    return np.arange(h)[:, None] >= top[None, :]


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


def pixel(u, v):
    """Index into depth_buffer's pixels of view coordinates (u, v)."""
    h, w = ZB_W // 2, ZB_W
    return np.clip((v * h).astype(int), 0, h - 1) * w + (u * w).astype(int) % w


def _at(grid, u, v):
    h, w = grid.shape[:2]
    return grid[np.clip((v * h).astype(int), 0, h - 1), np.clip((u * w).astype(int), 0, w - 1)]


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
        ok = (r <= near[pixel(u, v)] + 0.1) & (r < max_m) & (below < NADIR_DEG) & ~_at(ph[1], u, v)
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
            colours[idx[keep]] += wt[keep, None] * _at(img, u[keep], v[keep]) / 255.0
            total[idx[keep]] += wt[keep]
    colours /= np.maximum(total, 1e-9)[:, None]
    return colours, who
