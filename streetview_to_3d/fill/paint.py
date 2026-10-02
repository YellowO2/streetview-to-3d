"""Colour for what fill adds (the one ground, Google's walls).

DA3's own points keep the colours DA3 gave them.

A point no camera may colour (the spot right under a camera, masked
spots, floor behind a fence) is left to the caller: the fill gives it the
colour of the nearest painted point.

Each added point mixes every camera that can colour it, as the ground's
height is mixed (postprocess.ground.blend): where two cameras' patches
meet the colour fades from one photo's exposure to the other's over a few
metres instead of a hard edge.
A camera can colour a point when it:
  - sees it: no DA3 point in front of it along that line of sight. Only
    DA3's points count -- its ground is gone by now, so they are the real
    things in the way (fences, cars, bushes, walls) -- and each covers a
    ZB_W-wide view's pixel and its neighbours, so a surface blocks solidly
    rather than leaking through the gaps between its points. The added
    ground and walls never hide each other: at a shallow angle one pixel
    spans metres of ground, and its near end would hide its far end.
  - sees it as itself: not a masked car, person or pole in that photo,
    nor the blur some panos have below the horizon (blurred, below)
  - is not looking at its own rig: not more than NADIR_DEG below the
    horizon, where every pano has its blurred spot (a capture car's roof
    is masked as a car; 55 cost backpack captures their clean ground)
  - is within MAX_M
The nearest camera is also the pano whose own blind disc a point fills,
so a filled hole matches the ground around it.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import label, uniform_filter
from scipy.spatial import cKDTree

from streetview_to_3d.postprocess.ground import blend

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
    """Where a pano's photo is blur below the horizon, as a mask.

    Many Google panos hide the capture rig under a smooth grey smear
    reaching 15-35 deg below the horizon, ragged with where the car was --
    no road in it at all, and nearest-camera colour made the road a
    patchwork of it and real asphalt. Blur is what has next to no detail
    (BLUR_DETAIL); only blur joined to straight down counts (a smooth car
    door or wall does not), everything under it in its column too (the
    "(c) Google" marks in it), and its edge grows BLUR_GROW_DEG upward
    over the fade into the real photo. The photo's last SEAM_DEG rows are
    a hard seam and are left out of the search."""
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
    # it reaches the bottom of each column, so growing it is moving its top up
    top = np.where(top < h, top - int(BLUR_GROW_DEG / 180 * h), h)
    return np.arange(h)[:, None] >= top[None, :]


def _at(grid, u, v):
    h, w = grid.shape[:2]
    return grid[np.clip((v * h).astype(int), 0, h - 1), np.clip((u * w).astype(int), 0, w - 1)]


def paint(points, occluders, cameras, photos, max_m=MAX_M):
    """(colours, which camera painted each point -- the nearest -- or -1).
    points: the added points; occluders: DA3's points, what can hide them;
    photos[k]: (image path, drop mask) of cameras[k]'s pano, or None. A
    camera only paints within max_m, so only what lies that near it -- to
    paint, or to hide what it paints -- is looked at, never all of them."""
    n = len(points)
    h, w = ZB_W // 2, ZB_W
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
        near = np.full(h * w, np.inf)
        if hiding is not None:
            o = np.asarray(hiding.query_ball_point(cam.centre, max_m + 0.1), int)
            if len(o):
                u, v, r, _ = cam.look(occluders[o])
                iu, iv = (u * w).astype(int), (v * h).astype(int)
                for du in (-1, 0, 1):                 # a point covers its neighbours too; the nearest wins
                    for dv in (-1, 0, 1):
                        np.minimum.at(near, np.clip(iv + dv, 0, h - 1) * w + (iu + du) % w, r)
        u, v, r, below = cam.look(points[idx])
        px = np.clip((v * h).astype(int), 0, h - 1) * w + (u * w).astype(int) % w
        ok = (r <= near[px] + 0.1) & (r < max_m) & (below < NADIR_DEG) & ~_at(ph[1], u, v)
        idx, r, u, v = idx[ok], r[ok], u[ok], v[ok]
        seen[k] = idx, r, u, v
        closer = r < best[idx]                        # the nearest camera; the first of equals
        best[idx[closer]], who[idx[closer]] = r[closer], k

    # every camera that can colour a point, mixed by how much further it is than the nearest
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
