"""Colour for what fill adds (the one ground, Google's walls).

DA3's own points keep the colours DA3 gave them.

A point no camera may colour (the spot right under a camera, masked
spots, floor behind a fence) is left to the caller: the fill gives it the
colour of the nearest painted point.

Each added point mixes every camera that can colour it, the nearest most,
one BLEND_M further 1/e as much: deep in one camera's patch it is that
photo, and where two cameras' patches meet the colour fades from one
photo's exposure to the other's over a few metres instead of a hard edge.
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
from scipy.ndimage import binary_dilation, label, uniform_filter

NADIR_DEG = 70
MAX_M = 25.0
ZB_W = 512
BLEND_M, BLEND_MIN = 2.0, 0.01     # a camera BLEND_M further than the nearest counts 1/e as much
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
    grow = int(BLUR_GROW_DEG / 180 * h)
    return binary_dilation(np.arange(h)[:, None] >= top[None, :], np.ones((2 * grow + 1, 1), bool))


def _at(grid, u, v):
    h, w = grid.shape[:2]
    return grid[np.clip((v * h).astype(int), 0, h - 1), np.clip((u * w).astype(int), 0, w - 1)]


def paint(points, occluders, cameras, photos, max_m=MAX_M):
    """(colours, which camera painted each point -- the nearest -- or -1).
    points: the added points; occluders: DA3's points, what can hide them;
    photos[k]: (image path, drop mask) of cameras[k]'s pano, or None."""
    n, K = len(points), len(cameras)
    sees = np.zeros((K, n), bool)
    dist = np.full((K, n), np.inf)
    looks = [None] * K
    h, w = ZB_W // 2, ZB_W
    for k, (cam, ph) in enumerate(zip(cameras, photos)):
        if ph is None:
            continue
        u, v, r, _ = cam.look(occluders)
        near = np.full(h * w, np.inf)
        o = np.argsort(-r)                            # nearest written last wins
        iu, iv = (u[o] * w).astype(int), (v[o] * h).astype(int)
        for du in (-1, 0, 1):                         # a point covers its neighbours too
            for dv in (-1, 0, 1):
                near[np.clip(iv + dv, 0, h - 1) * w + (iu + du) % w] = r[o]
        u, v, r, below = cam.look(points)
        px = np.clip((v * h).astype(int), 0, h - 1) * w + (u * w).astype(int) % w
        visible = (r <= near[px] + 0.1) & (r < max_m)
        sees[k] = visible & (below < NADIR_DEG) & ~_at(ph[1], u, v)
        dist[k] = r
        looks[k] = (u, v)

    # per point: the nearest camera that can colour it (whose node it joins),
    # and every camera that can, mixed by how much further it is than that
    near = np.where(sees, dist, np.inf)
    best = near.min(0)
    who = np.where(np.isfinite(best), near.argmin(0), -1)
    colours, total = np.zeros((n, 3)), np.zeros(n)
    for k, ph in enumerate(photos):
        w = np.exp(-(near[k] - np.where(np.isfinite(best), best, 0)) / BLEND_M)
        idx = np.flatnonzero(w > BLEND_MIN)
        if len(idx):
            img = np.asarray(Image.open(ph[0]).convert("RGB"))
            u, v = looks[k]
            colours[idx] += w[idx, None] * _at(img, u[idx], v[idx]) / 255.0
            total[idx] += w[idx]
    colours /= np.maximum(total, 1e-9)[:, None]
    return colours, who
