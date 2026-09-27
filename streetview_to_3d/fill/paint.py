"""Colour for what fill adds (the one ground, Google's walls).

DA3's own points keep the colours DA3 gave them. Each added point takes its
colour from the nearest pano that sees it cleanly -- the one whose camera
is closest among those that:
  - see it: nothing nearer along that line of sight (a z-buffer of the
    whole scene from that camera)
  - see it as itself: not a masked car, person or pole in that photo
  - are not looking at their own rig: not more than NADIR_DEG below the
    horizon, where every pano has its blurred spot (a capture car's roof is
    masked as a car)
  - are within MAX_M
The nearest camera is also the pano whose own blind disc a point fills, so
a filled hole matches the ground around it.
"""
import numpy as np
from PIL import Image

NADIR_DEG = 70
MAX_M = 25.0
ZB_W = 1024


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


def _at(grid, u, v):
    h, w = grid.shape[:2]
    return grid[np.clip((v * h).astype(int), 0, h - 1), np.clip((u * w).astype(int), 0, w - 1)]


def paint(points, scene, cameras, photos):
    """(colours, which camera painted each point, -1 for none).
    points: the added points; scene: everything, for what hides what;
    photos[k]: (image path, drop mask) of cameras[k]'s pano, or None."""
    best = np.full(len(points), np.inf)
    who = np.full(len(points), -1)
    looks = []
    h, w = ZB_W // 2, ZB_W
    for k, (cam, ph) in enumerate(zip(cameras, photos)):
        looks.append(None)
        if ph is None:
            continue
        u, v, r, _ = cam.look(scene)
        near = np.full(h * w, np.inf)
        o = np.argsort(-r)                            # nearest written last wins
        iu, iv = (u[o] * w).astype(int), (v[o] * h).astype(int)
        for du in (-1, 0, 1):                         # a point covers its neighbours too
            for dv in (-1, 0, 1):
                near[np.clip(iv + dv, 0, h - 1) * w + (iu + du) % w] = r[o]
        u, v, r, below = cam.look(points)
        px = np.clip((v * h).astype(int), 0, h - 1) * w + (u * w).astype(int) % w
        sees = ((r <= near[px] * 1.03 + 0.1) & (below < NADIR_DEG) & (r < MAX_M)
                & ~_at(ph[1], u, v))
        closer = sees & (r < best)
        best[closer], who[closer] = r[closer], k
        looks[k] = (u, v)
    colours = np.zeros((len(points), 3))
    for k, ph in enumerate(photos):
        idx = np.flatnonzero(who == k)
        if len(idx):
            img = np.asarray(Image.open(ph[0]).convert("RGB"))
            u, v = looks[k]
            colours[idx] = _at(img, u[idx], v[idx]) / 255.0
    return colours, who
