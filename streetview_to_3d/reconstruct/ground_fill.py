"""Fill the floor under each camera from Google's own ground.

DA3's views stop about 29 deg below the horizon, so every camera leaves a
disc of road about 4 m across that nothing sees -- the floor hole. Google's
depth map for the same pano (services.streetview_fetch.fetch_depth) is
coarse everywhere except the ground, which it has exactly, in metres --
ramps and steps included: a plaza can sit a metre above or below the road.
So the disc is rebuilt from Google's depth there, pixel by pixel, scaled
into the piece's DA3 units by comparing the pano's own points with
Google's depth at the same pixels.

Its colour cannot come from the pano itself: from about 35 deg down, a
Street View photo shows the car's own roof. It comes from a neighbour in
the same piece, which sees the same patch of road at a shallow angle --
the nearest one that sees it unblocked, checked against that neighbour's
own Google depth. What no pano sees cleanly is left out, as are cells the
cloud already covers.

Google panos only; Apple has no depth map.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image

FILL_GROUND = True
GRID_W = 1024        # fill sampled on this equirect grid, GRID_W x GRID_W/2
FLAT_DEG = 15        # a Google surface is ground if it faces up to within this
MIN_DROP_M = 0.5     # and lies at least this far below the camera
MIN_BELOW_DEG = 25   # the fill starts just above where DA3's views stop (~29)
CAR_BELOW_DEG = 33   # below this, a pano's photo shows its own car
MIN_SOURCE_BELOW_DEG = 3
MAX_M = 15.0         # no fill farther from its camera than this
MAX_SOURCE_M = 20.0  # neighbours are often 10-20 m apart
SEEN_TOL = 0.2       # a colour source sees a point if its depth there agrees to 20%
CELL_UNITS = 0.25    # coverage is judged per cell of this size (DA3 units)
COVERED = 0.5        # a cell counts as covered with half as many points as the fill
MIN_MATCHES = 200    # own points on Google's ground needed to measure scale


@dataclass(eq=False)
class Pano:
    """One Google pano in a piece: its points and pose in the piece's DA3
    frame, Google's depth map and its photo."""
    points: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    depth: np.ndarray
    image_path: str
    ground: np.ndarray = None   # where Google's depth is ground, set by prepare()
    scale: float = None         # metres per DA3 unit
    _image: np.ndarray = None

    def image(self):
        if self._image is None:
            self._image = np.asarray(Image.open(self.image_path).convert("RGB"))
        return self._image


def _rays(h, w):
    """Unit rays for an h x w equirect grid, in the photo's own frame (x
    right, y down, z forward), plus each pixel's (u, v) in [0, 1)."""
    y, x = np.mgrid[0:h, 0:w]
    u, v = (x + .5) / w, (y + .5) / h
    lon, lat = (u - .5) * 2 * np.pi, (v - .5) * np.pi
    return np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], -1), u, v


def _look(d):
    """(u, v, distance, degrees below the horizon) for photo-frame vectors."""
    r = np.linalg.norm(d, axis=1)
    s = np.clip(d[:, 1] / np.maximum(r, 1e-9), -1, 1)
    return (np.arctan2(d[:, 0], d[:, 2]) / (2 * np.pi) + .5, np.arcsin(s) / np.pi + .5,
            r, np.degrees(np.arcsin(s)))


def _at(grid, u, v):
    h, w = grid.shape[:2]
    return grid[np.clip((v * h).astype(int), 0, h - 1), np.clip((u * w).astype(int), 0, w - 1)]


def _bilinear(grid, u, v):
    """grid sampled at (u, v) in [0, 1), wrapping around left-right."""
    h, w = grid.shape
    x, y = u * w - .5, np.clip(v * h - .5, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    x0, x1 = x0 % w, (x0 + 1) % w
    y1 = np.minimum(y0 + 1, h - 1)
    return ((grid[y0, x0] * (1 - fx) + grid[y0, x1] * fx) * (1 - fy)
            + (grid[y1, x0] * (1 - fx) + grid[y1, x1] * fx) * fy)


def prepare(p):
    """Find where p's Google depth is ground, and p's metres per DA3 unit.
    False if it has too few of its own points on that ground."""
    rays, _, _ = _rays(*p.depth.shape)
    pts = rays * p.depth[..., None]
    # a surface's facing from the depth map's own slope, pixel to pixel
    normal = np.cross(np.gradient(pts, axis=1), np.gradient(pts, axis=0))
    up = np.abs(normal[..., 1]) / np.maximum(np.linalg.norm(normal, axis=-1), 1e-9)
    p.ground = (p.depth > 0) & (up > np.cos(np.radians(FLAT_DEG))) & (pts[..., 1] > MIN_DROP_M)

    u, v, r, _ = _look((p.points - p.position) @ p.rotation.T)
    on = _at(p.ground, u, v) & (r > 0)
    if on.sum() < MIN_MATCHES:
        return False
    p.scale = float(np.median(_at(p.depth, u[on], v[on]) / r[on]))
    return True


def prepare_piece(panos):
    """prepare() each pano of one piece, and give them all the piece's
    median scale: they share one DA3 frame, so one scale, and a single
    pano's own reading can be far off (0.57 against 1.2 when its view of
    the ground was mostly something else). Returns the prepared ones."""
    ready = [p for p in panos if prepare(p)]
    if ready:
        scale = float(np.median([p.scale for p in ready]))
        for p in ready:
            p.scale = scale
    return ready


def _uncovered(new, existing):
    """Which fill points land in cells the existing cloud covers only
    thinly. Per cell rather than per point: DA3 often leaves a sparse grid
    of points over the hole, and a nearest-point test would keep the fill
    off every one of them, in stripes."""
    if not len(existing):
        return np.ones(len(new), bool)
    cells = np.floor(new / CELL_UNITS).astype(np.int64)
    keys, inv, fill_n = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    ex = np.floor(existing / CELL_UNITS).astype(np.int64)
    ex = ex[np.all((ex >= keys.min(0)) & (ex <= keys.max(0)), axis=1)]
    both, both_inv = np.unique(np.concatenate([keys, ex]), axis=0, return_inverse=True)
    both_inv = both_inv.reshape(-1)
    counts = np.bincount(both_inv[len(keys):], minlength=len(both))[both_inv[:len(keys)]]
    return (counts < COVERED * fill_n)[inv.reshape(-1)]


def _colour(pts, sources):
    """Colours for pts from the nearest source that sees each one cleanly,
    and which pts got one."""
    best = np.full(len(pts), np.inf)
    colours = np.zeros((len(pts), 3))
    for s in sources:
        u, v, r, below = _look((pts - s.position) @ s.rotation.T)
        metres = r * s.scale
        seen = ((below > MIN_SOURCE_BELOW_DEG) & (below < CAR_BELOW_DEG) & (metres < MAX_SOURCE_M)
                & _at(s.ground, u, v)
                & (np.abs(_at(s.depth, u, v) - metres) < SEEN_TOL * metres) & (metres < best))
        if seen.any():
            colours[seen] = _at(s.image(), u[seen], v[seen]) / 255.0
            best[seen] = metres[seen]
    return colours, np.isfinite(best)


def fill(target, sources, existing):
    """(points, colours) to add to target's cloud, in its piece's DA3
    frame, or None. target and sources are prepared Panos of the same
    piece (target may be among sources); existing is every point the
    piece already has."""
    rays, u, v = _rays(GRID_W // 2, GRID_W)
    rays, u, v = rays.reshape(-1, 3), u.reshape(-1), v.reshape(-1)
    below = (v - .5) * 180
    keep = np.flatnonzero((below > MIN_BELOW_DEG) & _at(target.ground, u, v))
    metres = _bilinear(np.where(target.ground, target.depth, 0), u[keep], v[keep])
    near = (metres > 0) & (metres < MAX_M)
    if not near.any():
        return None
    keep, metres = keep[near], metres[near]
    new = target.position + (rays[keep] * metres[:, None] / target.scale) @ target.rotation
    new = new[_uncovered(new, existing)]
    if not len(new):
        return None
    colours, ok = _colour(new, sources)
    return (new[ok], colours[ok]) if ok.any() else None
