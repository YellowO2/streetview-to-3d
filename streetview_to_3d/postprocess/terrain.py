"""The land around the scene, from free global maps.

DA3 and the backdrop reach a few tens of metres; hills and mountains
further out come from AWS Terrain Tiles (terrarium PNGs, no key, ~30 m
data, mostly SRTM -- the ground, big buildings at most a blur), coloured
from EOX's Sentinel-2 cloudless mosaic (no key, 10 m, CC BY-NC-SA: credit
"Sentinel-2 cloudless by EOX", not for sale). Sampled as points, dense
near the scene and sparser with distance so each covers about the same
share of the view:

1. points on rings out to RADIUS_M, STEP of their distance apart (never
   under MIN_STEP_M), none within NEAR_M of a camera -- the scene's own
   ground is better there
2. height read off the tiles, then the whole terrain shifted so it meets
   the ground under the cameras (their panos' elevation): the
   map's datum and Google's differ by a few metres. The sea (the tiles
   also carry the sea bed) laid flat at sea level
3. coloured from the satellite, lifted a little (it is dark from above),
   with a light slope shading so relief reads; plain if the imagery
   cannot be had

Written to terrain.ply beside scene.json (its "terrain"), already in the
world frame. The viewer draws its points larger with distance, as they
are spaced (scene-store.js, terrainBands).

    python -m streetview_to_3d.postprocess.terrain SCENE_DIR
"""
import io
import math
import os
import sys
import urllib.request

import numpy as np
from PIL import Image

from streetview_to_3d import scene as scene_mod
from streetview_to_3d.postprocess.ply_io import write_ply

FILENAME = "terrain.ply"
HEIGHT_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
HEIGHT_ZOOM = 13          # ~19 m a pixel at the equator, finer than the data
COLOUR_URL = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2025_3857/default/g/{z}/{y}/{x}.jpg"
COLOUR_ZOOM = 14          # ~10 m a pixel, the imagery's own
RADIUS_M = 2000.0
NEAR_M = 25.0
STEP, MIN_STEP_M = 0.01, 0.5
M_PER_LAT = 111320.0
PLAIN = np.array([0.50, 0.55, 0.45])
LIFT = 0.75               # colour ** LIFT: brighter shadows, same hues
SEA_M = 0.5               # map height at or under this is sea
SUN = np.array([-0.5, -0.7, 0.5])        # world frame: x east, y down, z north


class TileMap:
    """A web-mercator tile map read at any (lat, lon), bilinear; each tile
    downloaded once. decode turns a tile's RGB (0-255) into its values."""

    def __init__(self, url, zoom, decode):
        self.url, self.zoom, self.decode, self.tiles = url, zoom, decode, {}

    def _tile(self, x, y):
        if (x, y) not in self.tiles:
            req = urllib.request.Request(self.url.format(z=self.zoom, x=x, y=y),
                                         headers={"User-Agent": "streetview-to-3d"})
            with urllib.request.urlopen(req, timeout=30) as r:
                rgb = np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"), float)
            self.tiles[x, y] = self.decode(rgb)
        return self.tiles[x, y]

    def __call__(self, lat, lon):
        n = 256 * 2 ** self.zoom
        px = (lon + 180) / 360 * n - 0.5
        py = (1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / math.pi) / 2 * n - 0.5
        x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
        fx, fy = px - x0, py - y0
        tx0, ty0 = x0.min() // 256, y0.min() // 256
        grid = np.concatenate([np.concatenate([self._tile(tx, ty) for tx in range(tx0, (x0.max() + 1) // 256 + 1)], 1)
                               for ty in range(ty0, (y0.max() + 1) // 256 + 1)], 0)
        gx, gy = x0 - tx0 * 256, y0 - ty0 * 256
        if grid.ndim == 3:
            fx, fy = fx[:, None], fy[:, None]
        v = lambda dx, dy: grid[gy + dy, gx + dx]
        return ((1 - fx) * (1 - fy) * v(0, 0) + fx * (1 - fy) * v(1, 0)
                + (1 - fx) * fy * v(0, 1) + fx * fy * v(1, 1))


def height_map():
    """Metres above sea level."""
    return TileMap(HEIGHT_URL, HEIGHT_ZOOM, lambda c: c[..., 0] * 256 + c[..., 1] + c[..., 2] / 256 - 32768)


def colour_map():
    """RGB, 0-1."""
    return TileMap(COLOUR_URL, COLOUR_ZOOM, lambda c: c / 255)


def sample_points():
    """(east, north) on rings about the centre, STEP of their radius apart."""
    r, rings = MIN_STEP_M, []
    rng = np.random.default_rng(0)
    while r < RADIUS_M:
        step = max(MIN_STEP_M, STEP * r)
        k = max(6, int(2 * math.pi * r / step))
        a = (np.arange(k) + rng.random()) * 2 * math.pi / k
        rings.append(np.stack([r * np.cos(a), r * np.sin(a)], 1))
        r += step
    return np.concatenate(rings)


def build(scene_dir, log=print):
    """Write scene_dir/FILENAME around the placed scene's cameras."""
    from scipy.spatial import cKDTree
    sc = scene_mod.Scene.load(scene_dir)
    lat0, lon0 = sc.origin
    m_per_lon = M_PER_LAT * math.cos(math.radians(lat0))
    cams = [n for n in sc.nodes if n.transform and n.position is not None]
    if not cams:
        log("terrain: no placed cameras")
        return
    cam_xz = np.array([(np.asarray(n.transform)[:3, :3] @ n.position + np.asarray(n.transform)[:3, 3])[[0, 2]]
                       for n in cams])
    en = sample_points()
    en = en[cKDTree(cam_xz).query(en)[0] > NEAR_M]
    lat, lon = lat0 + en[:, 1] / M_PER_LAT, lon0 + en[:, 0] / m_per_lon

    # one shift so the map meets the ground the cameras stand on
    heights = height_map()
    under = heights(np.array([n.pano.lat for n in cams]), np.array([n.pano.lon for n in cams]))
    ground = np.array([n.pano.elevation if n.pano.elevation is not None else np.nan for n in cams])
    shift = float(np.nanmedian(ground - under)) if np.isfinite(ground).any() else 0.0
    raw = heights(lat, lon)
    sea = raw <= SEA_M
    h = np.where(sea, 0.0, raw) + shift          # the tiles carry the sea bed too; lay the sea flat
    pts = np.stack([en[:, 0], -h, en[:, 1]], 1)

    # slope shading from the height a metre east and north
    d = 1.0
    he = heights(lat, lon0 + (en[:, 0] + d) / m_per_lon)
    hn = heights(lat0 + (en[:, 1] + d) / M_PER_LAT, lon)
    normal = np.stack([-(he - raw) / d, -np.ones(len(h)), -(hn - raw) / d], 1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    shade = np.clip(normal @ (SUN / np.linalg.norm(SUN)), 0, 1)
    try:
        cols = colour_map()(lat, lon) ** LIFT * (0.8 + 0.2 * shade[:, None])
        source = "satellite"
    except OSError as e:                  # the land still stands without its colour
        log(f"terrain: no satellite colour ({e!r}), plain")
        cols = PLAIN * (0.55 + 0.45 * shade[:, None])
        source = "plain"
    write_ply(os.path.join(scene_dir, FILENAME), pts, cols)
    sc.terrain = FILENAME
    sc.save(scene_dir)
    log(f"terrain: {len(pts)} points to {RADIUS_M:.0f} m ({int(sea.sum())} sea, {source} colour), "
        f"heights {h.min():.0f}..{h.max():.0f} m, shifted {shift:+.1f} m to meet the cameras' "
        f"ground (spread {np.nanstd(ground - under):.1f} m)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__.splitlines()[-1].strip())
    build(sys.argv[1])
