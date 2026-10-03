"""Web-mercator tile maps (heights, imagery, water) read at any (lat, lon), each tile downloaded
once into TILES_DIR and kept."""
import io
import math
import os
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from streetview_to_3d.common.paths import DATA_DIR

TILE_THREADS = 8
TILES_DIR = os.path.join(DATA_DIR, "tiles")     # downloaded tiles, kept forever
TILE_TRIES = 3


class TileMap:
    """A web-mercator tile map read bilinearly at any (lat, lon), tiles cached in TILES_DIR.

    decode turns a tile's pixels (0-255, in mode) into values; a 404 tile
    reads as missing (or raises if missing is None)."""

    def __init__(self, url, zoom, decode, mode="RGB", missing=None, headers=None):
        self.url, self.zoom, self.decode, self.tiles = url, zoom, decode, {}
        self.mode, self.missing = mode, missing
        self.headers = headers or {"User-Agent": "streetview-to-3d"}

    def download(self, url):
        """The tile's bytes, or None where the map has none (a 404)."""
        import hashlib
        path = os.path.join(TILES_DIR, hashlib.sha1(url.encode()).hexdigest())
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read() or None
        for attempt in range(TILE_TRIES):
            try:
                req = urllib.request.Request(url, headers=self.headers)
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = r.read()
                break
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
                data = b""
                break
            except OSError:
                if attempt == TILE_TRIES - 1:
                    raise
        os.makedirs(TILES_DIR, exist_ok=True)
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)
        return data or None

    def _tile(self, x, y):
        if (x, y) not in self.tiles:
            data = self.download(self.url.format(z=self.zoom, x=x, y=y))
            if data is None:
                if self.missing is None:
                    raise OSError(f"no tile {self.zoom}/{x}/{y}")
                self.tiles[x, y] = np.full((256, 256, 3) if self.mode == "RGB" else (256, 256), self.missing)
            else:
                self.tiles[x, y] = self.decode(np.asarray(Image.open(io.BytesIO(data)).convert(self.mode), float))
        return self.tiles[x, y]

    def _xy(self, lat, lon):
        """Pixel (x, y) of (lat, lon) on the whole map."""
        n = 256 * 2 ** self.zoom
        return ((lon + 180) / 360 * n,
                (1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / math.pi) / 2 * n)

    def fetch(self, lat, lon, near=0):
        """Prefetch every tile within near pixels of the points, TILE_THREADS at once."""
        from concurrent.futures import ThreadPoolExecutor
        x, y = self._xy(np.asarray(lat), np.asarray(lon))
        tiles = {(int(a // 256), int(b // 256)) for dx in (-near, near) for dy in (-near, near)
                 for a, b in zip(x + dx, y + dy)}
        with ThreadPoolExecutor(TILE_THREADS) as pool:
            list(pool.map(lambda t: self.download(self.url.format(z=self.zoom, x=t[0], y=t[1])), tiles))
        return len(tiles)

    def __call__(self, lat, lon):
        if not np.size(lat):                # empty, shaped as the map's values
            return self.decode(np.zeros((0, 3) if self.mode == "RGB" else (0,), float))
        px, py = self._xy(lat, lon)
        px, py = px - 0.5, py - 0.5
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
