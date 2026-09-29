"""The water around the scene: flat surfaces, not points.

Where it is comes from the JRC Global Surface Water map (EC JRC / Google,
no key, ~30 m Landsat, credit "EC JRC/Google"): how often, 1984-2021, a
pixel was water; at least WET of the time counts. OpenStreetMap's water
was tried first and is not usable here: a lake or the sea is one outline
of thousands of ways, and around Stockholm the request timed out. The sea
is also where the height tiles are at sea level (terrain.SEA_M), as the
land has always laid it.

Laid on a grid of CELL_M, the wet cells become outlines (shapely), each
one body of water flat at its own level: the ground's height there
(terrain's, Google's datum) at its LEVEL_PCT percentile -- the height map
reads a lake's surface, noisily.

Written to water.json beside scene.json (its "water"), east/north metres:
    {"surfaces": [{"level": m, "outer": [[e, n], ...], "holes": [[[e, n], ...], ...]}]}
The viewer draws each as one flat shape (effects/water.js); terrain.py
leaves out its points where there is water.
"""
import json
import os

import numpy as np

FILENAME = "water.json"
OCCURRENCE_URL = "https://storage.googleapis.com/global-surface-water/tiles2021/occurrence/{z}/{x}/{y}.png"
OCCURRENCE_ZOOM = 13      # the finest the map is served at
WET = 0.5                 # water at least this share of the time
CELL_M = 5.0
MIN_M2 = 400.0            # smaller bodies are left to the land
LEVEL_PCT = 20


def occurrence_map():
    """How often water, 0-1, at any (lat, lon); 0 where the map has none.

    The tiles run from pink (rarely) to blue (always): 1 - red. Where
    there never was water they are clear, and a tile with none at all is
    not served."""
    from streetview_to_3d.postprocess.terrain import TileMap
    return TileMap(OCCURRENCE_URL, OCCURRENCE_ZOOM,
                   lambda c: np.where(c[..., 3] > 0, 1 - c[..., 0] / 255, 0.0),
                   mode="RGBA", missing=0.0)


class Water:
    """Where the water is within radius_m, on a CELL_M grid.

    wet(xy) says it for east/north points; surfaces(ground) are its
    outlines, each at its level."""

    def __init__(self, radius_m, to_ll, sea):
        n = int(np.ceil(radius_m / CELL_M))
        self.lo = -n * CELL_M
        c = self.lo + CELL_M * (np.arange(2 * n) + 0.5)
        gx, gy = np.meshgrid(c, c)
        self.centres = np.stack([gx.ravel(), gy.ravel()], 1)
        lat, lon = to_ll(self.centres)
        try:
            often = occurrence_map()(lat, lon)
            self.source = "JRC"
        except OSError:                       # the sea still stands
            often = np.zeros(len(lat))
            self.source = "sea only"
        self.mask = ((often >= WET) | sea(lat, lon)).reshape(gx.shape)
        self.mask &= (gx ** 2 + gy ** 2) < radius_m ** 2

    def wet(self, xy):
        i = np.floor((xy - self.lo) / CELL_M).astype(int)
        inside = ((i >= 0) & (i < self.mask.shape[0])).all(1)
        out = np.zeros(len(xy), bool)
        out[inside] = self.mask[i[inside, 1], i[inside, 0]]
        return out

    def surfaces(self, ground):
        """[{"level", "outer", "holes"}]: the wet cells as outlines, one
        per body of water, simplified to half a cell."""
        import shapely
        from shapely.geometry import box
        boxes = []
        for r, row in enumerate(self.mask):      # a run of wet cells in a row is one box
            edges = np.flatnonzero(np.diff(np.r_[0, row.astype(int), 0]))
            y = self.lo + r * CELL_M
            boxes += [box(self.lo + a * CELL_M, y, self.lo + b * CELL_M, y + CELL_M)
                      for a, b in zip(edges[::2], edges[1::2])]
        if not boxes:
            return []
        merged = shapely.union_all(boxes).simplify(CELL_M / 2)
        out = []
        for p in getattr(merged, "geoms", [merged]):
            if p.area < MIN_M2:
                continue
            inside = self.centres[shapely.contains_xy(p, *self.centres.T)]
            level = float(np.percentile(ground(inside if len(inside) else
                                               np.array(p.representative_point().coords)), LEVEL_PCT))
            out.append({"level": round(level, 2),
                        "outer": np.round(p.exterior.coords, 2).tolist(),
                        "holes": [np.round(h.coords, 2).tolist() for h in p.interiors]})
        return out


def save(scene_dir, surfaces):
    with open(os.path.join(scene_dir, FILENAME), "w") as f:
        json.dump({"surfaces": surfaces}, f)
    return FILENAME
