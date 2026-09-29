"""The water around the scene: flat surfaces, not points.

Where it is comes from the JRC Global Surface Water map (EC JRC / Google,
no key, ~30 m Landsat, credit "EC JRC/Google"): how often, 1984-2021, a
pixel was water; at least WET of the time counts. OpenStreetMap's water
was tried first and is not usable here: a lake or the sea is one outline
of thousands of ways, and around Stockholm the request timed out. The sea
is also where the height tiles are at sea level (terrain.SEA_M), as the
land has always laid it.

Laid on a grid of CELL_M, the wet cells become outlines (shapely), each
grown UNDER_LAND_M, under its shore: the map's outline is rough (30 m),
and cut exactly to it the land stopped short of the water or the water of
the land, a gap between them. Grown, the land's edge stands over the
water's (terrain.py leaves out only land below its level), as a game lays
its sea under the coast. Water whose grown outlines touch is one body --
the map breaks a harbour at every bridge -- flat at one level, in metres
above the sea as Google's elevation is (it matched the scene's own ground
to 0.1 m, Stockholm):

  - the height map's lowest (LOW_PCT) over it, moved as the land is onto
    Google's (shift): over Stockholm's harbour, at the sea, the map read
    3.6-20 m, as high as it reads the land -- radar over water is waves,
    boats and bridges, and a 30 m pixel takes in the quay
  - never below the sea (0)
  - never within CLEAR_M of the ground at a pano CAP_M near it: the street
    was dry

Written to water.json beside scene.json (its "water"), east/north metres:
    {"surfaces": [{"level": m, "outer": [[e, n], ...], "holes": [[[e, n], ...], ...]}]}
The viewer draws each as one flat shape (effects/water.js); terrain.py
leaves out its points where there is water.
"""
import json
import os

import numpy as np
import shapely

FILENAME = "water.json"
OCCURRENCE_URL = "https://storage.googleapis.com/global-surface-water/tiles2021/occurrence/{z}/{x}/{y}.png"
OCCURRENCE_ZOOM = 13      # the finest the map is served at
WET = 0.5                 # water at least this share of the time
CELL_M = 5.0
MIN_M2 = 400.0            # smaller bodies are left to the land
UNDER_LAND_M = 20.0
LOW_PCT = 5
CAP_M, CLEAR_M = 200.0, 0.5


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
    """The water within radius_m, on a CELL_M grid: to_ll(east/north) ->
    (lat, lon); height(lat, lon), the height map's metres above the sea,
    and shift, what moves it onto Google's; panos, the (east/north (n, 2),
    ground height (n,)) of the cameras.

    surfaces are its outlines, grown under the land, each at its level;
    level_at(xy) the level of the water over east/north points (-inf where
    there is none): land below it is under water."""

    def __init__(self, radius_m, to_ll, height, shift, panos):
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
        from streetview_to_3d.postprocess.terrain import SEA_M    # terrain imports this module
        h = height(lat, lon)
        mask = ((often >= WET) | (h <= SEA_M)).reshape(gx.shape)
        mask &= (gx ** 2 + gy ** 2) < radius_m ** 2
        wet = mask.ravel()
        self.surfaces, self.level = [], np.full(gx.shape, -np.inf)
        bodies = _bodies(mask, self.lo)
        if not bodies:
            return
        grown = shapely.union_all([b.buffer(UNDER_LAND_M) for b in bodies]).simplify(CELL_M / 2)
        pano_xy, pano_ground = panos
        for body in getattr(grown, "geoms", [grown]):
            under = shapely.contains_xy(body, *self.centres.T)
            level = float(np.percentile(h[under & wet], LOW_PCT)) + shift if (under & wet).any() else 0.0
            level = max(level, 0.0)
            near = np.array([body.distance(shapely.Point(p)) < CAP_M for p in pano_xy], bool)
            if near.any():
                level = min(level, float(pano_ground[near].min()) - CLEAR_M)
            level = round(level, 2)
            under = under.reshape(gx.shape)
            self.level[under] = np.maximum(self.level[under], level)
            self.surfaces.append({"level": level,
                                  "outer": np.round(body.exterior.coords, 2).tolist(),
                                  "holes": [np.round(r.coords, 2).tolist() for r in body.interiors]})

    def level_at(self, xy):
        i = np.floor((xy - self.lo) / CELL_M).astype(int)
        inside = ((i >= 0) & (i < self.level.shape[0])).all(1)
        out = np.full(len(xy), -np.inf)
        out[inside] = self.level[i[inside, 1], i[inside, 0]]
        return out


def _bodies(mask, lo):
    """The wet cells of mask (a grid from lo, CELL_M a cell) as one outline
    per body of water, simplified to half a cell; small ones left out."""
    from shapely.geometry import box
    boxes = []
    for r, row in enumerate(mask):               # a run of wet cells in a row is one box
        edges = np.flatnonzero(np.diff(np.r_[0, row.astype(int), 0]))
        y = lo + r * CELL_M
        boxes += [box(lo + a * CELL_M, y, lo + b * CELL_M, y + CELL_M)
                  for a, b in zip(edges[::2], edges[1::2])]
    if not boxes:
        return []
    merged = shapely.union_all(boxes).simplify(CELL_M / 2)
    return [p for p in getattr(merged, "geoms", [merged]) if p.area >= MIN_M2]


def save(scene_dir, surfaces):
    with open(os.path.join(scene_dir, FILENAME), "w") as f:
        json.dump({"surfaces": surfaces}, f)
    return FILENAME
