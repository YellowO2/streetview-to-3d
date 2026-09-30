"""The water around the scene: flat surfaces, not points.

Where it is comes from OpenStreetMap within its box (osm.py; outlines
drawn to about a metre, a bridge not part of the water): its water
outlines, cut to the box, split it into pieces, each water or land as OSM
itself says (outline). Past the box, or without OSM, the JRC Global
Surface Water map itself (EC JRC / Google, no key, ~30 m Landsat, credit
"EC JRC/Google"): how often, 1984-2021, a pixel was water, at least WET of
the time counting, and the sea where the height tiles are at sea level
(terrain.SEA_M). At 30 m it is no more than roughly where water is:
around Stockholm's bridge it had water as land, and the bridge as water.

As a game has it: the land is one ground that goes on under the water,
the water a flat surface over it, and the shore is wherever the one
crosses the other -- nothing is cut. Laid on a grid of CELL_M, the wet
cells become outlines (shapely), each grown UNDER_LAND_M under its shore,
so its edge is under the land; and the shore is shaped as a game shapes
one (carve), measured on OSM's own line near it: the bed eases down to
DEPTH_M, the land eases down to the water over a bank (a quay by a road
stands), with sand along it (sand) -- whatever the height map says there (by a bridge it blurs the bridge and the island
into the harbour: 9-13 m over Stockholm's water). Land deeper than SEE_M
is left out (keep): the water hides it; shallower, it shows through the
water near the shore. Water whose grown outlines touch is one body --
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
    {"surfaces": [{"level": m, "outer": [[e, n], ...], "holes": [[[e, n], ...], ...]}],
     "shore": {"lo": m, "cell": m, "size": n, "metres": [n x n, row by row north-wards]}}
shore being how far each cell is from dry land, 0 on it, to SHORE_MAX_M: the
viewer draws each surface as one flat shape (effects/water.js), light and
clear near the shore, deep further out. DA3's own water -- at about
street height, grainy -- never becomes points: the masker marks it
(services.segment, "water"); what it sees through a bridge's railing
(ADE20K calls that strip railing) the fill does not take for ground
(fill.one_ground, WALKABLE).
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
DEPTH_M, SHELF_M = 3.0, 12.0      # the bed eases down to DEPTH_M over SHELF_M from the shore
BANK_M, BANK_VARY = 6.0, 0.5      # the land eases down to the water over about BANK_M, +- BANK_VARY of it
BANK_ABOVE_M = 0.15               # ... to this far above it at the shore
QUAY_M = 2.0                      # no bank this near a road: a quay stands
SAND_M, SAND_UP_M = 2.5, 1.0      # sand: this near the shore, at most this far above the water
EXACT_M, TRACE_M = 30.0, 0.25     # nearer the shore than this, how far measured on OSM's own line
SEE_M = 1.5
SHORE_CELL_M, SHORE_MAX_M = 10.0, 60
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
    ground height (n,)) of the cameras; where it is, osm (OSM's water, its
    box's half-width: outline) within the box, wet (east/north -> whether:
    jrc) past it.

    surfaces are its outlines, grown under the land, each at its level;
    carve and keep shape the land under it; shore how far from dry land
    (see the module)."""

    def __init__(self, radius_m, to_ll, height, shift, panos, wet, osm=None):
        n = int(np.ceil(radius_m / CELL_M))
        self.lo = -n * CELL_M
        c = self.lo + CELL_M * (np.arange(2 * n) + 0.5)
        gx, gy = np.meshgrid(c, c)
        self.centres = np.stack([gx.ravel(), gy.ravel()], 1)
        h = height(*to_ll(self.centres))
        near = (gx ** 2 + gy ** 2).ravel() < radius_m ** 2
        mask = np.zeros(len(h), bool)
        inside = (np.abs(self.centres) < osm[1]).all(1) & near if osm else np.zeros(len(h), bool)
        if inside.any():
            mask[inside] = shapely.contains_xy(osm[0], *self.centres[inside].T)
        past = near & ~inside
        if past.any():
            mask[past] = wet(self.centres[past])
        self.source = ("OSM" + (", JRC past it" if past.any() else "")) if osm else "JRC"
        mask = mask.reshape(gx.shape)
        from scipy.ndimage import distance_transform_edt
        self.mask = mask
        self.shore = distance_transform_edt(mask) * CELL_M
        self.inland = distance_transform_edt(~mask) * CELL_M
        # OSM's own shoreline, as points TRACE_M apart: the box's edge is no shore
        self.osm, self.coast = None, None
        if osm and not osm[0].is_empty:
            box = shapely.box(-osm[1], -osm[1], osm[1], osm[1])
            line = shapely.difference(osm[0].boundary, box.exterior.buffer(0.5))
            xy = shapely.get_coordinates(shapely.segmentize(line, TRACE_M))
            if len(xy):
                from scipy.spatial import cKDTree
                self.osm, self.coast = osm, cKDTree(xy)
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

    def _at(self, grid, xy, outside):
        i = np.floor((xy - self.lo) / CELL_M).astype(int)
        inside = ((i >= 0) & (i < grid.shape[0])).all(1)
        out = np.full(len(xy), outside, grid.dtype)
        out[inside] = grid[i[inside, 1], i[inside, 0]]
        return out

    def signed(self, xy):
        """Metres from the shore at east/north points: + in the water, - on
        land. On OSM's own line near it, else on the grid (half a cell off)."""
        wet = self._at(self.mask, xy, False)
        d = np.where(wet, self._at(self.shore, xy, np.inf), -self._at(self.inland, xy, np.inf)) \
            - np.sign(np.where(wet, 1, -1)) * CELL_M / 2
        if self.coast is not None:
            near = np.flatnonzero((np.abs(d) < EXACT_M) & (np.abs(xy) < self.osm[1]).all(1))
            if len(near):
                dist = self.coast.query(xy[near])[0]
                d[near] = np.where(shapely.contains_xy(self.osm[0], *xy[near].T), dist, -dist)
        return d

    def carve(self, xy, height, quay=None):
        """height of the land at east/north points, shaped as a game shapes
        a shore: under the water the bed eases down to DEPTH_M over SHELF_M;
        on land a bank eases it down to BANK_ABOVE_M over the water at the
        shore, about BANK_M wide, wider and narrower as it goes -- but none
        where quay(xy) (by a road: a quay stands). Water with no level (a
        pond too small to be a body) is land."""
        level = self._at(self.level, xy, -np.inf)
        d = self.signed(xy)
        body = np.isfinite(level)
        lv = np.where(body, level, 0.0)
        bed = lv - DEPTH_M * _ease(d / SHELF_M)
        h = np.where(body & (d > 0), np.minimum(height, bed), height)
        width = BANK_M * (1 + BANK_VARY * _wiggle(xy))
        top = lv + BANK_ABOVE_M
        bank = body & (d <= 0) & (height > top)
        if quay is not None and bank.any():
            bank[bank] &= ~quay(xy[bank])
        return np.where(bank, top + (height - top) * _ease(-d / width), h)

    def sand(self, xy, height):
        """How much of the shore's sand, 0-1, at east/north points that high:
        within SAND_M of the water, not over SAND_UP_M above it."""
        level = self._at(self.level, xy, -np.inf)
        d = self.signed(xy)
        return np.where(np.isfinite(level) & (d <= 0),
                        (1 - _ease(-d / SAND_M)) * (1 - _ease((height - level) / SAND_UP_M)), 0.0)

    def inside(self, xy):
        """Whether east/north points are in the water (of a body)."""
        return np.isfinite(self._at(self.level, xy, -np.inf)) & (self.signed(xy) > 0)

    def keep(self, xy, height):
        """Whether land at east/north points, that high, is seen: not
        deeper than SEE_M under the water."""
        return height >= self._at(self.level, xy, -np.inf) - SEE_M

    def shore_grid(self):
        """water.json's "shore": the distance from dry land, SHORE_CELL_M a
        cell, whole metres to SHORE_MAX_M."""
        k = int(round(SHORE_CELL_M / CELL_M))
        d = self.shore[k // 2::k, k // 2::k]
        return {"lo": self.lo, "cell": SHORE_CELL_M, "size": len(d),
                "metres": np.minimum(np.round(d), SHORE_MAX_M).astype(int).ravel().tolist()}


def _ease(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def _wiggle(xy):
    """Smooth noise, -1..1, over tens of metres: so a bank is not ruled."""
    x, y = xy[:, 0], xy[:, 1]
    return (np.sin(x / 23 + 1.3) * np.cos(y / 19 - 0.7) + 0.6 * np.sin((x + y) / 11.0)
            + 0.4 * np.cos((x - y) / 7.0 + 2.1)) / 2.0


def jrc(to_ll, height):
    """f(east/north (n, 2)) -> whether the JRC map (or the sea, by height)
    has water there: to_ll(east/north) -> (lat, lon), height the height
    map; the sea only if JRC cannot be had."""
    from streetview_to_3d.postprocess.terrain import SEA_M
    often = occurrence_map()

    def f(xy):
        lat, lon = to_ll(xy)
        sea = height(lat, lon) <= SEA_M
        try:
            return (often(lat, lon) >= WET) | sea
        except OSError:
            return sea
    return f


def outline(elements, to_xy, box_m, water_at, wet):
    """The water within box_m (a square half-width) of the centre, as one
    shapely geometry, all from OSM: its water outlines (osm.fetch's; to_xy:
    an OSM geometry to east/north metres) and the box's edge split the box
    into pieces, each all water or all land -- a shoreline bounds it. Only
    the outlines near us are had, never a lake's whole loop, so a line
    alone does not say which side is water; each piece is:

      - water if a point inside it lies in an OSM water area (water_at:
        east/north (n, 2) -> whether, osm.water_at, one request)
      - water if the coastline runs along it with it on its right: OSM
        draws the sea only as its coastline, the water always on the right
      - land otherwise; wet (east/north -> whether, the JRC map) decides a
        piece no shoreline bounds, the whole box one piece."""
    from streetview_to_3d.postprocess.osm import is_water
    inner = shapely.box(-box_m, -box_m, box_m, box_m)
    ways = [(e, to_xy(e["geometry"])) for e in elements if is_water(e) and len(e.get("geometry") or []) >= 2]
    lines = [inner.intersection(shapely.LineString(xy)) for _, xy in ways]
    faces = list(shapely.get_parts(shapely.polygonize([shapely.union_all(lines + [inner.exterior])])))
    if not faces:
        return shapely.Polygon()
    inside = np.array([f.representative_point().coords[0] for f in faces])
    if len(faces) == 1:
        return faces[0] if wet(inside).all() else shapely.Polygon()
    water = np.array(water_at(inside), bool)
    # the sea: a metre to the right of the coastline, all along it
    right, left = [], []
    for e, xy in ways:
        if e.get("tags", {}).get("natural") != "coastline":
            continue
        a, b = xy[:-1], xy[1:]
        d = b - a
        n = np.stack([d[:, 1], -d[:, 0]], 1) / np.maximum(np.linalg.norm(d, axis=1), 1e-9)[:, None]
        mid = (a + b) / 2
        right.append(mid + n)
        left.append(mid - n)
    if right:
        right, left = np.concatenate(right), np.concatenate(left)
        for k, f in enumerate(faces):
            water[k] |= shapely.contains_xy(f, *right.T).sum() > shapely.contains_xy(f, *left.T).sum()
    return shapely.union_all([f for f, w in zip(faces, water) if w]) if water.any() else shapely.Polygon()


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


def save(scene_dir, wet):
    with open(os.path.join(scene_dir, FILENAME), "w") as f:
        json.dump({"surfaces": wet.surfaces, "shore": wet.shore_grid()}, f, separators=(",", ":"))
    return FILENAME
