"""The water round the scene: flat bodies over the land, its bed.

From OSM's outlines (water_outline.py) or the JRC Global Surface Water map (WaterMap.py), each
at the height map's low percentile, never above the panos' ground near it.

Saved as water.json beside scene.json, east/north metres:
    {"surfaces": [{"level": m, "outer": [[e, n], ...], "holes": [[[e, n], ...], ...]}],
     "shore": {"lo": m, "cell": m, "size": n, "metres": [n x n, row by row north-wards]}}
shore being each cell's distance from dry land, up to SHORE_MAX_M.
"""

import numpy as np
import shapely

from streetview_to_3d.postprocess.world.tools.reach import OSM_M, RADIUS_M
from streetview_to_3d.postprocess.world.tools.seams import ramp

CELL_M = 5.0
MIN_M2 = 400.0            # smaller bodies are left to the land
UNDER_LAND_M = 20.0       # bodies grown this far under the shore
DEPTH_M, SHELF_M = 3.0, 12.0      # the bed eases down to DEPTH_M over SHELF_M from the shore
BANK_M, BANK_VARY = 6.0, 0.5      # bank width, varying by this share
BANK_ABOVE_M = 0.15               # bank height over the water at the shore
QUAY_M = 2.0                      # no bank this near a road (a quay)
SAND_M, SAND_UP_M = 2.5, 1.0      # sand within this of the shore, at most this far above the water
EXACT_M, TRACE_M = 30.0, 0.25     # within EXACT_M of the shore, distance measured to OSM's line sampled every TRACE_M
BED, BED_M = (0.72, 0.84, 0.90), 2.5  # bed colour (the pale sky it mirrors), fully at this depth
SHORE_CELL_M, SHORE_MAX_M = 10.0, 60
LOW_PCT = 5                       # a body's level: this percentile of the height map over it
CAP_M, CLEAR_M = 200.0, 0.5       # at least CLEAR_M under the ground of any pano within CAP_M


class Water:
    """The water within RADIUS_M on a CELL_M grid; surfaces are its bodies, each at its level.

    site: Site (its panos' ground heights); elevation: Elevation (its height map and datum
    shift); outline: OSM's water (water_outline), used within OSM_M, wet(xy) (WaterMap) past
    it or without one."""

    def __init__(self, site, elevation, outline, wet):
        radius_m, to_ll, height, shift = RADIUS_M, site.to_ll, elevation.heights, elevation.shift
        panos = (site.anchors, site.elevations)
        osm = (outline, OSM_M) if outline is not None else None
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
        # OSM's shoreline as points TRACE_M apart, the box's edge excluded
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
        """Signed metres from the shore (+ in the water): to OSM's line near it, else from the grid."""
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
        """Land height at east/north points shaped into a shore: the bed eases to DEPTH_M under the water,
        a varying BANK_M bank eases down to the water on land, except where quay(xy)."""
        level = self._at(self.level, xy, -np.inf)
        d = self.signed(xy)
        body = np.isfinite(level)
        lv = np.where(body, level, 0.0)
        bed = lv - DEPTH_M * ramp(d / SHELF_M)
        h = np.where(body & (d > 0), np.minimum(height, bed), height)
        width = BANK_M * (1 + BANK_VARY * _wiggle(xy))
        top = lv + BANK_ABOVE_M
        bank = body & (d <= 0) & (height > top)
        if quay is not None and bank.any():
            bank[bank] &= ~quay(xy[bank])
        return np.where(bank, top + (height - top) * ramp(-d / width), h)

    def sand(self, xy, height):
        """Sand share 0-1 at east/north points that high: within SAND_M of the water, under SAND_UP_M above it."""
        level = self._at(self.level, xy, -np.inf)
        d = self.signed(xy)
        return np.where(np.isfinite(level) & (d <= 0),
                        (1 - ramp(-d / SAND_M)) * (1 - ramp((height - level) / SAND_UP_M)), 0.0)

    def inside(self, xy):
        """Whether east/north points are in the water (of a body)."""
        return np.isfinite(self._at(self.level, xy, -np.inf)) & (self.signed(xy) > 0)

    def bed(self, xy, height):
        """Bed colour share 0-1 for land that high: 0 at the water's level, 1 at BED_M under it."""
        level = self._at(self.level, xy, -np.inf)
        return np.where(np.isfinite(level), ramp((level - height) / BED_M), 0.0)

    def shore_grid(self):
        """water.json's "shore": whole metres from dry land per SHORE_CELL_M cell, up to SHORE_MAX_M."""
        k = int(round(SHORE_CELL_M / CELL_M))
        d = self.shore[k // 2::k, k // 2::k]
        return {"lo": self.lo, "cell": SHORE_CELL_M, "size": len(d),
                "metres": np.minimum(np.round(d), SHORE_MAX_M).astype(int).ravel().tolist()}


def _wiggle(xy):
    """Smooth noise in -1..1 over tens of metres."""
    x, y = xy[:, 0], xy[:, 1]
    return (np.sin(x / 23 + 1.3) * np.cos(y / 19 - 0.7) + 0.6 * np.sin((x + y) / 11.0)
            + 0.4 * np.cos((x - y) / 7.0 + 2.1)) / 2.0


def _bodies(mask, lo):
    """The wet cells of mask (CELL_M grid from lo) as one polygon per body, small ones dropped."""
    from shapely.geometry import box
    boxes = []
    for r, row in enumerate(mask):               # one box per run of wet cells
        edges = np.flatnonzero(np.diff(np.r_[0, row.astype(int), 0]))
        y = lo + r * CELL_M
        boxes += [box(lo + a * CELL_M, y, lo + b * CELL_M, y + CELL_M)
                  for a, b in zip(edges[::2], edges[1::2])]
    if not boxes:
        return []
    merged = shapely.union_all(boxes).simplify(CELL_M / 2)
    return [p for p in getattr(merged, "geoms", [merged]) if p.area >= MIN_M2]
