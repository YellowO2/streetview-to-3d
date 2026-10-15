"""OSM's roads, squares and tagged sidewalks laid out as one surface."""
import numpy as np

from streetview_to_3d.postprocess.world.roads.tools.paving import (
    KERB, PAINT, PAVEMENT, PAVING_JOINT, STRIP_M, lane_dividers, sidewalks,
)
from streetview_to_3d.postprocess.world.tools.osm_roads import KEPT, PATHS, colour, is_area, on_ground, ways
from streetview_to_3d.postprocess.world.tools.seams import ramp

DETAIL_M = 100.0          # sidewalks, kerbs, paint and crossings only on ways this near a camera
PROFILE_M, PROFILE_CELL_M = 12.0, 2.0     # road heights: ground smoothed over this, on this grid
SINK_M, SHOULDER_M = 0.25, 4.0            # land sunk this far under a road, easing back over this


class Network:
    """Kept roads (KEPT), squares and tagged sidewalks as one surface.

    Each spot belongs to one layer: paint, carriageway, pedestrian street,
    square, then pavement. shapes: {colour: surface}; strips and fills are
    where points are laid before being clipped to shapes. Sidewalks, kerbs,
    paint and crossings only within DETAIL_M of a camera of site (Site). heights() gives
    the roads' height field; adapt() fits the land to it."""

    def __init__(self, elements, site):
        to_xy = site.to_xy
        detail = lambda xy: site.from_cameras(xy).min() < DETAIL_M
        import shapely
        from collections import defaultdict
        layers = defaultdict(list)                  # priority -> [(shape, colour)]
        self.strips, self.fills = [], []
        self.elements, self.to_xy = elements, to_xy
        self.street_lines = [(xy, w, tags) for xy, w, _, tags in ways(elements, to_xy)
                             if tags["highway"] in KEPT and tags["highway"] not in PATHS]
        # junctions from shared OSM nodes: paint stops short of them
        from collections import Counter
        degree, node_xy = Counter(), {}
        for e in elements:
            tags = e.get("tags", {})
            if tags.get("highway") not in KEPT or not on_ground(tags) or is_area(e):
                continue
            nodes = e.get("nodes", [])
            coordinates = to_xy(e.get("geometry", []))
            for k, (node, coordinate) in enumerate(zip(nodes, coordinates)):
                degree[node] += 1 if k in (0, len(nodes) - 1) else 2
                node_xy[node] = coordinate
        junctions = shapely.union_all([shapely.Point(node_xy[node]).buffer(8)
                                      for node, n in degree.items() if n > 2])
        obstacles = shapely.union_all([shapely.make_valid(shapely.Polygon(to_xy(e["geometry"])))
                                      for e in elements if "building" in e.get("tags", {})
                                      and len(e.get("geometry", [])) >= 4])
        for xy, w, c, tags in ways(elements, to_xy):
            near = detail(xy)
            if tags["highway"] == "footway" and tags.get("footway") == "sidewalk":
                if not near:
                    continue
                shape = shapely.LineString(xy).buffer(w / 2, cap_style="flat", join_style="mitre")
                layers[3].append((shape, np.array(PAVEMENT)))
                self.strips.append((xy, w, np.array(PAVEMENT)))
                continue
            if tags["highway"] not in KEPT:
                continue
            if tags["highway"] not in PATHS and near:
                for shape, centre, pavement_width in sidewalks(xy, w, tags):
                    shape = shapely.difference(shape, obstacles)
                    layers[3].append((shape, np.array(PAVEMENT)))
                    for part in shapely.get_parts(centre):
                        if part.geom_type == "LineString":
                            self.strips.append((shapely.get_coordinates(part), pavement_width, np.array(PAVEMENT)))
                for mark in lane_dividers(xy, w, tags):
                    mark = shapely.difference(mark, junctions)
                    if mark.is_empty:
                        continue
                    layers[-1].append((mark, np.array(PAINT)))
                    for part in shapely.get_parts(mark):
                        if part.geom_type == "Polygon":
                            self.fills.append((shapely.get_coordinates(part.exterior), np.array(PAINT)))
            layers[1 if tags["highway"] in PATHS else 0].append(
                (shapely.LineString(xy).buffer(w / 2, quad_segs=2, cap_style="flat"), c))
            self.strips.append((xy, w, c))
        # marked crossing nodes become zebra bars aligned to their road
        for e in elements:
            tags = e.get("tags", {})
            if e.get("type") != "node" or tags.get("highway") != "crossing":
                continue
            if tags.get("crossing") in ("no", "unmarked") or tags.get("crossing:markings") == "no":
                continue
            if tags.get("crossing:markings") not in ("zebra", "yes") and tags.get("crossing") not in ("zebra", "marked", "traffic_signals"):
                continue
            location = to_xy([e])[0]
            if not self.street_lines or not detail(location[None]):
                continue
            xy, width, _ = min(self.street_lines, key=lambda r: shapely.LineString(r[0]).distance(shapely.Point(location)))
            line = shapely.LineString(xy)
            if line.distance(shapely.Point(location)) > width / 2 + 2:
                continue
            at = line.project(shapely.Point(location))
            a = np.array(line.interpolate(max(0, at - .5)).coords[0])
            b = np.array(line.interpolate(min(line.length, at + .5)).coords[0])
            along = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            side = np.array([-along[1], along[0]])
            centre = np.array(line.interpolate(at).coords[0])
            for offset in np.arange(-width / 2 + .5, width / 2 - .3, 1.1):
                q = np.array([centre + along * u + side * v for u, v in
                              ((-1.8, offset), (1.8, offset), (1.8, offset + .55), (-1.8, offset + .55))])
                layers[-2].append((shapely.Polygon(q), np.array(PAINT)))
                self.fills.append((q, np.array(PAINT)))
        for e in elements:
            tags = e.get("tags", {})
            if e["type"] == "way" and is_area(e) and tags.get("highway") in KEPT and on_ground(tags):
                xy = to_xy(e["geometry"])
                layers[2].append((shapely.make_valid(shapely.Polygon(xy)), colour(tags)))
                self.fills.append((xy, colour(tags)))
        self.shapes, taken = {}, shapely.Polygon()
        for p in sorted(layers):
            by = defaultdict(list)
            for g, c in layers[p]:
                by[tuple(c)].append(g)
            for c, gs in by.items():
                g = shapely.difference(shapely.union_all(gs), taken)
                self.shapes[c] = shapely.union_all([self.shapes[c], g]) if c in self.shapes else g
            taken = shapely.union_all([taken] + [g for g, _ in layers[p]])
        pavement = self.shapes.get(PAVEMENT, shapely.Polygon())
        if not pavement.is_empty:
            curb = shapely.difference(pavement, pavement.buffer(-.18, join_style="mitre"))
            self.shapes[PAVEMENT] = shapely.difference(pavement, curb)
            self.shapes[KERB] = curb
            # the same strips and fills serve both colours, each clipped to its own shape
            self.fills += [(xy, np.array(KERB)) for xy, c in list(self.fills) if tuple(c) == PAVEMENT]
            self.strips += [(xy, w, np.array(KERB)) for xy, w, c in list(self.strips) if tuple(c) == PAVEMENT]
        # paving joints as real polygons, so they survive triangle scattering
        paving = self.shapes.get(PAVEMENT, shapely.Polygon())
        if not paving.is_empty:
            lo_x, lo_y, hi_x, hi_y = paving.bounds
            seams = []
            for axis, lo, hi, other_lo, other_hi in ((0, lo_x, hi_x, lo_y, hi_y), (1, lo_y, hi_y, lo_x, hi_x)):
                for t in np.arange(np.floor(lo / 1.2) * 1.2, hi, 1.2):
                    coords = [(t, other_lo), (t, other_hi)] if axis == 0 else [(other_lo, t), (other_hi, t)]
                    seam = shapely.intersection(shapely.LineString(coords), paving)
                    if not seam.is_empty:
                        seams.append(seam.buffer(.035, cap_style="flat"))
            joints = shapely.intersection(shapely.union_all(seams), paving)
            self.shapes[PAVEMENT] = shapely.difference(paving, joints)
            self.shapes[PAVING_JOINT] = joints
            self.strips += [(xy, w, np.array(PAVING_JOINT)) for xy, w, c in list(self.strips) if tuple(c) == PAVEMENT]
            self.fills += [(xy, np.array(PAVING_JOINT)) for xy, c in list(self.fills) if tuple(c) == PAVEMENT]
        self.all = taken
        for g in self.shapes.values():
            shapely.prepare(g)
        shapely.prepare(self.all)

    def edges(self, step_m=STRIP_M):
        """(n, 2): points along the surface's outline, step_m apart."""
        import shapely
        return shapely.get_coordinates(shapely.segmentize(shapely.boundary(self.all), step_m)) \
            if not self.all.is_empty else np.zeros((0, 2))

    def heights(self, ground, lo, hi):
        """f(east/north (n, 2)) -> road height: ground(xy) under the roads smoothed over
        PROFILE_M, one field for all so junctions meet. Covers lo..hi."""
        import shapely
        from scipy.ndimage import gaussian_filter, map_coordinates
        c = PROFILE_CELL_M
        shape = tuple(np.ceil((np.asarray(hi) - lo) / c).astype(int) + 1)
        gx, gy = np.meshgrid(lo[0] + c * np.arange(shape[0]), lo[1] + c * np.arange(shape[1]), indexing="ij")
        cells = np.stack([gx.ravel(), gy.ravel()], 1)
        on = shapely.contains_xy(self.all.buffer(c), *cells.T) if not self.all.is_empty else np.zeros(len(cells), bool)
        total, count = np.zeros(len(cells)), on.astype(float)
        total[on] = ground(cells[on])
        sigma = PROFILE_M / c
        total = gaussian_filter(total.reshape(shape), sigma)
        count = gaussian_filter(count.reshape(shape), sigma)
        field = np.where(count > 1e-3, total / np.maximum(count, 1e-9), np.nan)
        value, known = np.nan_to_num(field), np.isfinite(field).astype(float)

        def f(xy):
            if not len(xy):
                return np.zeros(0)
            v = map_coordinates(value, ((xy - lo) / c).T, order=1, mode="nearest")
            w = map_coordinates(known, ((xy - lo) / c).T, order=1, mode="nearest")
            return np.where(w > 0.5, v, ground(xy))
        return f

    def adapt(self, xy, h, road_h):
        """Land heights h at xy fitted to the roads: SINK_M under them, easing back over SHOULDER_M."""
        import shapely
        from scipy.spatial import cKDTree
        edge = self.edges(1.0)
        if not len(edge):
            return h
        d = cKDTree(edge).query(xy, distance_upper_bound=SHOULDER_M, workers=-1)[0]
        d[shapely.contains_xy(self.all, *xy.T)] = 0.0
        near = np.flatnonzero(d < SHOULDER_M)
        if not len(near):
            return h
        w = 1 - ramp(d[near] / SHOULDER_M)
        h = h.copy()
        h[near] += (road_h(xy[near]) - SINK_M - h[near]) * w
        return h
