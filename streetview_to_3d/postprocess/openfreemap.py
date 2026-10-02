"""OpenStreetMap's buildings, roads and water from OpenFreeMap's vector
tiles, for when Overpass gives no answer (osm.fetch): a CDN's files, no
key, no queries -- they are there when Overpass's servers are busy.

The tiles (the OpenMapTiles schema, ZOOM's the finest) hold every
building's outline and height, every road and the water, but few tags:
no building kinds, roofs or levels (styles.py's regional defaults take
over), roads only by class, no street objects. They are given in
Overpass's shape -- ways with tags and geometry -- so what reads osm.fetch
reads them unchanged:

  - a building: building=yes, its height (render_height) only where
    OpenMapTiles did not guess it (DEFAULT_HEIGHT_M), its min_height
  - a road: highway from its class (subclass for paths, _link for a ramp),
    bridge/tunnel, layer; railways, ferries and the like left out
  - water: an area's outline natural=water, each island's untagged (a
    water relation's members, as water.outline reads them)

and which side of a shore is water (water_at: osm.water_at's), from the
tiles' areas whole -- the sea's too.

A tile holds what crosses it, cut at its edge plus a buffer: each piece
is cut to the tile, and the two halves of a building split by a tile's
edge joined again (two tiles' pieces sharing a stretch of that edge);
water joined whole. Credit "© OpenMapTiles © OpenStreetMap contributors".
"""
import json
import math
import urllib.request

import numpy as np
import shapely

TILEJSON = "https://tiles.openfreemap.org/planet"
ZOOM = 14                 # the schema's finest: every building
HEADERS = {"User-Agent": "streetview-to-3d (https://github.com/YellowO2/streetview-to-3d)"}
DEFAULT_HEIGHT_M = 5      # OpenMapTiles' render_height for a building with no height or levels
ROADS = {"motorway": "motorway", "trunk": "trunk", "primary": "primary", "secondary": "secondary",
         "tertiary": "tertiary", "minor": "residential", "service": "service", "track": "track",
         "path": "path", "busway": "busway"}
PATHS = {"footway", "cycleway", "steps", "path", "pedestrian", "bridleway"}
MAJOR = {"motorway", "trunk", "primary", "secondary", "tertiary"}
WATER = {"lake", "river", "ocean", "pond", "dock"}


def tile_url():
    """The current tiles' URL template ({z}/{x}/{y}), from the TileJSON."""
    req = urllib.request.Request(TILEJSON, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)["tiles"][0]


def _tile(lat, lon):
    """The ZOOM tile (x, y) holding (lat, lon), fractional."""
    n = 2 ** ZOOM
    return (np.asarray(lon) + 180) / 360 * n, (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n


def _ll(xy):
    """Whole-map tile coordinates (n, 2) -> [{lat, lon}], Overpass's geometry."""
    n, xy = 2 ** ZOOM, np.asarray(xy, float)
    lon = xy[:, 0] / n * 360 - 180
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * xy[:, 1] / n))))
    return [{"lat": float(a), "lon": float(o)} for a, o in zip(lat, lon)]


def decode(data, x, y):
    """A tile's bytes -> {layer: [(properties, geometry)]}, each geometry
    in whole-map tile coordinates and cut to the tile."""
    import mapbox_vector_tile
    out = {}
    cell = shapely.box(x, y, x + 1, y + 1)
    for name, layer in mapbox_vector_tile.decode(data, default_options={"y_coord_down": True}).items():
        ext = layer.get("extent", 4096)
        for f in layer["features"]:
            g = shapely.geometry.shape(f["geometry"])
            g = shapely.transform(g, lambda c: c / ext + (x, y))
            g = shapely.make_valid(g).intersection(cell) if g.geom_type.endswith("Polygon") else g.intersection(cell)
            if not g.is_empty:
                out.setdefault(name, []).append((f["properties"], g))
    return out


def tiles(lat0, lon0, reach_m, m_per_lat, m_per_lon):
    """{(x, y): decoded tile} for every ZOOM tile within reach_m of (lat0, lon0)."""
    x0, y0 = _tile(lat0 + reach_m / m_per_lat, lon0 - reach_m / m_per_lon)
    x1, y1 = _tile(lat0 - reach_m / m_per_lat, lon0 + reach_m / m_per_lon)
    return fetch([(x, y) for x in range(int(x0), int(x1) + 1) for y in range(int(y0), int(y1) + 1)])


_decoded = {}   # (url, x, y) -> decoded tile: the buildings' and the water's asked of the same (~1 s each)


def fetch(cells, url=None, download=None):
    """{(x, y): decoded tile} for each ZOOM tile of cells; download(url) ->
    its bytes (terrain.TileMap's, kept on disk: a rebuild asks for none)."""
    from concurrent.futures import ThreadPoolExecutor
    url = url or tile_url()
    if download is None:
        from streetview_to_3d.postprocess.terrain import TileMap
        download = TileMap(url, ZOOM, None, headers=HEADERS)._download
    new = [c for c in cells if (url, *c) not in _decoded]
    with ThreadPoolExecutor(8) as pool:
        datas = list(pool.map(lambda c: download(url.format(z=ZOOM, x=c[0], y=c[1])), new))
    if len(_decoded) > 64:
        _decoded.clear()
    for c, d in zip(new, datas):
        _decoded[(url, *c)] = decode(d, *c) if d else {}
    return {c: _decoded[(url, *c)] for c in cells}


def _joined(pieces):
    """[(cell, properties, polygon)] -> [(properties, polygon)]: the pieces
    of one building, split by a tile's edge, joined -- two pieces of
    neighbouring tiles sharing a stretch of it."""
    parent = list(range(len(pieces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    tree = shapely.STRtree([p for _, _, p in pieces])
    for i, (cell, _, p) in enumerate(pieces):
        for j in tree.query(p):
            if j > i and pieces[j][0] != cell and p.intersection(pieces[j][2]).length > 1e-7:
                parent[find(j)] = find(i)
    groups = {}
    for i in range(len(pieces)):
        groups.setdefault(find(i), []).append(i)
    return [(pieces[g[0]][1], shapely.union_all([pieces[i][2] for i in g])) for g in groups.values()]


def _polygons(g):
    return [p for p in shapely.get_parts(g) if p.geom_type == "Polygon" and not p.is_empty]


def elements(lat0, lon0, buildings_m, roads_m, m_per_lat, m_per_lon, water_m=0, full_m=250.0,
             far_building_m=80.0, far_roads=MAJOR, decoded=None, cams=None):
    """osm.fetch's elements, from the tiles: every building and road within
    full_m of the cameras' box (cams: their (lats, lons); else the centre);
    past it the buildings far_building_m round and more within buildings_m
    and the roads of far_roads within roads_m; the water within water_m.
    decoded: tiles()'s, or fetched."""
    if decoded is None:
        decoded = tiles(lat0, lon0, max(buildings_m, roads_m, water_m), m_per_lat, m_per_lon)
    cx, cy = _tile(lat0, lon0)
    centre = (cx, cy, cx, cy)
    if cams is not None and len(cams[0]):
        kx, ky = _tile(np.asarray(cams[0]), np.asarray(cams[1]))
        cams = (kx.min(), ky.min(), kx.max(), ky.max())
    else:
        cams = centre
    m_per_tile = 2 * math.pi * 6378137 * math.cos(math.radians(lat0)) / 2 ** ZOOM

    def reach(g, box=centre):
        """How far, east or north, g's nearest lies from box (metres)."""
        x0, y0, x1, y1 = g.bounds
        dx = max(x0 - box[2], box[0] - x1, 0) * m_per_tile
        dy = max(y0 - box[3], box[1] - y1, 0) * m_per_tile
        return max(dx, dy)

    out, ids = [], iter(range(-1, -10 ** 9, -1))
    way = lambda tags, xy: {"type": "way", "id": next(ids), "tags": tags, "geometry": _ll(xy)}

    pieces = [(c, p, poly) for c, t in decoded.items() for p, g in t.get("building", [])
              for poly in _polygons(g)]
    for p, g in _joined(pieces):
        for poly in _polygons(g):
            if reach(poly) > buildings_m or (reach(poly, cams) > full_m
                                             and poly.exterior.length * m_per_tile < far_building_m):
                continue
            tags = {"building": "yes"}
            h, base = p.get("render_height"), p.get("render_min_height") or 0
            if h is not None and (h != DEFAULT_HEIGHT_M or base):
                tags["height"] = str(h)
            if base:
                tags["min_height"] = str(base)
            out.append(way(tags, poly.exterior.coords))

    for c, t in decoded.items():
        for p, g in t.get("transportation", []):
            cls = p.get("class")
            if cls not in ROADS:
                continue
            highway = p.get("subclass") if cls == "path" and p.get("subclass") in PATHS else ROADS[cls]
            if cls in MAJOR and p.get("ramp"):
                highway += "_link"
            for line in shapely.get_parts(shapely.line_merge(g) if g.geom_type == "MultiLineString" else g):
                if line.geom_type != "LineString" or len(line.coords) < 2:
                    continue
                if reach(line) > roads_m or (reach(line, cams) > full_m and cls not in far_roads):
                    continue
                tags = {"highway": highway}
                if p.get("brunnel") in ("bridge", "tunnel"):
                    tags[p["brunnel"]] = "yes"
                if p.get("layer"):
                    tags["layer"] = str(p["layer"])
                out.append(way(tags, line.coords))

    if water_m:
        for poly in _polygons(_water(decoded)):
            if reach(poly) > water_m:
                continue
            out.append(way({"natural": "water"}, poly.exterior.coords))
            out += [way({}, r.coords) for r in poly.interiors]
    return out


def _water(decoded):
    """The tiles' water areas (WATER's, not the intermittent), joined."""
    wet = [g for t in decoded.values() for p, g in t.get("water", [])
           if p.get("class") in WATER and not p.get("intermittent")]
    return shapely.union_all(wet) if wet else shapely.Polygon()


def water_at(latlon, decoded=None):
    """Whether each (lat, lon) lies in the tiles' water; decoded: the
    tiles holding them (fetched if not given)."""
    latlon = np.asarray(latlon, float).reshape(-1, 2)
    if not len(latlon):
        return []
    x, y = _tile(latlon[:, 0], latlon[:, 1])
    if decoded is None:
        decoded = fetch(sorted({(int(a), int(b)) for a, b in zip(x, y)}))
    return shapely.contains_xy(_water(decoded), x, y).tolist()
