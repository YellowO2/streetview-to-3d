"""The land, buildings and roads around the scene, from free global maps.

DA3 reaches a few tens of metres; hills and mountains
further out come from AWS Terrain Tiles (terrarium PNGs, no key, ~30 m
data, mostly SRTM -- the ground, big buildings at most a blur), coloured
from Google's satellite tiles (GOOGLE_URL, the panos' own source: ~0.5 m
at its finest), each corner of the land read at the zoom whose pixel is as
wide as its corners are apart (google_colours: its lower zooms are the
finer ones averaged, so a far corner is its patch's colour, not a car's),
roofs at ROOF_ZOOM; where Google has none, EOX's Sentinel-2 cloudless
mosaic (no key, 10 m, CC BY-NC-SA: credit "Sentinel-2 cloudless by EOX",
not for sale) -- at 10 m a house is a pixel or two, and Matsumoto
Castle's roof came out the green of the trees round it. The land is a surface, as
a game has it: triangles, fine near the scene and coarser with distance;
roads and buildings on it are points, spaced the same way:

1. the land's corners out to RADIUS_M (past it the viewer's haze has
   hidden everything), further apart the further from the nearest
   camera (LAND_EVERY x gap_at), joined into triangles;
   where the scene has its own ground, just beneath it (UNDER_M) -- one
   shared ground, the scene's no longer seen through, the land never over
   it -- and around it meeting its ground and taking its colour at its
   edge (seams.py)
2. height read off the tiles, the sea (the tiles also carry the sea bed)
   laid flat at sea level; the whole map shifted onto Google's datum (the
   median of its panos' elevation minus the map's), then bent near the
   panos onto each one's own elevation (correction). 30 m SRTM is the top
   of whatever stands there, trees included, and old: on NTU it was off by
   -4..+2.5 m along the route, where Google's matched DA3's own slope
3. coloured from the satellite, lifted a little (it is dark from above),
   with a light slope shading so relief reads; plain if the imagery
   cannot be had
4. OpenStreetMap's buildings (buildings.py) and roads (roads.py) on it
   (osm.py); the roads, like the ground, not where the scene is; the
   buildings fitted onto DA3's walls, what DA3 has of them left to it; a
   landmark mapped in parts (a spire, a dome) as its parts, each roof as
   OSM shapes it (roofs.py) and coloured as the satellite sees it.
   Roads and bridges are points within NEAR_M of a camera, and stored
   as triangles further off (drawn as points all the same); a building
   DA3 reaches (buildings.reached) points, to meet DA3's own, every
   other stored solid, as triangles (the viewer builds it of brush
   strokes: effects/blocks.js)
5. near the cameras (PAINT_M), all of it -- land, roads, walls -- coloured
   from the scene's own panos as the fill colours its ground (_paint), so
   it matches DA3 where they meet; the maps' colours are only for what no
   pano sees -- and for buildings DA3 never reaches (buildings.reached):
   nothing of DA3's for them to meet
6. water (water.py), each body at its level, drawn by the viewer as
   points mirroring the world over the land; the land goes on under it as its bed,
   carved down from the shore and ever more the sky's pale blue the deeper, the shore
   wherever it crosses the water

Written to land.ply beside scene.json (its "land", triangles), the roads
near the scene and bridges to terrain.ply (its "terrain"), the roads
further out to roads.ply (its "roads", triangles), the buildings DA3
reaches, spaced the same way, to buildings.ply (its "buildings"), the
rest solid to blocks.ply (its "blocks"), all already in the world frame, and the water to water.json (its "water").
The land is drawn as a surface, the ground everything stands on, painted
in patches as dabs of paint (effects/land.js); the solid buildings built
of brush strokes (effects/blocks.js); all else as points: the viewer
scatters points over the far roads' triangles as it loads them
(effects/scatter.js), each triangle's corners saying how far apart; it draws every point as big as it is spaced (the ply's "gap":
point_gap; scene-store.js, terrainBands).

    python -m streetview_to_3d.postprocess.terrain SCENE_DIR
"""
import io
import math
import os
import sys
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from streetview_to_3d import scene as scene_mod
from streetview_to_3d.paths import DATA_DIR
from streetview_to_3d.postprocess import buildings, osm, roads, seams, water
from streetview_to_3d.postprocess.ply_io import write_mesh, write_ply

FILENAME = "terrain.ply"
ROADS_FILENAME = "roads.ply"
LAND_FILENAME = "land.ply"
BUILDINGS_FILENAME = "buildings.ply"
BLOCKS_FILENAME = "blocks.ply"
HEIGHT_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
HEIGHT_ZOOM = 13          # ~19 m a pixel at the equator, finer than the data
COLOUR_URL = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2025_3857/default/g/{z}/{y}/{x}.jpg"
COLOUR_ZOOM = 14          # ~10 m a pixel, the imagery's own
GOOGLE_URL = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
GOOGLE_ZOOMS = (11, 18)   # the land read between these: ~75 m a pixel to ~0.6 m (at the equator)
ROOF_ZOOM = 18
EQUATOR_M = 156543.03     # a zoom 0 pixel's width at the equator
ROOF_INSET_M, ROOF_STEP_M = 1.0, 1.0   # a roof sampled this far in from its edge, this far apart
TILE_THREADS = 8          # tiles fetched at once, ahead of reading them (TileMap.fetch)
RADIUS_M = 1000.0                          # the land's reach: the viewer's haze is whole by then
OSM_M, ROADS_M = RADIUS_M, 700.0           # OSM's reach: its water the land's, its roads only where a point wide (~600 m)
BUILDINGS_M = 800.0                        # buildings this far: past it the haze has all but hidden them
NEAR_M = 50.0                              # roads, bridges: points this near a camera, triangles past
PAINT_M = 30.0                            # map points this near a camera are coloured from the panos
TINT_M, MEET_M = 8.0, 10.0                # the ground's seam with the scene (seams.py): bands
BRIDGE_CLEAR_M = {"road": 4.5, "water": 2.5}     # a bridge's deck at least this over each (the land fits under it)
ROAD_MEET_M, ROAD_MEET_MAX_M = 40.0, 10.0  # roads, bridges: the scene's road's height where they touch it, their
                                           # own this far out (unless 10 m apart: not the same road)
UNDER_M = 0.1                             # the land under the scene: this far beneath its lowest points
LOW_M, COVER = 1.0, 0.75                  # the scene's ground: points this near the map's; land it has within
                                          # this much of the land's own gap is the scene's
TINT = 0.8                                # how far the map takes the scene's colour at its edge
GAP0_M, GAP_PER = 0.05, 0.018              # the land's corners: LAND_EVERY x gap_at apart
POINT_M = 0.10                             # roads' and buildings' points: as DA3's are drawn at the scene's edge,
RATE0, RATE, RAMP_M = 0.005, 0.018, 100.0  # coarse spacing beyond the detailed neighbourhood
DETAIL_RATE, DETAIL_END_M, DETAIL_BLEND_M = .007, 250.0, 150.0
BLOCK_RATE = 0.005                         # a solid building's strokes at most this of its distance apart: as fine on screen far off
SAND, SAND_MIX = (0.76, 0.70, 0.55), 0.7     # the shore's sand (water.sand), how far it covers the satellite's
LAND_EVERY = 2            # the land's triangles this many times the points' spacing: a surface has no gaps
M_PER_LAT = 111320.0
PLAIN = np.array([0.50, 0.55, 0.45])
LIFT = 0.75               # colour ** LIFT: brighter shadows, same hues
SEA_M = 0.5               # map height at or under this is sea
SUN = np.array([-0.5, -0.7, 0.5])        # world frame: x east, y down, z north
BEND_K, BEND_SOFT_M = 8, 10.0             # panos each bend is spread from; softening near one
BEND_FROM_M, BEND_TO_M = 30.0, 150.0      # the bend fades out between these from the nearest pano


TILES_DIR = os.path.join(DATA_DIR, "tiles")     # every tile ever downloaded, kept: a rebuild asks for none
TILE_TRIES = 3


class TileMap:
    """A web-mercator tile map read at any (lat, lon), bilinear; each tile
    downloaded once, ever (TILES_DIR), tried TILE_TRIES times. decode turns
    a tile's pixels (0-255, in mode) into its values; a map that leaves out
    empty tiles (a 404) gives missing there."""

    def __init__(self, url, zoom, decode, mode="RGB", missing=None, headers=None):
        self.url, self.zoom, self.decode, self.tiles = url, zoom, decode, {}
        self.mode, self.missing = mode, missing
        self.headers = headers or {"User-Agent": "streetview-to-3d"}

    def _download(self, url):
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
            data = self._download(self.url.format(z=self.zoom, x=x, y=y))
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
        """Download (TILE_THREADS at once) every tile within near pixels of
        any of (lat, lon), so reading them later asks for none."""
        from concurrent.futures import ThreadPoolExecutor
        x, y = self._xy(np.asarray(lat), np.asarray(lon))
        tiles = {(int(a // 256), int(b // 256)) for dx in (-near, near) for dy in (-near, near)
                 for a, b in zip(x + dx, y + dy)}
        with ThreadPoolExecutor(TILE_THREADS) as pool:
            list(pool.map(lambda t: self._download(self.url.format(z=self.zoom, x=t[0], y=t[1])), tiles))
        return len(tiles)

    def __call__(self, lat, lon):
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


def height_map():
    """Metres above sea level."""
    return TileMap(HEIGHT_URL, HEIGHT_ZOOM, lambda c: c[..., 0] * 256 + c[..., 1] + c[..., 2] / 256 - 32768)


def colour_map():
    """RGB, 0-1."""
    return TileMap(COLOUR_URL, COLOUR_ZOOM, lambda c: c / 255)


def google_map(zoom):
    """RGB, 0-1, NaN where Google has no tile."""
    from streetview_to_3d.services.http_headers import BROWSER_HEADERS
    return TileMap(GOOGLE_URL, zoom, lambda c: c / 255, missing=np.nan, headers=BROWSER_HEADERS)


def google_colours(lat, lon, spacing):
    """(n, 3) RGB at each (lat, lon), read at the zoom (GOOGLE_ZOOMS) whose
    pixel is about spacing (n,) metres wide; NaN where Google has none."""
    zoom = np.clip(np.floor(np.log2(EQUATOR_M * np.cos(np.radians(lat)) / np.maximum(spacing, 1e-3))),
                   *GOOGLE_ZOOMS).astype(int)
    out = np.full((len(lat), 3), np.nan)
    for z in np.unique(zoom):
        at = zoom == z
        tiles = google_map(int(z))
        tiles.fetch(lat[at], lon[at])
        out[at] = tiles(lat[at], lon[at])
    return out


def gap_at(cam_d):
    """How far apart map points are cam_d metres from the nearest camera:
    GAP0_M there (DA3's own are ~4 cm apart), GAP_PER of the distance more
    further out -- one even growth from the scene outwards. Measured from
    the centre with a 50 cm floor instead, the land was coarse right at the
    scene's edge."""
    return GAP0_M + GAP_PER * np.asarray(cam_d)


def point_gap(edge_d):
    """Saved spacing from the reconstructed edge: 0.8 m at 100 m, 1.5 m
    at 200 m, easing into the coarser outer map between 250 and 400 m.
    The viewer draws each point as big as this spacing.
    """
    e = np.maximum(np.asarray(edge_d, float), 0)
    near = POINT_M + RATE0 * e + (RATE - RATE0) * e ** 2 / (2 * RAMP_M)
    far = POINT_M + RATE0 * RAMP_M + (RATE - RATE0) * RAMP_M / 2 + RATE * (e - RAMP_M)
    coarse = np.where(e < RAMP_M, near, far)
    t = np.clip((e - DETAIL_END_M) / DETAIL_BLEND_M, 0, 1)
    blend = t * t * (3 - 2 * t)
    return (POINT_M + DETAIL_RATE * e) * (1 - blend) + coarse * blend


def sample_points(radius_m, cams, every=1):
    """(east, north) within radius_m of the centre, every x gap_at their
    distance to the nearest camera (cams, (n, 2)) apart: nested grids, each
    twice the last's spacing, each laid only where the gap wanted is
    between its spacing and twice that -- and shaken a little, so no grid
    shows."""
    from scipy.spatial import cKDTree
    tree, rng, out = cKDTree(cams), np.random.default_rng(0), []
    lo_c, hi_c = cams.min(0), cams.max(0)
    s = GAP0_M * every
    while (s / every - GAP0_M) / GAP_PER < 2 * radius_m:
        d_lo, d_hi = (s / every - GAP0_M) / GAP_PER, (2 * s / every - GAP0_M) / GAP_PER
        lo, hi = np.maximum(lo_c - d_hi, -radius_m), np.minimum(hi_c + d_hi, radius_m)
        if (lo < hi).all():
            gx, gy = np.meshgrid(np.arange(lo[0], hi[0], s), np.arange(lo[1], hi[1], s))
            grid = np.stack([gx.ravel(), gy.ravel()], 1) + rng.uniform(-0.3 * s, 0.3 * s, (gx.size, 2))
            d = tree.query(grid)[0]
            out.append(grid[(d >= d_lo) & (d < d_hi) & (np.linalg.norm(grid, axis=1) < radius_m)])
        s *= 2
    return np.concatenate(out) if out else np.zeros((0, 2))


def correction(anchors, fixes):
    """f(east/north (n, 2)) -> metres to add to the map there: near the
    anchors (the panos, whose Google elevation is right -- it matched DA3's
    own slope to 0.1 m where SRTM was twice as steep, NTU), their fixes
    (Google minus the map) spread by inverse distance; faded out between
    BEND_FROM_M and BEND_TO_M from the nearest one, where the map stands."""
    from scipy.spatial import cKDTree
    if not len(anchors):
        return lambda xy: np.zeros(len(xy))
    tree = cKDTree(anchors)

    def f(xy):
        k = min(BEND_K, len(anchors))
        d, i = tree.query(xy, k=k)
        d, i = d.reshape(len(xy), k), i.reshape(len(xy), k)
        w = 1 / (d ** 2 + BEND_SOFT_M ** 2)
        spread = (w * fixes[i]).sum(1) / w.sum(1)
        t = np.clip((d[:, 0] - BEND_FROM_M) / (BEND_TO_M - BEND_FROM_M), 0, 1)
        return spread * (1 - t * t * (3 - 2 * t))
    return f


def scene_points(sc, scene_dir, every=4):
    """(points, colours): every node's cloud, placed in the world (every
    n-th point)."""
    from streetview_to_3d.postprocess.ply_io import read_ply
    pts, cols = [np.zeros((0, 3))], [np.zeros((0, 3))]
    for nd in sc.nodes:
        if nd.ply and nd.transform:
            p, c = read_ply(os.path.join(scene_dir, nd.ply))
            T = np.asarray(nd.transform, float)
            pts.append(p[::every] @ T[:3, :3].T + T[:3, 3])
            cols.append((c if c is not None else np.full((len(p), 3), 0.5))[::every])
    return np.concatenate(pts), np.concatenate(cols)


def build(scene_dir, log=print):
    """Write scene_dir/FILENAME around the placed (and filled) scene's
    cameras."""
    from scipy.spatial import cKDTree
    sc = scene_mod.Scene.load(scene_dir)
    lat0, lon0 = sc.origin
    m_per_lon = M_PER_LAT * math.cos(math.radians(lat0))
    to_ll = lambda xy: (lat0 + xy[:, 1] / M_PER_LAT, lon0 + xy[:, 0] / m_per_lon)
    to_xy = lambda geometry: np.array([((p["lon"] - lon0) * m_per_lon, (p["lat"] - lat0) * M_PER_LAT)
                                       for p in geometry])
    cams = [n for n in sc.nodes if n.transform and n.position is not None]
    if not cams:
        log("terrain: no placed cameras")
        return
    cam_xz = np.array([(np.asarray(n.transform)[:3, :3] @ n.position + np.asarray(n.transform)[:3, 3])[[0, 2]]
                       for n in cams])

    # the ground: the map, one shift to Google's datum, then bent onto every
    # pano's own elevation near them
    heights = height_map()
    known = [n for n in sc.nodes if n.pano.elevation is not None]
    anchors = np.array([((n.pano.lon - lon0) * m_per_lon, (n.pano.lat - lat0) * M_PER_LAT) for n in known])
    under = heights(*to_ll(anchors)) if len(known) else np.zeros(0)
    fixes = np.array([n.pano.elevation for n in known]) - under
    shift = float(np.median(fixes)) if len(known) else 0.0
    bend = correction(anchors, fixes - shift)

    def ground(xy, raw=None):
        raw = heights(*to_ll(xy)) if raw is None else raw
        return np.where(raw <= SEA_M, 0.0, raw) + shift + bend(xy)

    # OpenStreetMap's buildings, roads and water, in one request
    try:
        elements = osm.fetch(lat0, lon0, OSM_M, ROADS_M, M_PER_LAT, m_per_lon, scene_dir,
                             water_m=OSM_M)
    except (OSError, ValueError) as e:    # the land stands without them
        log(f"terrain: no OpenStreetMap ({e!r})")
        elements = []

    # water: each body at its level over the land, which goes on under it (water.py);
    # its outline OSM's where OSM has one
    radius = RADIUS_M
    jrc = water.jrc(to_ll, heights)
    wet = water.Water(radius, to_ll, heights, shift, (anchors, np.array([n.pano.elevation for n in known])), jrc,
                      osm=(water.outline(elements, to_xy, OSM_M,
                                         lambda xy: osm.water_at(np.stack(to_ll(xy), 1), scene_dir), jrc),
                           OSM_M) if elements else None)

    # the scene always wins: the map only around it, faded in at its edge
    scene, scene_cols = scene_points(sc, scene_dir)
    foot = seams.Footprint(scene, scene_cols) if len(scene) else None
    near = (lambda xy: foot.at(xy)) if foot else \
        (lambda xy: (np.full(len(xy), np.inf), np.full(len(xy), np.nan), np.full((len(xy), 3), np.nan)))
    cam_tree = cKDTree(cam_xz)
    # points spaced from the scene's edge: its 1 m squares holding a few points
    cell, n = np.unique(np.floor(scene[:, [0, 2]]), axis=0, return_counts=True) if len(scene) else (cam_xz, None)
    edge_tree = cKDTree(cell[n >= 3] + 0.5 if n is not None and (n >= 3).any() else cam_xz)
    gap = lambda xy: point_gap(edge_tree.query(xy)[0])
    # the roads a game map keeps, one surface; they decide their own height
    # (the ground smoothed) and the land fits itself to them (roads.py)
    net = roads.Network(elements, to_xy)
    # ... and the scene's own road, where they touch it, its height (DA3's
    # ground there), easing back to theirs over ROAD_MEET_M: aligned where they meet
    road_raw = net.heights(ground, (-ROADS_M - 50, -ROADS_M - 50), (ROADS_M + 50, ROADS_M + 50))

    def to_scene(xy, h):
        d, g, _ = near(xy)
        return seams.meet(h, g, d, ROAD_MEET_M, ROAD_MEET_MAX_M)
    # OpenStreetMap's buildings, onto the scene's walls first: the land fits itself to them
    outlines = [o for o in buildings.outlines(elements, to_xy)
                if np.linalg.norm(o[0], axis=1).min() < BUILDINGS_M]
    n_fitted = n_trimmed = 0
    if outlines:
        from streetview_to_3d.postprocess.ground import normals_from_neighbours
        scene_normals = normals_from_neighbours(scene) if len(scene) >= 12 else np.zeros_like(scene)
        outlines, n_fitted, n_trimmed = buildings.fit_to_scene(outlines, scene, scene_normals, ground,
                                                               roads.coverage(roads.lines(elements, to_xy)))
    # the land: a surface, triangles between points LAND_EVERY times the gap apart,
    # and corners along the roads' edges, so a triangle never spans one
    en = sample_points(radius, cam_xz, LAND_EVERY)
    # bridges whole, their decks clear of the roads and water they cross; the land fits under them
    import shapely

    def crossed(road_h):
        def low(xy):
            out = np.full(len(xy), -np.inf)
            road = shapely.contains_xy(net.all, *xy.T) if not net.all.is_empty else np.zeros(len(xy), bool)
            out = np.where(road, road_h(xy) + BRIDGE_CLEAR_M["road"], out)
            level = wet._at(wet.level, xy, -np.inf)
            return np.where(wet.inside(xy), np.maximum(out, level + BRIDGE_CLEAR_M["water"]), out)
        return low
    # the scene's road is the roads its panos stand on (on a bridge over one,
    # or under it, by their height) and those joining them near it: only
    # these meet it; a road passing by, over or under keeps its own height
    found = roads.bridges(elements, to_xy)
    ids = roads.standing(elements, to_xy, cam_xz, ground(cam_xz),
                         roads.decks(found, road_raw, crossed(road_raw)))
    ids = roads.joined(elements, to_xy, ids, lambda xy: near(xy)[0] < ROAD_MEET_M)
    met = roads.region(elements, to_xy, ids, roads.SHOULDER_M)

    def road_h(xy):
        h = road_raw(xy)
        on = shapely.contains_xy(met, *xy.T) if len(xy) else np.zeros(0, bool)
        if on.any():
            h[on] = to_scene(xy[on], h[on])
        return h
    over = roads.decks(found, road_h, crossed(road_h), to_scene, ids)
    edges = np.concatenate([net.edges(), roads.deck_edges(over)])
    walls, owner = buildings.corners(outlines, lambda xy: LAND_EVERY * gap_at(cam_tree.query(xy)[0]))
    inside = np.linalg.norm(walls, axis=1) < radius
    edges = edges[np.linalg.norm(edges, axis=1) < radius]
    owner = np.r_[np.full(len(en) + len(edges), -1), owner[inside]]
    en = np.concatenate([en, edges, walls[inside]])
    # the scene's ground-level points: the land fills exactly where they are not
    low = scene[-scene[:, 1] < ground(scene[:, [0, 2]]) + LOW_M] if len(scene) else scene
    low_tree = cKDTree(low[:, [0, 2]]) if len(low) else None
    uncovered = (lambda xy, gap: low_tree.query(xy)[0] > COVER * gap) if low_tree else \
        (lambda xy, gap: np.ones(len(xy), bool))
    under = ~uncovered(en, LAND_EVERY * gap_at(cam_tree.query(en)[0]))
    dist, edge_h, edge_c = near(en)
    lat, lon = to_ll(en)
    raw = heights(lat, lon)
    h = seams.meet(ground(en, raw), edge_h, dist, MEET_M)
    h = roads.under_decks(over, en, net.adapt(en, h, road_h))
    # under the scene's own ground too, just beneath it: one shared ground,
    # so the scene's is not seen through, the land never over it
    h = np.where(under & np.isfinite(edge_h), np.minimum(h, np.nan_to_num(edge_h) - UNDER_M), h)
    # every building stood where the land meets it, the land cut down round it (buildings.settle)
    h = buildings.settle(outlines, en, h, owner)
    # the shore shaped as a game's (water.py): the land eases into the water,
    # a quay by a road stands; under it the land goes on as the water's bed
    lines = roads.lines(elements, to_xy)
    h = wet.carve(en, h, roads.near(lines, water.QUAY_M))
    from scipy.spatial import Delaunay
    faces = Delaunay(en).simplices
    seen = np.zeros(len(en), bool)
    seen[faces] = True
    faces = (np.cumsum(seen) - 1)[faces]
    en, h, dist, edge_c, lat, lon, raw = (a[seen] for a in (en, h, dist, edge_c, lat, lon, raw))
    land = np.stack([en[:, 0], -h, en[:, 1]], 1)

    # slope shading from the map's height a metre east and north
    d = 1.0
    he = heights(lat, lon0 + (en[:, 0] + d) / m_per_lon)
    hn = heights(lat0 + (en[:, 1] + d) / M_PER_LAT, lon)
    normal = np.stack([-(he - raw) / d, -np.ones(len(h)), -(hn - raw) / d], 1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    shade = np.clip(normal @ (SUN / np.linalg.norm(SUN)), 0, 1)
    colours = colour_map()
    try:
        cols = google_colours(lat, lon, LAND_EVERY * gap_at(cam_tree.query(en)[0]))
        source = "Google's satellite"
        gone = np.isnan(cols[:, 0])
        if gone.any():                    # where Google has none, Sentinel-2's, lifted (it is dark from above)
            cols[gone] = colours(lat[gone], lon[gone]) ** LIFT
            source += f", Sentinel-2's for {gone.mean():.0%}"
        cols = cols * (0.8 + 0.2 * shade[:, None])
    except OSError as e:                  # the land still stands without its colour
        log(f"terrain: no satellite colour ({e!r}), plain")
        colours = None
        cols = PLAIN * (0.55 + 0.45 * shade[:, None])
        source = "plain"
    cols = cols + (np.array(SAND) - cols) * (SAND_MIX * wet.sand(en, h))[:, None]
    cols = cols + (np.array(water.BED) - cols) * wet.bed(en, h)[:, None]     # under the water, the sky's pale blue
    land_cols = seams.tint(cols, edge_c, dist, 0.0, TINT_M, TINT)
    pts, cols = np.zeros((0, 3)), np.zeros((0, 3))

    # OpenStreetMap's buildings and roads, on this ground
    panos = None
    n_buildings = n_seen = n_roads = 0
    bp = bc = np.zeros((0, 3))
    b_roof = b_alone = np.zeros(0, bool)
    solid, solid_base = [], np.zeros((0, 3))
    n_cut = n_sat = 0
    if outlines:
        try:                                # roofs as the satellite sees them: Google's, sharp
            roof_at = google_map(ROOF_ZOOM)
            roof_at.fetch(*to_ll(np.concatenate([o[0] for o in outlines])))
            n_sat = buildings.satellite_roofs(outlines, lambda en: roof_at(*to_ll(en)),
                                              ROOF_INSET_M, ROOF_STEP_M, lively=False)
        except OSError as e:
            log(f"terrain: no Google roofs ({e!r}), Sentinel-2's")
        try:                                # ... else Sentinel-2's, for those still without
            n_sat += buildings.satellite_roofs(outlines, lambda en: colours(*to_ll(en)) ** LIFT) if colours else 0
        except OSError as e:
            log(f"terrain: no satellite roofs ({e!r})")
        try:
            panos = panos or _panos(sc, scene_dir)
            pal = buildings.palette(panos[1])
        except (OSError, ValueError) as e:  # a pastel palette stands
            log(f"terrain: no palette from the panos ({e!r})")
            panos, pal = ([], []), (np.array(buildings.PASTEL), np.full(len(buildings.PASTEL), 1 / 8))
        base = buildings.colours(outlines, pal)
        n_buildings = len(outlines)
        blocks = buildings.points(outlines, gap, ground, base, SUN / np.linalg.norm(SUN))
        scene_tree = cKDTree(scene) if len(scene) else None
        reached = buildings.reached(blocks, scene_tree, len(outlines))
        if panos[0]:
            # a building the panos see enough of: their colour, softened as the palette's is (in
            # shade or far off they see it dark), over the palette's
            own = buildings.pano_colours(blocks.pts, blocks.which, len(outlines), *panos, scene)
            seen = ~np.isnan(own[:, 0]) & reached
            base[seen] = [buildings.soften(c) for c in own[seen]]
            recolour = seen[blocks.which] & ~blocks.own               # a roof:colour stands
            b = base[blocks.which[recolour]]
            roof = np.isnan(blocks.light[recolour])
            blocks.cols[recolour] = np.where(
                roof[:, None], np.clip(b * blocks.shade[recolour][:, None], 0, 1),
                b * (buildings.WALL_SHADE + (1 - buildings.WALL_SHADE) * np.nan_to_num(blocks.light[recolour])[:, None]))
            n_seen = int(seen.sum())
        # one DA3 never reaches solid, not points -- coloured by the panos all the same
        solid, solid_base = [o for o, r in zip(outlines, reached) if not r], base[~reached]
        blocks.take(reached[blocks.which])
        # what DA3 already has of a building is left to it; the rest meets it
        roofs_near = scene_tree.query(blocks.pts, distance_upper_bound=1.0)[0] if scene_tree \
            else np.full(len(blocks.pts), np.inf)
        n_cut = buildings.seam(blocks, scene, scene_normals, scene_cols, roofs_near)
        buildings.windows(blocks, outlines, ground)
        bp, bc, b_roof, b_gap = blocks.pts, blocks.cols, blocks.edge == buildings.ROOF, blocks.gap
        b_alone = ~reached[blocks.which]                                # no DA3 near: no pano paint
    road_mesh = None
    if net.shapes:
        near_cams = shapely.union_all(shapely.buffer(shapely.points(cam_xz), NEAR_M, quad_segs=16))
        road_mesh = roads.surface(net, road_h,
                                  shapely.difference(shapely.box(-radius, -radius, radius, radius), near_cams))
        rp, rc = roads.points(net, gap, road_h, lambda xy: cam_tree.query(xy)[0] < NEAR_M)
        keep = uncovered(rp[:, [0, 2]], gap(rp[:, [0, 2]]))
        keep &= ~wet.inside(rp[:, [0, 2]])                              # no road in the water
        rp = rp[keep]
        pts, cols = np.concatenate([pts, rp]), np.concatenate([cols, rc[keep]])
        n_roads = len(net.strips) + len(net.fills)
    # bridges, end to end over whatever they cross; DA3's own where it has them
    # bridges (their decks, made before the land)
    n_bridges = len(over)
    if over:
        # near the cameras points, as the roads; further, a surface with the far roads
        close = np.array([cam_tree.query(b.xy)[0].min() < NEAR_M for b in over], bool)
        far_b = roads.bridge_surface([b for b, c in zip(over, close) if not c])
        if len(far_b[2]):
            road_mesh = far_b if road_mesh is None or not len(road_mesh[2]) else (
                np.concatenate([road_mesh[0], far_b[0]]), np.concatenate([road_mesh[1], far_b[1]]),
                np.concatenate([road_mesh[2], far_b[2] + len(road_mesh[0])]))
        over = [b for b, c in zip(over, close) if c]
    if over:
        bp_, bc_ = roads.bridge_points(over, gap)
        d, g, _ = near(bp_[:, [0, 2]])
        keep = d > 0                                     # the scene's footprint: its own bridge
        bp_, bc_ = bp_[keep], bc_[keep]
        pts, cols = np.concatenate([pts, bp_]), np.concatenate([cols, bc_])

    # near the cameras, everything coloured from the panos as the fill's ground is
    n_painted = 0
    try:
        panos = panos or _panos(sc, scene_dir)
        cols, n = _paint(panos, pts, cols, scene)
        land_cols, k = _paint(panos, land, land_cols, scene)
        bc, m = _paint(panos, bp, bc, scene, skip=b_roof | b_alone)
        n_painted = n + k + m
    except (OSError, ValueError) as e:    # the maps' colours stand
        log(f"terrain: no pano paint ({e!r})")
    write_mesh(os.path.join(scene_dir, LAND_FILENAME), land, land_cols, faces)
    sc.roads = None
    if road_mesh is not None and len(road_mesh[2]):
        write_mesh(os.path.join(scene_dir, ROADS_FILENAME), *road_mesh, gap=gap(road_mesh[0][:, [0, 2]]))
        sc.roads = ROADS_FILENAME
    sc.land = LAND_FILENAME
    sc.terrain = None
    if len(pts):
        write_ply(os.path.join(scene_dir, FILENAME), pts, cols, gap(pts[:, [0, 2]]))
        sc.terrain = FILENAME
    surfaces = wet.surfaces
    sc.water = water.save(scene_dir, wet) if surfaces else None
    sc.buildings = None
    if len(bp):
        write_ply(os.path.join(scene_dir, BUILDINGS_FILENAME), bp, bc, b_gap)
        sc.buildings = BUILDINGS_FILENAME
    sc.blocks = None
    if solid:
        # a building's points all alike: spaced as its nearest point is, but no
        # coarser far off than BLOCK_RATE (the map's points are: a storey apart by 400 m)
        def block_gap(xy):
            d = edge_tree.query(xy)[0]
            return np.minimum(point_gap(d), POINT_M + BLOCK_RATE * d).min()
        one = np.array([block_gap(xy) for xy, *_ in solid])
        v, c, f, facade, g = buildings.solid(solid, ground, solid_base, SUN / np.linalg.norm(SUN), one)
        write_mesh(os.path.join(scene_dir, BLOCKS_FILENAME), v, c, f, facade, g)
        sc.blocks = BLOCKS_FILENAME
    sc.save(scene_dir)
    fix = np.abs(fixes - shift)
    log(f"terrain: land {len(land)} vertices, {len(faces)} triangles to {radius:.0f} m, "
        f"{len(pts)} road points, {len(road_mesh[2]) if road_mesh else 0} road triangles, {len(bp)} building points, {n_painted} of them "
        f"painted from the panos ({len(surfaces)} water surfaces ({wet.source}), {source} colour, "
        f"{n_buildings} buildings ({len(solid)} solid) -- {n_fitted} fitted onto DA3's walls, {n_trimmed} trimmed to them, "
        f"{n_cut} of their points "
        f"left to DA3's own, {n_seen} coloured by the panos, {n_sat} roofs by the satellite -- {n_roads} roads, {n_bridges} bridges), map shifted "
        f"{shift:+.1f} m to Google's datum, then bent onto {len(known)} panos' elevation "
        f"(by up to {fix.max() if len(fix) else 0:.1f} m, median {np.median(fix) if len(fix) else 0:.1f})")


def _panos(sc, scene_dir):
    """(cameras, photos) of the scene's own panos, as the fill has them:
    photos[k] is (image, class map, mask of what may not colour) or None."""
    from streetview_to_3d.fill import _photo
    from streetview_to_3d.fill.paint import Camera, blurred
    from streetview_to_3d.services.segment import pano_mask
    nodes = [nd for nd in sc.nodes if nd.ply and nd.transform and nd.rotation is not None]
    photos = []
    for nd in nodes:
        ph = _photo(nd.pano, scene_dir)
        photos.append(ph and (ph[0], ph[1], pano_mask(ph[1]) | blurred(ph[0])))
    return [Camera(nd) for nd in nodes], photos


def _paint(panos, pts, cols, scene, skip=None):
    """cols, those of pts within PAINT_M of a camera re-coloured from the
    panos exactly as the fill colours its ground (fill.paint: the nearest
    camera that sees a point cleanly, patch by patch, the scene's points
    what stands in front), but not those marked skip (roofs: a pano sees
    their edge against the sky); how many were."""
    from streetview_to_3d.fill.paint import paint
    cams, photos = panos
    if not cams or not len(pts):
        return cols, 0
    centres = np.array([c.centre for c in cams])
    from scipy.spatial import cKDTree
    near = cKDTree(centres[:, [0, 2]]).query(pts[:, [0, 2]])[0] < PAINT_M
    near = np.flatnonzero(near & ~skip if skip is not None else near)
    if not len(near):
        return cols, 0
    painted, who = paint(pts[near], scene, cams, [ph and (ph[0], ph[2]) for ph in photos], max_m=PAINT_M)
    cols = cols.copy()
    cols[near[who >= 0]] = painted[who >= 0]
    return cols, int((who >= 0).sum())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__.splitlines()[-1].strip())
    build(sys.argv[1])
