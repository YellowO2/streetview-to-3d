"""The land, buildings and roads around the scene, from free global maps.

DA3 reaches a few tens of metres; hills and mountains
further out come from AWS Terrain Tiles (terrarium PNGs, no key, ~30 m
data, mostly SRTM -- the ground, big buildings at most a blur), coloured
from EOX's Sentinel-2 cloudless mosaic (no key, 10 m, CC BY-NC-SA: credit
"Sentinel-2 cloudless by EOX", not for sale). Sampled as points, dense
near the scene and sparser with distance so each covers about the same
share of the view:

1. points on rings out to NEAR_RADIUS_M, or FAR_RADIUS_M where hills rise
   beyond it (reach), STEP of their distance apart (never
   under MIN_STEP_M), none where the scene has its own points: the
   scene always wins, the map only fills around it, faded in at its edge
   and meeting its ground (seams.py)
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
   buildings fitted onto DA3's walls, what DA3 has of them left to it

Written to terrain.ply beside scene.json (its "terrain"), the buildings,
spaced by distance to the nearest camera (B_STEP), to buildings.ply (its "buildings"), both
already in the world frame. The viewer draws its points larger with distance, as they
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
from streetview_to_3d.postprocess import buildings, osm, roads, seams
from streetview_to_3d.postprocess.ply_io import write_ply

FILENAME = "terrain.ply"
BUILDINGS_FILENAME = "buildings.ply"
HEIGHT_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
HEIGHT_ZOOM = 13          # ~19 m a pixel at the equator, finer than the data
COLOUR_URL = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2025_3857/default/g/{z}/{y}/{x}.jpg"
COLOUR_ZOOM = 14          # ~10 m a pixel, the imagery's own
NEAR_RADIUS_M, FAR_RADIUS_M = 1000.0, 2000.0  # the land's reach: far only where hills rise HILL_M over the scene
HILL_M = 50.0
BUILDINGS_M, ROADS_M = 1000.0, 700.0         # OSM's reach (roads are drawn only where a point wide, ~600 m)
FADE_M, TINT_M, MEET_M = 4.0, 8.0, 10.0  # the ground's seam with the scene (seams.py): bands
TINT = 0.8                                # how far the map takes the scene's colour at its edge
WALL_ABOVE_M = 1.0                        # scene points this far over the ground are walls, trees
B_STEP, B_MIN_STEP_M = 0.01, 0.15         # buildings' spacing by distance to the nearest camera: DA3's next to them
STEP, MIN_STEP_M = 0.01, 0.5
M_PER_LAT = 111320.0
PLAIN = np.array([0.50, 0.55, 0.45])
LIFT = 0.75               # colour ** LIFT: brighter shadows, same hues
SEA_M = 0.5               # map height at or under this is sea
SUN = np.array([-0.5, -0.7, 0.5])        # world frame: x east, y down, z north
BEND_K, BEND_SOFT_M = 8, 10.0             # panos each bend is spread from; softening near one
BEND_FROM_M, BEND_TO_M = 30.0, 150.0      # the bend fades out between these from the nearest pano


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


def step_at(d):
    """Point spacing d metres from the centre."""
    return max(MIN_STEP_M, STEP * d)


def reach(ground, ground_here):
    """How far the land is laid: FAR_RADIUS_M where it rises HILL_M above
    the scene's own ground somewhere between NEAR_RADIUS_M and that --
    hills worth seeing -- else NEAR_RADIUS_M."""
    r = np.linspace(NEAR_RADIUS_M, FAR_RADIUS_M, 12)[:, None]
    a = np.linspace(0, 2 * math.pi, 96, endpoint=False)[None]
    ring = np.stack([(r * np.cos(a)).ravel(), (r * np.sin(a)).ravel()], 1)
    return FAR_RADIUS_M if ground(ring).max() - ground_here > HILL_M else NEAR_RADIUS_M


def sample_points(radius_m):
    """(east, north) on rings about the centre, step_at their radius apart."""
    r, rings = MIN_STEP_M, []
    rng = np.random.default_rng(0)
    while r < radius_m:
        step = step_at(r)
        k = max(6, int(2 * math.pi * r / step))
        a = (np.arange(k) + rng.random()) * 2 * math.pi / k
        rings.append(np.stack([r * np.cos(a), r * np.sin(a)], 1))
        r += step
    return np.concatenate(rings)


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

    # the scene always wins: the map only around it, faded in at its edge
    scene, scene_cols = scene_points(sc, scene_dir)
    foot = seams.Footprint(scene, scene_cols) if len(scene) else None
    near = (lambda xy: foot.at(xy)) if foot else \
        (lambda xy: (np.full(len(xy), np.inf), np.full(len(xy), np.nan), np.full((len(xy), 3), np.nan)))
    radius = reach(ground, float(np.median(fixes + under)) if len(known) else 0.0)
    en = sample_points(radius)
    dist, edge_h, edge_c = near(en)
    keep = seams.fade(dist, 0.0, FADE_M)
    en, dist, edge_h, edge_c = en[keep], dist[keep], edge_h[keep], edge_c[keep]
    lat, lon = to_ll(en)
    raw = heights(lat, lon)
    sea = raw <= SEA_M                      # the tiles carry the sea bed too; laid flat
    h = seams.meet(ground(en, raw), edge_h, dist, MEET_M)
    pts = np.stack([en[:, 0], -h, en[:, 1]], 1)
    n_ground = len(pts)

    # slope shading from the map's height a metre east and north
    d = 1.0
    he = heights(lat, lon0 + (en[:, 0] + d) / m_per_lon)
    hn = heights(lat0 + (en[:, 1] + d) / M_PER_LAT, lon)
    normal = np.stack([-(he - raw) / d, -np.ones(len(h)), -(hn - raw) / d], 1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    shade = np.clip(normal @ (SUN / np.linalg.norm(SUN)), 0, 1)
    colours = colour_map()
    try:
        cols = colours(lat, lon) ** LIFT * (0.8 + 0.2 * shade[:, None])
        source = "satellite"
    except OSError as e:                  # the land still stands without its colour
        log(f"terrain: no satellite colour ({e!r}), plain")
        colours = None
        cols = PLAIN * (0.55 + 0.45 * shade[:, None])
        source = "plain"
    cols = seams.tint(cols, edge_c, dist, 0.0, TINT_M, TINT)

    # OpenStreetMap's buildings and roads, on this ground
    n_buildings = n_seen = n_roads = 0
    try:
        elements = osm.fetch(lat0, lon0, BUILDINGS_M, ROADS_M, M_PER_LAT, m_per_lon, scene_dir)
    except (OSError, ValueError) as e:    # the land stands without them
        log(f"terrain: no OpenStreetMap ({e!r})")
        elements = []
    outlines = [o for o in buildings.outlines(elements, to_xy)
                if np.linalg.norm(o[0], axis=1).min() < BUILDINGS_M]   # an older, wider osm.json
    bp = bc = np.zeros((0, 3))
    n_fitted = n_cut = 0
    if outlines:
        from scipy.spatial import cKDTree
        above = -scene[:, 1] > ground(scene[:, [0, 2]]) + WALL_ABOVE_M
        outlines, n_fitted = buildings.fit_to_scene(outlines, scene[above], ground)
        cam_tree = cKDTree(cam_xz)
        blocks = buildings.points(
            outlines, lambda xy: np.maximum(B_MIN_STEP_M, B_STEP * cam_tree.query(xy)[0]), ground,
            lambda xy: colours(*to_ll(xy)) ** LIFT if colours else np.tile(PLAIN, (len(xy), 1)),
            SUN / np.linalg.norm(SUN))
        n_buildings = len(outlines)
        try:
            own = _pano_colours(sc, scene_dir, blocks.pts, blocks.which, len(outlines), scene)
            wall = ~np.isnan(blocks.light) & ~np.isnan(own[blocks.which, 0])
            blocks.cols[wall] = own[blocks.which[wall]] * (0.85 + 0.15 * blocks.light[wall, None])
            n_seen = int((~np.isnan(own[:, 0])).sum())
        except (OSError, ValueError) as e:  # the satellite's colour stands
            log(f"terrain: no pano colour for buildings ({e!r})")
        # what DA3 already has of a building is left to it; the rest meets it
        roofs_near = cKDTree(scene).query(blocks.pts, distance_upper_bound=1.0)[0] if len(scene) \
            else np.full(len(blocks.pts), np.inf)
        n_cut = buildings.seam(blocks, scene[above], scene_cols[above], roofs_near, np.random.default_rng(1))
        bp, bc = blocks.pts, blocks.cols
    lines = roads.lines(elements, to_xy)
    if lines:
        rp, rc = roads.points(lines, step_at, ground)
        d, g, _ = near(rp[:, [0, 2]])
        keep = seams.fade(d, 0.0, FADE_M, seed=2)
        rp, d, g = rp[keep], d[keep], g[keep]
        rp[:, 1] = -(seams.meet(-rp[:, 1] - roads.LIFT_M, g, d, MEET_M) + roads.LIFT_M)
        # no ground under a road: two layers 15 cm apart fight in the depth buffer far off
        if len(rp):
            from scipy.spatial import cKDTree
            xz = pts[:n_ground][:, [0, 2]]
            spacing = np.maximum(MIN_STEP_M, STEP * np.linalg.norm(xz, axis=1))
            under = cKDTree(rp[:, [0, 2]]).query(xz)[0] < spacing / 2
            pts, cols = pts[np.r_[~under, np.ones(len(pts) - n_ground, bool)]], \
                cols[np.r_[~under, np.ones(len(cols) - n_ground, bool)]]
        pts, cols = np.concatenate([pts, rp]), np.concatenate([cols, rc[keep]])
        n_roads = len(lines)

    write_ply(os.path.join(scene_dir, FILENAME), pts, cols)
    sc.terrain = FILENAME
    sc.buildings = None
    if len(bp):
        write_ply(os.path.join(scene_dir, BUILDINGS_FILENAME), bp, bc)
        sc.buildings = BUILDINGS_FILENAME
    sc.save(scene_dir)
    fix = np.abs(fixes - shift)
    log(f"terrain: {len(pts)} points to {radius:.0f} m, {len(bp)} building points ({int(sea.sum())} sea, {source} colour, "
        f"{n_buildings} buildings -- {n_fitted} fitted onto DA3's walls, {n_cut} of their points "
        f"left to DA3's own, {n_seen} coloured by the panos -- {n_roads} roads), map shifted "
        f"{shift:+.1f} m to Google's datum, then bent onto {len(known)} panos' elevation "
        f"(by up to {fix.max() if len(fix) else 0:.1f} m, median {np.median(fix) if len(fix) else 0:.1f})")


def _pano_colours(sc, scene_dir, pts, which, n, scene):
    """buildings.pano_colours, from the scene's own panos, with everything
    the scene holds (scene, its clouds placed) as what can stand in front."""
    from streetview_to_3d.fill import _photo
    from streetview_to_3d.fill.paint import Camera, blurred
    from streetview_to_3d.services.segment import pano_mask
    nodes = [nd for nd in sc.nodes if nd.ply and nd.transform and nd.rotation is not None]
    photos = []
    for nd in nodes:
        ph = _photo(nd.pano, scene_dir)
        photos.append(ph and (ph[0], ph[1], pano_mask(ph[1]) | blurred(ph[0])))
    return buildings.pano_colours(pts, which, n, [Camera(nd) for nd in nodes], photos, scene)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__.splitlines()[-1].strip())
    build(sys.argv[1])
