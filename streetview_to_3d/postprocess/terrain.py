"""The land, buildings and roads around the scene, from free global maps.

DA3 reaches a few tens of metres; hills and mountains
further out come from AWS Terrain Tiles (terrarium PNGs, no key, ~30 m
data, mostly SRTM -- the ground, big buildings at most a blur), coloured
from EOX's Sentinel-2 cloudless mosaic (no key, 10 m, CC BY-NC-SA: credit
"Sentinel-2 cloudless by EOX", not for sale). Sampled as points, dense
near the scene and sparser with distance so each covers about the same
share of the view:

1. points out to NEAR_RADIUS_M, or FAR_RADIUS_M where hills rise beyond it
   (reach), further apart the further from the nearest camera (gap_at);
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
   buildings fitted onto DA3's walls, what DA3 has of them left to it
5. near the cameras (PAINT_M), all of it -- land, roads, walls -- coloured
   from the scene's own panos as the fill colours its ground (_paint), so
   it matches DA3 where they meet; the maps' colours are only for what no
   pano sees

Written to terrain.ply beside scene.json (its "terrain"), the buildings,
spaced the same way, to buildings.ply (its "buildings"), both already in
the world frame. The viewer draws its points larger with distance, as they
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
PAINT_M = 30.0                            # map points this near a camera are coloured from the panos
TINT_M, MEET_M = 8.0, 10.0                # the ground's seam with the scene (seams.py): bands
UNDER_M = 0.1                             # the land under the scene: this far beneath its lowest points
LOW_M, COVER = 1.0, 0.75                  # the scene's ground: points this near the map's; land it has within
                                          # this much of the land's own gap is the scene's
TINT = 0.8                                # how far the map takes the scene's colour at its edge
GAP0_M, GAP_PER = 0.05, 0.015              # map points' spacing (gap_at)
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


def gap_at(cam_d):
    """How far apart map points are cam_d metres from the nearest camera:
    GAP0_M there (DA3's own are ~4 cm apart), GAP_PER of the distance more
    further out -- one even growth from the scene outwards. Measured from
    the centre with a 50 cm floor instead, the land was coarse right at the
    scene's edge."""
    return GAP0_M + GAP_PER * np.asarray(cam_d)


def reach(ground, ground_here):
    """How far the land is laid: FAR_RADIUS_M where it rises HILL_M above
    the scene's own ground somewhere between NEAR_RADIUS_M and that --
    hills worth seeing -- else NEAR_RADIUS_M."""
    r = np.linspace(NEAR_RADIUS_M, FAR_RADIUS_M, 12)[:, None]
    a = np.linspace(0, 2 * math.pi, 96, endpoint=False)[None]
    ring = np.stack([(r * np.cos(a)).ravel(), (r * np.sin(a)).ravel()], 1)
    return FAR_RADIUS_M if ground(ring).max() - ground_here > HILL_M else NEAR_RADIUS_M


def sample_points(radius_m, cams):
    """(east, north) within radius_m of the centre, gap_at their distance
    to the nearest camera (cams, (n, 2)) apart: nested grids, each twice
    the last's spacing, each laid only where the gap wanted is between its
    spacing and twice that -- and shaken a little, so no grid shows."""
    from scipy.spatial import cKDTree
    tree, rng, out = cKDTree(cams), np.random.default_rng(0), []
    lo_c, hi_c = cams.min(0), cams.max(0)
    s = GAP0_M
    while (s - GAP0_M) / GAP_PER < 2 * radius_m:
        d_lo, d_hi = (s - GAP0_M) / GAP_PER, (2 * s - GAP0_M) / GAP_PER
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

    # the scene always wins: the map only around it, faded in at its edge
    scene, scene_cols = scene_points(sc, scene_dir)
    foot = seams.Footprint(scene, scene_cols) if len(scene) else None
    near = (lambda xy: foot.at(xy)) if foot else \
        (lambda xy: (np.full(len(xy), np.inf), np.full(len(xy), np.nan), np.full((len(xy), 3), np.nan)))
    radius = reach(ground, float(np.median(fixes + under)) if len(known) else 0.0)
    cam_tree = cKDTree(cam_xz)
    gap = lambda xy: gap_at(cam_tree.query(xy)[0])
    en = sample_points(radius, cam_xz)
    # the scene's ground-level points: the land fills exactly where they are not
    low = scene[-scene[:, 1] < ground(scene[:, [0, 2]]) + LOW_M] if len(scene) else scene
    low_tree = cKDTree(low[:, [0, 2]]) if len(low) else None
    uncovered = (lambda xy, gap: low_tree.query(xy)[0] > COVER * gap) if low_tree else \
        (lambda xy, gap: np.ones(len(xy), bool))
    under = ~uncovered(en, gap(en))
    dist, edge_h, edge_c = near(en)
    lat, lon = to_ll(en)
    raw = heights(lat, lon)
    sea = raw <= SEA_M                      # the tiles carry the sea bed too; laid flat
    h = seams.meet(ground(en, raw), edge_h, dist, MEET_M)
    # under the scene's own ground too, just beneath it: one shared ground,
    # so the scene's is not seen through, the land never over it
    h = np.where(under & np.isfinite(edge_h), np.minimum(h, np.nan_to_num(edge_h) - UNDER_M), h)
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
    panos = None
    n_buildings = n_seen = n_roads = 0
    try:
        elements = osm.fetch(lat0, lon0, BUILDINGS_M, ROADS_M, M_PER_LAT, m_per_lon, scene_dir)
    except (OSError, ValueError) as e:    # the land stands without them
        log(f"terrain: no OpenStreetMap ({e!r})")
        elements = []
    outlines = [o for o in buildings.outlines(elements, to_xy)
                if np.linalg.norm(o[0], axis=1).min() < BUILDINGS_M]   # an older, wider osm.json
    bp = bc = np.zeros((0, 3))
    b_roof = np.zeros(0, bool)
    n_fitted = n_trimmed = n_cut = 0
    if outlines:
        from streetview_to_3d.postprocess.ground import normals_from_neighbours
        scene_normals = normals_from_neighbours(scene) if len(scene) >= 12 else np.zeros_like(scene)
        outlines, n_fitted, n_trimmed = buildings.fit_to_scene(outlines, scene, scene_normals, ground,
                                                               roads.coverage(roads.lines(elements, to_xy)))
        blocks = buildings.points(
            outlines, gap, ground,
            lambda xy: colours(*to_ll(xy)) ** LIFT if colours else np.tile(PLAIN, (len(xy), 1)),
            SUN / np.linalg.norm(SUN))
        n_buildings = len(outlines)
        try:
            panos = panos or _panos(sc, scene_dir)
            own = buildings.pano_colours(blocks.pts, blocks.which, len(outlines), *panos, scene)
            wall = ~np.isnan(blocks.light) & ~np.isnan(own[blocks.which, 0])
            blocks.cols[wall] = buildings.cheer(own[blocks.which[wall]]) * (
                buildings.WALL_SHADE + (1 - buildings.WALL_SHADE) * blocks.light[wall, None])
            n_seen = int((~np.isnan(own[:, 0])).sum())
        except (OSError, ValueError) as e:  # the satellite's colour stands
            log(f"terrain: no pano colour for buildings ({e!r})")
        # what DA3 already has of a building is left to it; the rest meets it
        roofs_near = cKDTree(scene).query(blocks.pts, distance_upper_bound=1.0)[0] if len(scene) \
            else np.full(len(blocks.pts), np.inf)
        n_cut = buildings.seam(blocks, scene, scene_normals, scene_cols, roofs_near)
        bp, bc, b_roof = blocks.pts, blocks.cols, blocks.edge == buildings.ROOF
    lines = roads.lines(elements, to_xy)
    if lines:
        rp, rc = roads.points(lines, gap, ground)
        keep = uncovered(rp[:, [0, 2]], gap(rp[:, [0, 2]]))
        rp = rp[keep]
        d, g, _ = near(rp[:, [0, 2]])
        rp[:, 1] = -(seams.meet(-rp[:, 1] - roads.LIFT_M, g, d, MEET_M) + roads.LIFT_M)
        # no ground under a road: two layers 15 cm apart fight in the depth buffer far off
        if len(rp):
            xz = pts[:n_ground][:, [0, 2]]
            under = cKDTree(rp[:, [0, 2]]).query(xz)[0] < gap(xz) / 2
            pts, cols = pts[np.r_[~under, np.ones(len(pts) - n_ground, bool)]], \
                cols[np.r_[~under, np.ones(len(cols) - n_ground, bool)]]
        pts, cols = np.concatenate([pts, rp]), np.concatenate([cols, rc[keep]])
        n_roads = len(lines)

    # near the cameras, everything coloured from the panos as the fill's ground is
    n_painted = 0
    try:
        panos = panos or _panos(sc, scene_dir)
        cols, n = _paint(panos, pts, cols, scene)
        bc, m = _paint(panos, bp, bc, scene, skip=b_roof)
        n_painted = n + m
    except (OSError, ValueError) as e:    # the maps' colours stand
        log(f"terrain: no pano paint ({e!r})")
    write_ply(os.path.join(scene_dir, FILENAME), pts, cols)
    sc.terrain = FILENAME
    sc.buildings = None
    if len(bp):
        write_ply(os.path.join(scene_dir, BUILDINGS_FILENAME), bp, bc)
        sc.buildings = BUILDINGS_FILENAME
    sc.save(scene_dir)
    fix = np.abs(fixes - shift)
    log(f"terrain: {len(pts)} points to {radius:.0f} m, {len(bp)} building points, {n_painted} of them "
        f"painted from the panos ({int(sea.sum())} sea, {source} colour, "
        f"{n_buildings} buildings -- {n_fitted} fitted onto DA3's walls, {n_trimmed} trimmed to them, "
        f"{n_cut} of their points "
        f"left to DA3's own, {n_seen} coloured by the panos -- {n_roads} roads), map shifted "
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
    painted, who, _ = paint(pts[near], scene, cams, [ph and (ph[0], ph[2]) for ph in photos], max_m=PAINT_M)
    cols = cols.copy()
    cols[near[who >= 0]] = painted[who >= 0]
    return cols, int((who >= 0).sum())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__.splitlines()[-1].strip())
    build(sys.argv[1])
