"""The land, buildings, roads, water and life around the scene, from free global maps (AWS
Terrain Tiles, Google and Sentinel-2 imagery, OSM), coloured from the panos near the cameras.
Writes land.ply, terrain.ply, roads.ply, buildings.ply, blocks.ply, water.json and life.json.
Sentinel-2 credit: "Sentinel-2 cloudless by EOX".

    python -m streetview_to_3d.postprocess.world.terrain SCENE_DIR
"""
import math
import os
import sys

import numpy as np

from streetview_to_3d.common import scene as scene_mod
from streetview_to_3d.postprocess import seams
from streetview_to_3d.postprocess.world import buildings, elevations, life, osm, roads, water
from streetview_to_3d.postprocess.ply_io import read_node, write_mesh, write_ply
from streetview_to_3d.postprocess.seams import ramp
from streetview_to_3d.postprocess.world.tiles import TileMap

FILENAME = "terrain.ply"
ROADS_FILENAME = "roads.ply"
LAND_FILENAME = "land.ply"
BUILDINGS_FILENAME = "buildings.ply"
BLOCKS_FILENAME = "blocks.ply"
HEIGHT_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
HEIGHT_ZOOM = 13          # ~19 m a pixel at the equator
COLOUR_URL = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2025_3857/default/g/{z}/{y}/{x}.jpg"
COLOUR_ZOOM = 14          # ~10 m a pixel
GOOGLE_URL = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
GOOGLE_ZOOMS = (11, 18)   # land colour zoom range: ~75 m to ~0.6 m a pixel
ROOF_ZOOM = 18
EQUATOR_M = 156543.03     # a zoom 0 pixel's width at the equator
ROOF_INSET_M, ROOF_STEP_M = 1.0, 1.0   # satellite roof sampling: inset from the edge, spacing
RADIUS_M = 1000.0                          # the land's reach (the viewer's haze is total by then)
OSM_M, ROADS_M = RADIUS_M, 700.0           # OSM reach for buildings/water and for main roads
BUILDINGS_M = 800.0                        # buildings this near the centre are kept
BY_PANO_M = 20.0                           # ... those within this of a pano, all of them
MAJOR_M2, MAJOR_HIGH_M = 1000.0, 25.0      # ... further off only the major: this big on the ground, or this high
NEAR_M = 50.0                              # roads and bridges are points this near a camera, meshes past
PAINT_M = 30.0                             # map points this near a camera are coloured from the panos
TINT_M, MEET_M = 8.0, 10.0                 # seam bands for colour and height (seams.py)
BRIDGE_CLEAR_M = {"road": 4.5, "water": 2.5}     # min deck clearance over each
ROAD_MEET_M, ROAD_MEET_MAX_M = 40.0, 10.0  # roads ease to the scene's road height over this; not if this far apart
TINT = 0.8                                 # how far the map takes the scene's colour at its edge
GAP0_M, GAP_PER = 0.05, 0.018              # gap_at: spacing at a camera, growth per metre
SAND, SAND_MIX = (0.76, 0.70, 0.55), 0.7   # shore sand colour, its max cover
LAND_EVERY = 2                             # land triangles this many times the point spacing
M_PER_LAT = 111320.0
PLAIN = np.array([0.50, 0.55, 0.45])
LIFT = 0.75                                # colour ** LIFT brightens Sentinel-2's dark shadows
PANOS_K, PANOS_SOFT_M = 8, 10.0            # panos the ground is interpolated from; distance softening
PANOS_FROM_M, PANOS_TO_M = 30.0, 150.0     # pano ground fades into the map between these distances

def height_map():
    """Metres above sea level."""
    return TileMap(HEIGHT_URL, HEIGHT_ZOOM, lambda c: c[..., 0] * 256 + c[..., 1] + c[..., 2] / 256 - 32768)


def colour_map():
    """RGB, 0-1."""
    return TileMap(COLOUR_URL, COLOUR_ZOOM, lambda c: c / 255)


def google_map(zoom):
    """RGB, 0-1, NaN where Google has no tile."""
    from streetview_to_3d.common.http import BROWSER_HEADERS
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
    """Map point spacing cam_d metres from the nearest camera: GAP0_M plus GAP_PER per metre."""
    return GAP0_M + GAP_PER * np.asarray(cam_d)


def sample_points(radius_m, cams, every=1):
    """(east, north) points within radius_m, every x gap_at(distance to the nearest camera) apart:
    nested jittered grids, each twice the last's spacing, each used where its spacing fits."""
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


def _merged(a, b):
    """Two (points, colours, triangles) meshes as one; a may be None."""
    if a is None or not len(a[2]):
        return b
    if not len(b[2]):
        return a
    return np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]]), np.concatenate([a[2], b[2] + len(a[0])])


def from_panos(at, elevation):
    """f(east/north (n, 2)) -> (height, weight): pano elevations (at (m, 2)) spread by inverse distance,
    weight 1 within PANOS_FROM_M of the nearest, fading to 0 at PANOS_TO_M."""
    from scipy.spatial import cKDTree
    if not len(at):
        return lambda xy: (np.zeros(len(xy)), np.zeros(len(xy)))
    tree = cKDTree(at)

    def f(xy):
        k = min(PANOS_K, len(at))
        d, i = tree.query(xy, k=k)
        d, i = d.reshape(len(xy), k), i.reshape(len(xy), k)
        w = 1 / (d ** 2 + PANOS_SOFT_M ** 2)
        spread = (w * elevation[i]).sum(1) / w.sum(1)
        return spread, 1 - ramp((d[:, 0] - PANOS_FROM_M) / (PANOS_TO_M - PANOS_FROM_M))
    return f


SCENE_EVERY = 4                            # scene points subsampled to every this many-th


def scene_points(sc, scene_dir, every=SCENE_EVERY):
    """(points, colours) of every placed node's cloud in the world, every every-th point."""
    pts, cols = [np.zeros((0, 3))], [np.zeros((0, 3))]
    for nd in sc.nodes:
        if nd.ply and nd.transform:
            p, c, world = read_node(scene_dir, nd, every)
            pts.append(world)
            cols.append(c if c is not None else np.full((len(p), 3), 0.5))
    return np.concatenate(pts), np.concatenate(cols)


def build(scene_dir, log=print):
    """Build the map around the placed (and filled) scene at scene_dir."""
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

    # ground: Google's pano elevations near panos, the height map shifted onto Google's datum elsewhere
    heights = height_map()
    known = [n for n in sc.nodes if n.pano.elevation is not None]
    anchors = np.array([((n.pano.lon - lon0) * m_per_lon, (n.pano.lat - lat0) * M_PER_LAT)
                        for n in known]).reshape(-1, 2)
    own_el = np.array([n.pano.elevation for n in known])
    try:
        elements = osm.fetch(lat0, lon0, OSM_M, ROADS_M, M_PER_LAT, m_per_lon, scene_dir,
                             water_m=OSM_M, cams=to_ll(cam_xz), log=log)
    except (OSError, ValueError) as e:
        log(f"terrain: no OpenStreetMap ({e!r})")
        elements = []

    # other panos' elevations around (elevations.py), where the scene's own are not
    try:
        la, lo, el = elevations.around(*to_ll(cam_xz))
        xy = np.c_[(lo - lon0) * m_per_lon, (la - lat0) * M_PER_LAT]
        level = elevations.on_ground(xy, elements, to_xy)
        more, more_el = elevations.thinned(xy[level], el[level])
    except (OSError, ValueError) as e:
        log(f"terrain: no other panos' elevation ({e!r})")
        more, more_el = np.zeros((0, 2)), np.zeros(0)
    if len(anchors) and len(more):
        keep = cKDTree(anchors).query(more)[0] > elevations.CELL_M
        more, more_el = more[keep], more_el[keep]
    at, elevation = np.r_[anchors, more], np.r_[own_el, more_el]
    fixes = elevation - heights(*to_ll(at)) if len(at) else np.zeros(0)
    shift = float(np.median(fixes)) if len(at) else 0.0
    panos_ground = from_panos(at, elevation)

    def ground(xy, raw=None):
        raw = heights(*to_ll(xy)) if raw is None else raw
        h, w = panos_ground(xy)
        return (np.where(raw <= water.SEA_M, 0.0, raw) + shift) * (1 - w) + h * w

    jrc = water.jrc(to_ll, heights)
    wet = water.Water(RADIUS_M, to_ll, heights, shift, (anchors, own_el), jrc,
                      osm=(water.outline(elements, to_xy, OSM_M,
                                         lambda xy: osm.water_at(np.stack(to_ll(xy), 1), scene_dir, log=log), jrc),
                           OSM_M) if elements else None)

    # the scene always wins: the map only around it, faded in at its edge
    scene, scene_cols = scene_points(sc, scene_dir)
    scene_tree = cKDTree(scene) if len(scene) else None
    scene_ground = seams.SceneGround.load(scene_dir)
    near = scene_ground.at
    cam_tree = cKDTree(cam_xz)
    # map points sized, so spaced, by their distance from DA3's points (seams.spacing_at): from the
    # scene's edge, 1 m squares holding at least 3 of its points
    cell, n = np.unique(np.floor(scene[:, [0, 2]]), axis=0, return_counts=True) if len(scene) else (cam_xz, None)
    edge_tree = cKDTree(cell[n >= 3] + 0.5 if n is not None and (n >= 3).any() else cam_xz)
    gap = lambda xy: seams.spacing_at(edge_tree.query(xy, workers=-1)[0])
    # roads decide their own height (smoothed ground); the land fits to them
    net = roads.Network(elements, to_xy, lambda xy: cam_tree.query(xy)[0].min() < roads.DETAIL_M)
    road_raw = net.heights(ground, (-ROADS_M - 50, -ROADS_M - 50), (ROADS_M + 50, ROADS_M + 50))

    def to_scene(xy, h):
        d, g, _ = near(xy)
        return seams.meet(h, g, d, ROAD_MEET_M, ROAD_MEET_MAX_M)
    # buildings, fitted onto the scene's walls
    # only those by the panos and the major ones: the rest cost time and points for little
    import shapely
    cams = shapely.multipoints(cam_xz)
    outlines = [o for o in buildings.outlines(elements, to_xy)
                if np.linalg.norm(o[0], axis=1).min() < BUILDINGS_M
                and (shapely.Polygon(o[0]).distance(cams) <= BY_PANO_M
                     or shapely.Polygon(o[0]).area >= MAJOR_M2 or o[1] >= MAJOR_HIGH_M)]
    n_fitted = n_trimmed = 0
    if outlines:
        from streetview_to_3d.postprocess.fill.ground import normals_from_neighbours
        scene_normals = normals_from_neighbours(scene) if len(scene) >= 12 else np.zeros_like(scene)
        outlines, n_fitted, n_trimmed = buildings.fit_to_scene(outlines, scene, scene_normals, ground,
                                                               roads.coverage(roads.lines(elements, to_xy)))
    # land corners, plus corners along road edges so no triangle spans one
    en = sample_points(RADIUS_M, cam_xz, LAND_EVERY)
    # bridge decks clear the roads and water they cross

    def crossed(road_h):
        def low(xy):
            out = np.full(len(xy), -np.inf)
            road = shapely.contains_xy(net.all, *xy.T) if not net.all.is_empty else np.zeros(len(xy), bool)
            out = np.where(road, road_h(xy) + BRIDGE_CLEAR_M["road"], out)
            level = wet._at(wet.level, xy, -np.inf)
            return np.where(wet.inside(xy), np.maximum(out, level + BRIDGE_CLEAR_M["water"]), out)
        return low
    # only the roads the panos stand on, and those joining them, meet the scene's road height
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
    # car roads near the scene (life.py)
    cars = {e["id"] for e in elements if e.get("tags", {}).get("highway") in life.CARS}
    stretches = life.roads(elements, to_xy, road_h, [d for d in over if d.ids & cars],
                           lambda xy: cam_tree.query(xy)[0] < life.REACH_M, [o[0] for o in outlines],
                           scene_ground)
    edges = np.concatenate([net.edges(), roads.deck_edges(over)])
    walls, owner = buildings.corners(outlines, lambda xy: LAND_EVERY * gap_at(cam_tree.query(xy)[0]))
    inside = np.linalg.norm(walls, axis=1) < RADIUS_M
    edges = edges[np.linalg.norm(edges, axis=1) < RADIUS_M]
    owner = np.r_[np.full(len(en) + len(edges), -1), owner[inside]]
    en = np.concatenate([en, edges, walls[inside]])
    # the land meets the scene's ground at its edge
    dist, edge_h, edge_c = near(en)
    under = dist == 0
    lat, lon = to_ll(en)
    raw = heights(lat, lon)
    h = seams.meet(ground(en, raw), edge_h, dist, MEET_M)
    h = roads.under_decks(over, en, net.adapt(en, h, road_h))
    # under the scene's ground the land runs on just beneath it (seams.beneath): all of it, as a
    # cut along its triangles shows through the scene's ground as zigzag gaps
    h = seams.beneath(h, edge_h, scene_ground.inside(en))[0]
    # land raised to DA3's ground beside buildings, then cut to each building's foot
    h = buildings.onto_scene(outlines, en, h, owner, scene_ground, under)
    h = buildings.settle(outlines, en, h, owner)
    # shape the shore; no bank by roads (quays)
    lines = roads.lines(elements, to_xy)
    h = wet.carve(en, h, roads.near(lines, water.QUAY_M))
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay
    tri = Delaunay(en)
    faces = tri.simplices
    at = LinearNDInterpolator(tri, h)

    def surface(xy):
        """Final land height (the map's ground past the mesh)."""
        v = at(xy)
        return np.where(np.isfinite(v), v, ground(xy))
    seen = np.zeros(len(en), bool)
    seen[faces] = True
    faces = (np.cumsum(seen) - 1)[faces]
    en, h, dist, edge_c, lat, lon = (a[seen] for a in (en, h, dist, edge_c, lat, lon))
    land = np.stack([en[:, 0], -h, en[:, 1]], 1)

    # unlit land colour; satellite colours corrected to the panos' look (seams.unhazed)
    try:
        unhazed = seams.unhazed(scene_ground, lambda xy: google_colours(*to_ll(xy), np.full(len(xy), seams.CELL_M)))
    except OSError as e:
        log(f"terrain: satellite colours as they are ({e!r})")
        unhazed = lambda cols: cols
    colours = colour_map()
    try:
        cols = google_colours(lat, lon, LAND_EVERY * gap_at(cam_tree.query(en, workers=-1)[0]))
        source = "Google's satellite"
        gone = np.isnan(cols[:, 0])
        if gone.any():                    # Sentinel-2 where Google has none
            cols[gone] = colours(lat[gone], lon[gone]) ** LIFT
            source += f", Sentinel-2's for {gone.mean():.0%}"
        cols = unhazed(cols)
    except OSError as e:
        log(f"terrain: no satellite colour ({e!r}), plain")
        colours = None
        cols = np.tile(PLAIN, (len(en), 1))
        source = "plain"
    cols = cols + (np.array(SAND) - cols) * (SAND_MIX * wet.sand(en, h))[:, None]
    cols = cols + (np.array(water.BED) - cols) * wet.bed(en, h)[:, None]
    land_cols = seams.tint(cols, edge_c, dist, 0.0, TINT_M, TINT)
    pts, cols = np.zeros((0, 3)), np.zeros((0, 3))

    # buildings and roads on this ground
    panos = None
    n_buildings = n_seen = n_roads = 0
    bp = bc = b_normal = np.zeros((0, 3))
    b_kind = np.zeros(0, int)
    b_roof = b_alone = np.zeros(0, bool)
    solid, solid_base = [], np.zeros((0, 3))
    n_cut = n_sat = 0
    if outlines:
        try:                                # roof colours from Google's satellite
            roof_at = google_map(ROOF_ZOOM)
            roof_at.fetch(*to_ll(np.concatenate([o[0] for o in outlines])))
            n_sat = buildings.satellite_roofs(outlines, lambda en: unhazed(roof_at(*to_ll(en))),
                                              ROOF_INSET_M, ROOF_STEP_M, lively=False)
        except OSError as e:
            log(f"terrain: no Google roofs ({e!r}), Sentinel-2's")
        try:                                # ... else Sentinel-2's
            n_sat += (buildings.satellite_roofs(outlines, lambda en: unhazed(colours(*to_ll(en)) ** LIFT))
                      if colours else 0)
        except OSError as e:
            log(f"terrain: no satellite roofs ({e!r})")
        try:
            panos = panos or _panos(sc, scene_dir)
            pal = buildings.palette(panos[1])
        except (OSError, ValueError) as e:
            log(f"terrain: no palette from the panos ({e!r})")
            panos, pal = ([], []), (np.array(buildings.PASTEL), np.full(len(buildings.PASTEL), 1 / 8))
        base = buildings.colours(outlines, pal)
        n_buildings = len(outlines)
        may = np.flatnonzero(buildings.reachable(outlines, scene))      # the rest are solid
        blocks = buildings.points([outlines[i] for i in may], gap, surface, base[may])
        blocks.which = may[blocks.which]
        reached = buildings.reached(blocks, scene_tree, len(outlines))
        if panos[0]:
            # buildings the panos see enough of take their colour, softened
            own = buildings.pano_colours(blocks.pts, blocks.which, len(outlines), *panos, scene)
            seen = ~np.isnan(own[:, 0]) & reached
            base[seen] = np.array([buildings.soften(c) for c in own[seen]]).reshape(-1, 3)
            recolour = seen[blocks.which] & ~blocks.own               # tagged roof colours stay
            blocks.cols[recolour] = base[blocks.which[recolour]]
            n_seen = int(seen.sum())
        # buildings DA3 never reaches become solid meshes
        solid, solid_base = [o for o, r in zip(outlines, reached) if not r], base[~reached]
        blocks.take(reached[blocks.which])
        # leave to DA3 what it already has of each building
        roofs_near = np.full(len(blocks.pts), np.inf)                 # only roofs use it
        roof = blocks.edge == buildings.ROOF
        if scene_tree is not None and roof.any():
            roofs_near[roof] = scene_tree.query(blocks.pts[roof], distance_upper_bound=1.0, workers=-1)[0]
        n_cut = buildings.seam(blocks, scene, scene_normals, scene_cols, roofs_near)
        buildings.windows(blocks, outlines, surface)
        bp, bc, b_roof, b_gap = blocks.pts, blocks.cols, blocks.edge == buildings.ROOF, blocks.gap
        b_normal, b_kind = blocks.normal, blocks.kind
        b_alone = ~reached[blocks.which]                                # no DA3 near: not pano-painted
    road_mesh = None
    if net.shapes:
        near_cams = shapely.union_all(shapely.buffer(shapely.points(cam_xz), NEAR_M, quad_segs=16))
        road_mesh = roads.surface(net, road_h,
                                  shapely.difference(shapely.box(-RADIUS_M, -RADIUS_M, RADIUS_M, RADIUS_M), near_cams))
        rp, rc = roads.points(net, gap, road_h, lambda xy: cam_tree.query(xy)[0] < NEAR_M)
        keep = ~scene_ground.covers(rp[:, [0, 2]])
        keep &= ~wet.inside(rp[:, [0, 2]])
        rp = rp[keep]
        pts, cols = np.concatenate([pts, rp]), np.concatenate([cols, rc[keep]])
        n_roads = len(net.strips) + len(net.fills)
    # bridges: points near the cameras, meshes further off
    n_bridges = len(over)
    if over:
        close = np.array([cam_tree.query(b.xy)[0].min() < NEAR_M for b in over], bool)
        road_mesh = _merged(road_mesh, roads.bridge_surface([b for b, c in zip(over, close) if not c]))
        over = [b for b, c in zip(over, close) if c]
    if over:
        bp_, bc_ = roads.bridge_points(over, gap)
        d, g, _ = near(bp_[:, [0, 2]])
        keep = d > 0                                     # the scene's own bridge wins
        bp_, bc_ = bp_[keep], bc_[keep]
        pts, cols = np.concatenate([pts, bp_]), np.concatenate([cols, bc_])

    # near the cameras, colour everything from the panos
    n_painted = 0
    try:
        panos = panos or _panos(sc, scene_dir)
        cols, n = _paint(panos, pts, cols, scene)
        land_cols, k = _paint(panos, land, land_cols, scene)
        bc, m = _paint(panos, bp, bc, scene, skip=b_roof | b_alone)
        n_painted = n + k + m
    except (OSError, ValueError) as e:
        log(f"terrain: no pano paint ({e!r})")
    # map points approaching DA3's turn into them (seams.toward)
    land_cols, land_near = seams.toward(land, land_cols, scene_tree, scene_cols)
    cols, _ = seams.toward(pts, cols, scene_tree, scene_cols)
    write_mesh(os.path.join(scene_dir, LAND_FILENAME), land, land_cols, faces, gap=gap(en), near=land_near)
    sc.roads = None
    # tunnel portals
    mouths = roads.portals(elements, to_xy, road_h)
    road_mesh = _merged(road_mesh, mouths)
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
    try:
        side = life.side(known[0].pano.id) if known else "right"
    except (OSError, ValueError) as e:
        log(f"terrain: no country for the cars' side ({e!r})")
        side = "right"
    sc.life = life.save(scene_dir, cars=life.cars(stretches, side),
                        birds=life.birds(surfaces, cam_xz, float(np.mean(ground(cam_xz)))),
                        boats=life.boats(surfaces, cam_xz), ducks=life.ducks(surfaces, cam_xz, scene_ground),
                        cats=life.cats(scene_ground, stretches, cam_xz))
    sc.buildings = None
    if len(bp):
        bc, b_near = seams.toward(bp, bc, scene_tree, scene_cols)
        write_ply(os.path.join(scene_dir, BUILDINGS_FILENAME), bp, bc, b_gap, b_normal, b_kind, b_near)
        sc.buildings = BUILDINGS_FILENAME
    sc.blocks = None
    if solid:
        # one spacing per solid building: its nearest point's
        one = np.array([gap(xy).min() for xy, *_ in solid])
        v, c, f, facade, g = buildings.solid(solid, surface, solid_base, one)
        write_mesh(os.path.join(scene_dir, BLOCKS_FILENAME), v, c, f, facade, g)
        sc.blocks = BLOCKS_FILENAME
    sc.save(scene_dir)
    fix = np.abs(fixes - shift)
    log(f"terrain: land {len(land)} vertices, {len(faces)} triangles to {RADIUS_M:.0f} m, "
        f"{len(pts)} road points, {len(road_mesh[2]) if road_mesh else 0} road triangles, {len(bp)} building points, {n_painted} of them "
        f"painted from the panos ({len(surfaces)} water surfaces ({wet.source}), {source} colour, "
        f"{n_buildings} buildings ({len(solid)} solid) -- {n_fitted} fitted onto DA3's walls, {n_trimmed} trimmed to them, "
        f"{n_cut} of their points "
        f"left to DA3's own, {n_seen} coloured by the panos, {n_sat} roofs by the satellite -- {n_roads} roads, {n_bridges} bridges, "
        f"{len(mouths[2]) // 14} tunnel mouths, {len(stretches)} stretches of road for cars, "
        f"keeping {side}), map shifted "
        f"{shift:+.1f} m to Google's datum, the ground near {len(known)} panos and {len(more)} other places "
        f"their elevation (the map off it by up to {fix.max() if len(fix) else 0:.1f} m, "
        f"median {np.median(fix) if len(fix) else 0:.1f})")


def _panos(sc, scene_dir):
    """(cameras, photos) of the scene's panos; photos[k] is (image, class map, drop mask) or None."""
    from streetview_to_3d.postprocess.fill import _photo
    from streetview_to_3d.postprocess.fill.paint import Camera, blurred
    from streetview_to_3d.models.segment import pano_mask
    nodes = [nd for nd in sc.nodes if nd.ply and nd.transform and nd.rotation is not None]
    photos = []
    for nd in nodes:
        ph = _photo(nd.pano, scene_dir)
        photos.append(ph and (ph[0], ph[1], pano_mask(ph[1]) | blurred(ph[0])))
    return [Camera(nd) for nd in nodes], photos


def _paint(panos, pts, cols, scene, skip=None):
    """(cols, count): pts within PAINT_M of a camera recoloured from the panos as the fill does
    (fill.paint), except those marked skip (roofs)."""
    from streetview_to_3d.postprocess.fill.paint import paint
    cams, photos = panos
    if not cams or not len(pts):
        return cols, 0
    centres = np.array([c.centre for c in cams])
    from scipy.spatial import cKDTree
    near = cKDTree(centres[:, [0, 2]]).query(pts[:, [0, 2]], workers=-1)[0] < PAINT_M
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
