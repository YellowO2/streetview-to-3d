"""OSM buildings around the scene, stood on the terrain: heights and roofs from tags or regional
guesses (styles.py), fitted onto DA3's walls, points near the cameras and solid meshes further
off, coloured from the panos, tags, a palette and the satellite. Called by terrain.build."""
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from streetview_to_3d.postprocess.world.geometry import hash01, sample_quads, turning
from streetview_to_3d.postprocess.world.osm import building_kind, tag_number
from streetview_to_3d.postprocess.world.roofs import Roof

LEVEL_M = 3.2
DEFAULT_M = {"house": 7, "detached": 7, "semidetached_house": 7, "terrace": 7, "bungalow": 5,
             "roof": 4, "shed": 3, "garage": 3, "garages": 3, "carport": 3, "hut": 3, "kiosk": 3}
DEFAULT_OTHER_M = 12.0
SEE_M = 150.0                 # panos colour buildings within this distance
BEHIND_M, BEHIND = 3.0, 0.05  # depth tolerance (m, plus share of distance) for a wall to count as seen
SEEN_MIN = 20                 # pixels a building needs to take the panos' colour
SAMPLE = 16                   # colour and reach judged from every SAMPLE-th point
SAMPLE_HIDING = 4             # occluders: every SAMPLE_HIDING-th DA3 point
REACH_M = 3.0                 # DA3 points this near make a building "reached"
FIT_M = 5.0                   # buildings this far past the scene's points are not fitted
WALL_SAMPLE_M = 1.0
SNAP_MAX_M = 2.0              # max slide onto DA3 walls
ROAD_SLACK = 0.02             # a slide may put this much more of an outline onto roads
PULL_MAX_M = 1.0              # max pull of seam points onto DA3's plane
TRIM_TOL_M, TRIM_FRONT_M = 0.3, 30.0   # trim from this far in front of a DA3 wall, out to this
TRIM_MIN_M2, TRIM_MAX = 1.0, 0.2       # a trim must remove at least this area, at most this share
CHEER_SAT, CHEER_LIFT = 1.3, 1.15
SAT_INSET_M, SAT_STEP_M = 3.0, 4.0     # coarse satellite roof sampling: inset from the edge, spacing
PALETTE_K, PALETTE_MIN, PALETTE_STRIDE = 8, 500, 8
SOFT_LIGHT, SOFT_SAT = (0.62, 0.88), (0.15, 0.45)
PASTEL = [(0.95, 0.90, 0.80), (0.90, 0.72, 0.62), (0.74, 0.83, 0.92), (0.78, 0.90, 0.80),
          (0.94, 0.80, 0.80), (0.93, 0.88, 0.70), (0.82, 0.80, 0.90), (0.86, 0.86, 0.84)]
NAMED = {"white": (0.95, 0.95, 0.93), "grey": (0.6, 0.6, 0.6), "gray": (0.6, 0.6, 0.6),
         "black": (0.2, 0.2, 0.2), "red": (0.75, 0.3, 0.25), "brown": (0.55, 0.38, 0.26),
         "beige": (0.88, 0.8, 0.65), "yellow": (0.93, 0.83, 0.45), "orange": (0.9, 0.6, 0.3),
         "blue": (0.45, 0.6, 0.8), "green": (0.5, 0.7, 0.5), "pink": (0.93, 0.72, 0.75),
         "cream": (0.95, 0.9, 0.78), "tan": (0.82, 0.7, 0.55)}
MATERIAL = {"brick": (0.72, 0.4, 0.32), "stone": (0.8, 0.76, 0.68), "sandstone": (0.86, 0.77, 0.6),
            "limestone": (0.88, 0.85, 0.76), "concrete": (0.76, 0.76, 0.73), "glass": (0.62, 0.74, 0.82),
            "metal": (0.66, 0.68, 0.7), "steel": (0.55, 0.52, 0.48), "copper": (0.45, 0.7, 0.62),
            "plaster": (0.9, 0.86, 0.78), "wood": (0.62, 0.46, 0.32), "timber_framing": (0.9, 0.85, 0.75),
            "roof_tiles": (0.72, 0.36, 0.26), "tile": (0.72, 0.36, 0.26), "slate": (0.36, 0.39, 0.43),
            "tar_paper": (0.3, 0.3, 0.3), "eternit": (0.55, 0.55, 0.55), "zinc": (0.62, 0.65, 0.68)}
ROOF, INNER = -1, -2          # Blocks.edge for roof and inner-wall points
SURFACE, EDGE = 0, 1          # Blocks.kind, as effects/blocks.js draws them
GLASS, GLASS_OWN = (0.22, 0.43, 0.64), 0.12   # window colour: blue glass plus a little of the wall's
# DA3's copy of a wall: points within SAME_M of its plane, facing within SAME_DEG, within SAME_NEAR_M,
# at least SAME_MIN; its fitted plane must face within AGREE_DEG; stretches found in SPAN_M steps of
# SPAN_PER_M points per metre, gaps up to BRIDGE_M closed
SAME_DEG, SAME_M, SAME_NEAR_M, SAME_MIN = 40, 2.5, 5.0, 200
AGREE_DEG = 15
ON_WALL_M, SPAN_M, BRIDGE_M, SPAN_PER_M = 0.5, 0.2, 0.6, 60
INNER_M, INNER_GAP = (0.6, 1.5), 2.0   # inner walls this far in, this many times sparser, so gaps don't see through
TRIM_EDGE_M = 5.0             # only straight walls this long trim (short curved edges found false planes)
TOP_PERCENTILE = 97
SOLID_STEP_M = 2.0            # solid wall/roof triangles about this long
NO_FACADE = -1e4              # facade value for roof vertices
MIN_HEIGHT_M = 2.5
PAD_M, CUT_SLOPE = 1.0, 0.5   # land kept level this far round a building's foot, then rising at most this
CUT_REACH_M = 20.0            # ... out to this far
BESIDE_M = 1.5                # DA3 ground this near a corner counts as beside it
ON_GROUND_M = 0.05            # wall points reach this far under the ground, no lower
CHUNK_M = 4.0                 # walls are spaced in pieces this long
COVER = 0.75                  # an OSM point with a DA3 point within COVER x its gap is dropped
SEAM_FADE_M = 3.0             # seam width
SEAM_TINT, SEAM_MIN = 0.8, 20


def _number(tags, key):
    v = tag_number(tags, key)
    return v if v is not None and v >= 0 else None


def _height(tags):
    """(height m, whether it is only a guess)."""
    h = _number(tags, "height")
    if h:
        return h, False
    levels = _number(tags, "building:levels")
    if levels:
        return (levels + (_number(tags, "roof:levels") or 0)) * LEVEL_M, False
    return DEFAULT_M.get(building_kind(tags), DEFAULT_OTHER_M), True


@dataclass
class Form:
    """An outline's roof, base height above its foot, part flag, tag colours and build state."""
    roof: Roof
    base_m: float = 0.0
    part: bool = False
    colour: object = None
    roof_colour: object = None
    foot_m: object = None         # foot height, set by settle
    tags: dict = field(default_factory=dict)
    front: int = 0
    seed: int = 0
    shared_edges: frozenset = frozenset()
    geometry_cache: dict = field(default_factory=dict, repr=False)
    physical_facade: bool = False
    skirt_m: float = 0.0          # walls extended this far below the foot, to land a neighbour cut lower
    stand_on: object = None       # a Form whose foot this one shares (styles: a castle's tiers)


def _form(tags, xy, h):
    base = _number(tags, "min_height")
    if base is None:
        base = (_number(tags, "building:min_level") or 0) * LEVEL_M
    base = min(base, max(h - MIN_HEIGHT_M, 0.0))
    roof_m = _number(tags, "roof:height")
    if roof_m is None and _number(tags, "roof:levels"):
        roof_m = _number(tags, "roof:levels") * LEVEL_M
    roof = Roof(xy, str(tags.get("roof:shape", "flat")).strip().lower(), roof_m,
                tags.get("roof:direction"), tags.get("roof:orientation") == "across")
    roof.height = min(roof.height, h - base)
    colour = lambda key, material: _parse_colour(tags[key]) if key in tags else \
        (np.array(MATERIAL[tags[material]]) if tags.get(material) in MATERIAL else None)
    return Form(roof, base, "building:part" in tags, colour("building:colour", "building:material"),
                colour("roof:colour", "roof:material"), tags=dict(tags))


def foot_of(xy, form, ground):
    """An outline's foot height: settled (settle), else the lowest ground(xy) under it."""
    return form.foot_m if form.foot_m is not None else float(ground(xy).min())


def corners(outlines, spacing):
    """(points (n, 2), owner (n,)) along every outline, spacing(middle) apart: land corners so no triangle spans a wall."""
    import shapely
    pts, owner = [np.zeros((0, 2))], [np.zeros(0, int)]
    for i, (xy, *_) in enumerate(outlines):
        step = float(np.atleast_1d(spacing(xy[:-1].mean(0, keepdims=True)))[0])
        p = shapely.get_coordinates(shapely.segmentize(shapely.linearrings(xy), step))[:-1]
        pts.append(p)
        owner.append(np.full(len(p), i))
    return np.concatenate(pts), np.concatenate(owner)


def onto_scene(outlines, xy, h, owner, own, under):
    """Land heights h raised to DA3's ground round buildings with DA3 ground beside two or more corners.

    Corners within BESIDE_M of own (seams.SceneGround) take its height; the
    land round them rises to the median, falling off at CUT_SLOPE past PAD_M.
    Never where under (DA3's own ground) is true."""
    import shapely
    if not len(owner):
        return h
    dist, ground, _ = own.at(xy)
    tree, out = cKDTree(xy), h.copy()
    for i, (ring, _, _, form, *_) in enumerate(outlines):
        mine = np.flatnonzero(owner == i)
        if not len(mine) or form.base_m > 0:
            continue
        has = mine[dist[mine] <= BESIDE_M]
        if len(has) < 2:
            continue
        out[has] = ground[has]
        level = float(np.median(ground[has]))
        c = ring[:-1].mean(0)
        reach = np.linalg.norm(ring - c, axis=1).max() + PAD_M + CUT_REACH_M
        idx = np.asarray(tree.query_ball_point(c, reach), int)
        idx = idx[~np.isin(idx, has) & ~under[idx]]
        if not len(idx):
            continue
        d = shapely.distance(shapely.polygons(ring), shapely.points(xy[idx]))
        out[idx] = np.maximum(out[idx], level - np.maximum(d - PAD_M, 0) * CUT_SLOPE)
    return out


def settle(outlines, xy, h, owner):
    """Stand every outline on the land and cut the land to it; returns the new h.

    Foot (Form.foot_m) is the lowest land height at its corners; land within
    PAD_M is cut to it, rising at most CUT_SLOPE beyond (only ever lowered).
    Where a neighbour's cut goes lower, walls extend down (Form.skirt_m)."""
    import shapely
    if not len(owner):
        return h
    tree, out = cKDTree(xy), h.copy()
    for i, (ring, _, _, form, *_) in enumerate(outlines):
        mine = owner == i
        if mine.any():
            form.foot_m = float(h[mine].min())
    for _, _, _, form, *_ in outlines:
        if form.stand_on is not None and form.stand_on.foot_m is not None:
            form.foot_m = form.stand_on.foot_m
    for ring, _, _, form, *_ in outlines:
        if form.foot_m is None:
            continue
        c = ring[:-1].mean(0)
        reach = np.linalg.norm(ring - c, axis=1).max() + PAD_M + CUT_REACH_M
        idx = np.asarray(tree.query_ball_point(c, reach), int)
        if not len(idx):
            continue
        d = shapely.distance(shapely.polygons(ring), shapely.points(xy[idx]))
        out[idx] = np.minimum(out[idx], form.foot_m + np.maximum(d - PAD_M, 0) * CUT_SLOPE)
    for i, (_, _, _, form, *_) in enumerate(outlines):
        mine = owner == i
        if mine.any():
            form.skirt_m = max(0.0, form.foot_m - float(out[mine].min()))
    return out


def _rings(elements):
    """(element, closed ring) for each building or building:part outline."""
    for e in elements:
        tags = e.get("tags", {})
        if "building" not in tags and "building:part" not in tags:
            continue
        rings = [e.get("geometry")] if e["type"] == "way" else \
            [m.get("geometry") for m in e.get("members", []) if m.get("role") == "outer"]
        for ring in rings:
            if ring and len(ring) >= 4 and ring[0] == ring[-1]:   # outers split over several ways are skipped
                yield e, ring


def outlines(elements, to_xy):
    """[(closed outline (n, 2) east/north, height m, guessed, Form)]; a building with parts is replaced by them."""
    import shapely
    from shapely.geometry import Polygon
    road_xy = [to_xy(e["geometry"]) for e in elements if e.get("tags", {}).get("highway")
               and len(e.get("geometry", [])) >= 2]
    road_tree = cKDTree(np.concatenate(road_xy)) if road_xy else None
    out = []
    for e, ring in _rings(elements):
        xy = to_xy(ring)
        if turning(xy) < 0:                       # counter-clockwise, so normals face out
            xy = xy[::-1]
        h, guessed = _height(e["tags"])
        form = _form(e["tags"], xy, h)
        form.seed = int(e.get("id", 0))
        mid = (xy[:-1] + xy[1:]) / 2
        lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        candidates = np.flatnonzero(lengths >= 3)
        if len(candidates):
            form.front = int(candidates[np.argmin(road_tree.query(mid[candidates])[0])] if road_tree
                             else candidates[np.argmax(lengths[candidates])])
        out.append((xy, max(h, form.base_m + MIN_HEIGHT_M), guessed, form))
    # parts inherit style hints from their containing outline (an untyped church nave is not apartments)
    parents = [o for o in out if not o[3].part]
    if parents:
        parent_polys = [Polygon(o[0]).buffer(0) for o in parents]
        parent_tree = shapely.STRtree(parent_polys)
        for xy, _, _, form in out:
            if not form.part:
                continue
            candidates = parent_tree.query(Polygon(xy).buffer(0).representative_point(), predicate="within")
            if len(candidates):
                parent = parents[min(candidates, key=lambda k: parent_polys[k].area)][3]
                for key in ("building", "building:architecture", "start_date", "building:material", "building:use"):
                    if key in parent.tags:
                        form.tags.setdefault(key, parent.tags[key])
    parts = [Polygon(o[0]).buffer(0).representative_point() for o in out if o[3].part]
    if parts:
        tree = shapely.STRtree(parts)
        out = [o for o in out if o[3].part or not len(tree.query(Polygon(o[0]).buffer(0), "contains"))]
    # regional roofs and landmarks (styles.py)
    from streetview_to_3d.postprocess.world import styles
    ring = next((r for _, r in _rings(elements)), None)
    out = styles.apply(out, styles.region(ring[0]["lat"], ring[0]["lon"]) if ring else "other", Form)
    # shared walls get no balconies
    polygons = [Polygon(o[0]).buffer(0) for o in out]
    tree = shapely.STRtree(polygons)
    for owner, (xy, _, _, form) in enumerate(out):
        blocked = set()
        for side, (a, b) in enumerate(zip(xy[:-1], xy[1:])):
            length = np.linalg.norm(b - a)
            if length < 1e-6:
                continue
            t = (b - a) / length
            normal = np.array([t[1], -t[0]])
            probes = a + np.array([.2, .5, .8])[:, None] * (b - a) + normal * .75
            hits = tree.query(shapely.points(probes), predicate="within")
            if len(np.unique(hits[0, hits[1] != owner])) >= 2:
                blocked.add(side)
        form.shared_edges = frozenset(blocked)
    return out


def _parse_colour(text):
    """RGB 0-1 of an OSM colour tag ("#c8a060", "#ca6" or a name), or None."""
    t = str(text).strip().lower()
    if t.startswith("#") and len(t) in (4, 7):
        h = t[1:] if len(t) == 7 else "".join(ch * 2 for ch in t[1:])
        try:
            return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255
        except ValueError:
            return None
    return np.array(NAMED[t]) if t in NAMED else None


def soften(rgb):
    """rgb with lightness mapped into SOFT_LIGHT and saturation into SOFT_SAT, hue kept."""
    import colorsys
    h, l, s = colorsys.rgb_to_hls(*np.clip(rgb, 0, 1))
    l = SOFT_LIGHT[0] + (SOFT_LIGHT[1] - SOFT_LIGHT[0]) * l
    s = float(np.clip(s * 1.4, *SOFT_SAT))
    return np.array(colorsys.hls_to_rgb(h, l, s))


def palette(photos):
    """(colours (k, 3), shares (k,)): PALETTE_K softened clusters of the panos' building and wall pixels,
    or PASTEL if they have under PALETTE_MIN. photos: (image, class map, mask) or None."""
    from PIL import Image
    from scipy.cluster.vq import kmeans2
    from streetview_to_3d.models.segment import LABEL_IDS
    ids = [LABEL_IDS["building"], LABEL_IDS["wall"]]
    samples = []
    for ph in photos:
        if ph is None:
            continue
        img = np.asarray(Image.open(ph[0]).convert("RGB"))
        h, w = img.shape[:2]
        v, u = np.mgrid[0:h:PALETTE_STRIDE, 0:w:PALETTE_STRIDE]
        at = lambda grid: grid[(v * grid.shape[0] // h).ravel(), (u * grid.shape[1] // w).ravel()]
        ok = np.isin(at(ph[1]), ids) & ~at(ph[2])
        samples.append(img[v.ravel(), u.ravel()][ok] / 255)
    samples = np.concatenate(samples) if samples else np.zeros((0, 3))
    if len(samples) < PALETTE_MIN:
        return np.array(PASTEL), np.full(len(PASTEL), 1 / len(PASTEL))
    centres, label = kmeans2(samples, PALETTE_K, seed=0, minit="++")
    share = np.bincount(label, minlength=PALETTE_K) / len(label)
    return np.array([soften(c) for c in centres]), share


def satellite_roofs(outlines, colour_at, inset=SAT_INSET_M, step=SAT_STEP_M, lively=True):
    """Give roofs without a colour the satellite's median colour (inset, step apart; cheer'd if lively).

    colour_at(east/north (n, 2)) -> RGB. Returns how many roofs were coloured."""
    n = 0
    for xy, _, _, form, *_ in outlines:
        if form.roof_colour is not None:
            continue
        ring = _inset(xy, inset)
        if ring is None:
            continue
        lo, hi = ring.min(0), ring.max(0)
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0], step) + step / 2,
                             np.arange(lo[1], hi[1], step) + step / 2)
        grid = np.stack([gx.ravel(), gy.ravel()], 1)
        at = np.concatenate([grid[_inside(grid, ring)], ring[:-1]])
        with warnings.catch_warnings():     # all-NaN slices are skipped below
            warnings.simplefilter("ignore", RuntimeWarning)
            c = np.nanmedian(colour_at(at), 0)
        if not np.isfinite(c).all():        # the map has none there
            continue
        form.roof_colour = cheer(c) if lively else c
        n += 1
    return n


def colours(outlines, palette_):
    """(n, 3) building colours: Form.colour, else a palette colour picked by position, weighted by share."""
    cols, share = palette_
    cum = np.cumsum(share) / share.sum()
    out = np.empty((len(outlines), 3))
    for i, (xy, _, _, form, *_) in enumerate(outlines):
        tag = form.colour
        if tag is not None:
            out[i] = tag
            continue
        c = xy.mean(0)
        pick = hash01(c[0], c[1])
        out[i] = cols[min(int(np.searchsorted(cum, pick)), len(cols) - 1)]
    return out


def near_box(xz, centre, half):
    """Indices of the points within half of centre each way (xz: a cKDTree of east/north)."""
    half = np.broadcast_to(np.asarray(half, float), (2,))
    idx = np.asarray(xz.query_ball_point(centre, float(np.hypot(*half))), int)
    return idx[np.all(np.abs(xz.data[idx] - centre) <= half, axis=1)] if len(idx) else idx


def da3_copy(n, d, x, da3, da3_normals, xz):
    """(n2, d2, agrees): plane fitted to DA3's points near wall plane (n, d) through samples x, or None.

    Uses points within SAME_M of the plane, facing within SAME_DEG and within
    SAME_NEAR_M of x (at least SAME_MIN); agrees if within AGREE_DEG of n."""
    lo, hi = x.min(0) - SAME_NEAR_M, x.max(0) + SAME_NEAR_M
    cand = near_box(xz, (lo[[0, 2]] + hi[[0, 2]]) / 2, (hi[[0, 2]] - lo[[0, 2]]) / 2)
    cand = cand[(da3[cand, 1] >= lo[1]) & (da3[cand, 1] <= hi[1])]
    cand = cand[(np.abs(da3[cand] @ n - d) < SAME_M)
                & (np.abs(da3_normals[cand] @ n) > np.cos(np.radians(SAME_DEG)))]
    if len(cand) < SAME_MIN:
        return None
    near, _ = cKDTree(x).query(da3[cand], distance_upper_bound=SAME_NEAR_M)
    P = da3[cand[np.isfinite(near)]]
    if len(P) < SAME_MIN:
        return None
    c = P.mean(0)
    n2 = np.linalg.svd(P - c, full_matrices=False)[2][-1]
    n2 = n2 if n2 @ n > 0 else -n2
    return n2, float(n2 @ c), bool(n2 @ n >= np.cos(np.radians(AGREE_DEG)))


def on_plane(n2, d2, da3, da3_normals):
    """True for DA3 points within ON_WALL_M of plane (n2, d2) and facing its way."""
    return (np.abs(da3 @ n2 - d2) < ON_WALL_M) & (np.abs(da3_normals @ n2) > np.cos(np.radians(SAME_DEG)))


def stretches(a):
    """[(start, end)] along a wall where DA3 has dense points (SPAN_PER_M per m), gaps to BRIDGE_M closed."""
    steps, count = np.unique(np.floor(a / SPAN_M).astype(int), return_counts=True)
    steps = steps[count >= SPAN_PER_M * SPAN_M]
    if not len(steps):
        return []
    breaks = np.flatnonzero((np.diff(steps) - 1) * SPAN_M >= BRIDGE_M)   # empty steps between
    starts = np.r_[steps[0], steps[breaks + 1]]
    ends = np.r_[steps[breaks], steps[-1]] + 1
    return [(a0 * SPAN_M, a1 * SPAN_M) for a0, a1 in zip(starts, ends)]


def _wall_samples(a, c, base, top):
    """Points every WALL_SAMPLE_M over the wall from a to c, base to top."""
    length = float(np.linalg.norm(c - a))
    u = np.arange(0, length + 1e-9, WALL_SAMPLE_M)
    v = np.arange(base, top + 1e-9, WALL_SAMPLE_M)
    xy = a + (c - a) * (u / max(length, 1e-9))[:, None]
    return np.column_stack([np.repeat(xy[:, 0], len(v)), -np.tile(v, len(u)), np.repeat(xy[:, 1], len(v))])


def _walls(xy, base, h, da3, da3_normals, xz):
    """{edge index: (n2, d2, DA3 point indices on it)} for the walls of xy that DA3 has (da3_copy)."""
    walls = {}
    for j, (a, c) in enumerate(zip(xy[:-1], xy[1:])):
        length = float(np.linalg.norm(c - a))
        if length < 1e-6:
            continue
        t = (c - a) / length
        n = np.array([t[1], 0.0, -t[0]])                 # outward: outlines run counter-clockwise
        copy = da3_copy(n, float(n @ [a[0], 0, a[1]]), _wall_samples(a, c, base, base + h), da3, da3_normals, xz)
        if copy is None or not copy[2]:
            continue
        n2, d2, _ = copy
        idx = near_box(xz, (a + c) / 2, length / 2 + FIT_M)
        walls[j] = (n2, d2, idx[on_plane(n2, d2, da3[idx], da3_normals[idx])])
    return walls


def _trim(xy, walls, da3):
    """xy with what stands in front of DA3's walls cut away (DA3's wall is the real face), or None.

    Only in front of the stretches DA3 has, from TRIM_TOL_M out, by walls at
    least TRIM_EDGE_M long. None if the cut is under TRIM_MIN_M2 or over TRIM_MAX."""
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient
    from shapely.ops import unary_union
    poly = Polygon(xy).buffer(0)
    strips = []
    for j, (n2, _, on) in walls.items():
        if len(on) < 2 or np.linalg.norm(xy[j + 1] - xy[j]) < TRIM_EDGE_M:
            continue
        n = n2[[0, 2]] / np.linalg.norm(n2[[0, 2]])
        along = np.array([-n[1], n[0]])
        c = da3[on][:, [0, 2]].mean(0)
        for u0, u1 in stretches((da3[on][:, [0, 2]] - c) @ along):
            near, far = c + n * TRIM_TOL_M, c + n * TRIM_FRONT_M
            strips.append(Polygon([near + along * u0, near + along * u1, far + along * u1, far + along * u0]))
    if not strips:
        return None
    kept = poly.difference(unary_union(strips))
    if kept.geom_type != "Polygon":
        kept = max(getattr(kept, "geoms", []), key=lambda g: g.area, default=kept)
    removed = poly.area - kept.area
    if kept.is_empty or removed < TRIM_MIN_M2 or removed > TRIM_MAX * poly.area:
        return None
    return np.asarray(orient(kept, 1.0).exterior.coords)


def fit_to_scene(outlines, da3, da3_normals, ground, on_road=None):
    """(outlines, moved, trimmed): buildings slid onto the DA3 walls they have, then trimmed.

    One least-squares slide (no rotation) per building, skipped if over
    SNAP_MAX_M or if it moves the outline onto roads (on_road(xy): share on
    road). A guessed height becomes DA3's wall top (TOP_PERCENTILE). Each
    outline gains {edge: (n2, d2)} for its DA3 walls."""
    if not len(da3):
        return [o + ({},) for o in outlines], 0, 0
    lo, hi = da3[:, [0, 2]].min(0) - FIT_M, da3[:, [0, 2]].max(0) + FIT_M
    xz = cKDTree(da3[:, [0, 2]])
    out, moved, trimmed = [], 0, 0
    for xy, h, guessed, form in outlines:
        if form.part or not ((xy.max(0) >= lo) & (xy.min(0) <= hi)).all():
            out.append((xy, h, guessed, form, {}))
            continue
        base = foot_of(xy, form, ground)
        walls = _walls(xy, base, h, da3, da3_normals, xz)
        if not walls:
            out.append((xy, h, guessed, form, {}))
            continue
        a = xy[:-1]
        rows = np.array([n2[[0, 2]] for n2, _, _ in walls.values()])
        want = np.array([d2 - n2 @ [a[j][0], 0, a[j][1]] for j, (n2, d2, _) in walls.items()])
        w = np.sqrt([max(len(on), 1) for _, _, on in walls.values()])[:, None]
        shift = np.linalg.lstsq(rows * w, want * w[:, 0], rcond=None)[0]
        if np.linalg.norm(shift) > SNAP_MAX_M or (on_road and on_road(xy + shift) > on_road(xy) + ROAD_SLACK):
            shift = np.zeros(2)
        xy = xy + shift
        moved += bool(shift.any())
        cut = _trim(xy, walls, da3)
        if cut is not None:
            xy, trimmed = cut, trimmed + 1
            walls = _walls(xy, base, h, da3, da3_normals, xz)
        if shift.any() or cut is not None:
            form.roof = form.roof.on(xy)
        if guessed and walls:
            tops = np.concatenate([-da3[on, 1] for _, _, on in walls.values()])
            if len(tops):
                h = max(MIN_HEIGHT_M, np.percentile(tops, TOP_PERCENTILE) - base)
                form.roof.height = min(form.roof.height, h)
        out.append((xy, h, guessed, form, {j: (n2, d2) for j, (n2, d2, _) in walls.items()}))
    return out, moved, trimmed


def _inside(pts, ring):
    """True where pts (m, 2) fall inside the closed ring (n, 2)."""
    a, b = ring[:-1], ring[1:]
    x, y = pts[:, :1], pts[:, 1:]
    crosses = ((a[:, 1] > y) != (b[:, 1] > y)) & \
        (x < (b[:, 0] - a[:, 0]) * (y - a[:, 1]) / (b[:, 1] - a[:, 1] + 1e-12) + a[:, 0])
    return crosses.sum(1) % 2 == 1


class Blocks:
    """Every building's points with, per point: building (which), wall edge (edge; ROOF, INNER),
    u metres along and v metres up its wall, spacing (gap), tag colour kept (own), normal and kind
    (SURFACE or EDGE). edges[e]: (start, along, length, DA3 plane (n2, d2) or None)."""

    def __init__(self, pts, cols, which, edge, u, v, gap, edges, own, normal, kind):
        self.pts, self.cols, self.which = pts, cols, which
        self.edge, self.u, self.v, self.gap, self.edges = edge, u, v, gap, edges
        self.own, self.normal, self.kind = own, normal, kind

    def take(self, keep):
        for k in ("pts", "cols", "which", "edge", "u", "v", "gap", "own", "normal", "kind"):
            setattr(self, k, getattr(self, k)[keep])


def cheer(rgb):
    """A dull satellite roof colour made CHEER_SAT more saturated and CHEER_LIFT brighter."""
    rgb = np.asarray(rgb, float)
    grey = rgb.mean(-1, keepdims=True)
    return np.clip((grey + (rgb - grey) * CHEER_SAT) * CHEER_LIFT, 0, 1)


def _inset(xy, d):
    """Outline xy inset d metres, counter-clockwise, or None if nothing is left."""
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient
    p = Polygon(xy).buffer(-d, join_style="mitre")
    if p.is_empty:
        return None
    if p.geom_type != "Polygon":
        p = max(p.geoms, key=lambda g: g.area)
    return np.asarray(orient(p, 1.0).exterior.coords)


def points(outlines, spacing, ground, colour):
    """Blocks of points for every outline: roof, walls (in CHUNK_M pieces), inner walls and details.

    spacing(xy): point spacing; ground(xy): final land height; colour: (n, 3)
    per building. A part starting in the air gets a flat underside."""
    roofs, walls, edge_of, us, vs, gaps, roof_gaps, edges = [], [], [], [], [], [], [], []
    roof_normals, wall_normals, kinds = [], [], []
    for xy, h, _, form, planes in outlines:
        foot = foot_of(xy, form, ground)
        base, top = foot + form.base_m, foot + h
        low = base - (0.0 if form.base_m else form.skirt_m)      # wall bottom
        roof = form.roof
        eaves = max(base, top - roof.height)
        # roof, spaced as at its middle; an underside if it starts in the air
        s = float(spacing(xy.mean(0)[None])[0])
        rp, rn = roof.surface(s)
        rp[:, 2] += eaves
        if form.base_m > 0:
            under = Roof(xy, "flat").surface(s)[0]
            rp = np.concatenate([rp, np.c_[under[:, :2], np.full(len(under), base)]])
            rn = np.concatenate([rn, np.tile([0.0, 0.0, -1.0], (len(under), 1))])
        roofs.append(rp[:, [0, 2, 1]] * [1, -1, 1])
        roof_normals.append(rn[:, [0, 2, 1]] * [1, -1, 1])   # east, down, north
        roof_gaps.append(np.full(len(rp), s))
        # walls: along each edge, up to the roof
        w, eo, uu, vv, gg, nn, kk = [], [], [], [], [], [], []
        for j, (a, c) in enumerate(zip(xy[:-1], xy[1:])):
            length = float(np.linalg.norm(c - a))
            if length < 1e-6:
                continue
            t = (c - a) / length
            out = np.array([t[1], -t[0]])                    # outward for a counter-clockwise ring
            e = len(edges)
            edges.append((a, t, length, planes.get(j)))
            # pieces spaced as at their middle; ground under all columns at once
            pieces = []
            for c0 in np.arange(0, length, CHUNK_M):
                c1 = min(length, c0 + CHUNK_M)
                s = float(spacing((a + t * (c0 + c1) / 2)[None])[0])
                pieces.append((c0, s, np.arange(c0, c1, s)))
            columns = np.concatenate([p[2] for p in pieces])
            floors = ground(a + t * columns[:, None]) if not form.base_m else np.full(len(columns), low)
            for (c0, s, along), floor in zip(pieces, np.split(floors, np.cumsum([len(p[2]) for p in pieces])[:-1])):
                rise = roof.rise(a + t * along[:, None])
                # columns start at the ground (a part in the air: its base), rows aligned building-wide
                start = low - s * np.ceil(max(0.0, low - floor.min()) / s) if len(floor) else low
                levels = np.arange(start, eaves + (rise.max() if len(rise) else 0), s)
                U, V = np.repeat(along, len(levels)), np.tile(levels, len(along))
                ok = (V <= eaves + np.repeat(rise, len(levels)) + 1e-6) & \
                    (V >= np.repeat(floor, len(levels)) - ON_GROUND_M)
                U, V = U[ok], V[ok]
                kind = np.full(len(U), SURFACE)
                # corner and eaves points, as EDGE
                corner = np.arange(start, eaves + (rise[0] if len(rise) else 0), s) if c0 == 0 else np.zeros(0)
                corner = corner[corner >= (floor[0] if len(floor) else low) - ON_GROUND_M]
                top = eaves + rise
                U = np.r_[U, np.zeros(len(corner)), along]
                V = np.r_[V, corner, top]
                kind = np.r_[kind, np.full(len(corner) + len(along), EDGE)]
                w.append(np.column_stack([a[0] + t[0] * U, -V, a[1] + t[1] * U]))
                eo.append(np.full(len(U), e))
                uu.append(U)
                vv.append(V)
                gg.append(np.full(len(U), s))
                nn.append(np.tile([out[0], 0.0, out[1]], (len(U), 1)))
                kk.append(kind)
        # inner walls: further in, sparser
        for inset in INNER_M:
            ring = _inset(xy, inset)
            for a, c in zip(ring[:-1], ring[1:]) if ring is not None else ():
                length = float(np.linalg.norm(c - a))
                if length < 1e-6:
                    continue
                t = (c - a) / length
                U, V, G = [], [], []
                for c0 in np.arange(0, length, CHUNK_M):
                    c1 = min(length, c0 + CHUNK_M)
                    s = INNER_GAP * float(spacing((a + t * (c0 + c1) / 2)[None])[0])
                    along, levels = np.arange(c0, c1, s), np.arange(low, eaves, s)
                    U.append(np.repeat(along, len(levels)))
                    V.append(np.tile(levels, len(along)))
                    G.append(np.full(len(along) * len(levels), s))
                U, V, G = np.concatenate(U), np.concatenate(V), np.concatenate(G)
                if not form.base_m and len(U):                            # from the ground up
                    keep = V >= ground(a + t * U[:, None]) - ON_GROUND_M
                    U, V, G = U[keep], V[keep], G[keep]
                w.append(np.column_stack([a[0] + t[0] * U, -V, a[1] + t[1] * U]))
                eo.append(np.full(len(U), INNER))
                uu.append(np.zeros(len(U)))
                vv.append(V)
                gg.append(G)
                nn.append(np.tile([t[1], 0.0, -t[0]], (len(U), 1)))
                kk.append(np.full(len(U), SURFACE))
        walls.append(np.concatenate(w) if w else np.zeros((0, 3)))
        edge_of.append(np.concatenate(eo) if eo else np.zeros(0, int))
        us.append(np.concatenate(uu) if uu else np.zeros(0))
        vs.append(np.concatenate(vv) if vv else np.zeros(0))
        gaps.append(np.concatenate(gg) if gg else np.zeros(0))
        wall_normals.append(np.concatenate(nn) if nn else np.zeros((0, 3)))
        kinds.append(np.concatenate(kk) if kk else np.zeros(0, int))
    if not roofs:
        z = np.zeros((0, 3))
        return Blocks(z, z, np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0), [],
                      np.zeros(0, bool), z, np.zeros(0, int))
    pts, cols, which, edge, u, v, gap, own = [], [], [], [], [], [], [], []
    normal, kind = [], []
    for i, (roof, wall, eo, uu, vv, rg, gg) in enumerate(zip(roofs, walls, edge_of, us, vs, roof_gaps, gaps)):
        normal += [roof_normals[i], wall_normals[i]]
        kind += [np.full(len(roof), SURFACE), kinds[i]]
        mine = outlines[i][3].roof_colour
        pts += [roof, wall]
        cols += [np.tile(np.clip(colour[i] if mine is None else mine, 0, 1), (len(roof), 1)),
                 np.tile(colour[i], (len(wall), 1))]
        which.append(np.full(len(roof) + len(wall), i))
        edge += [np.full(len(roof), ROOF), eo.astype(int)]
        u += [np.zeros(len(roof)), uu]
        v += [-roof[:, 1], vv]
        gap += [rg, gg]
        own += [np.full(len(roof), mine is not None), np.zeros(len(wall), bool)]
    from streetview_to_3d.postprocess.world.facade_geometry import enabled
    for i, (xy, h, _, form, planes) in enumerate(outlines):
        step = float(max(spacing(xy).min(), .12))
        form.physical_facade = enabled(form, h, step)
        details = list(detail_quads(xy, h, form, ground, colour[i], planes))
        if not details:
            continue
        quads = np.array([q for q, _ in details], float)
        facing = np.cross(quads[:, 1] - quads[:, 0], quads[:, 3] - quads[:, 0])
        size = np.linalg.norm(facing, axis=1)
        flat = size > 1e-9                                          # skip zero-area quads
        q, at = sample_quads(quads[flat], step)
        facing = (facing[flat] / size[flat, None])[at]
        normal.append(facing)
        kind.append(np.full(len(q), SURFACE))
        pts.append(q)
        cols.append(np.array([np.broadcast_to(c, 3) for _, c in details], float)[flat][at])
        which.append(np.full(len(q), i))
        edge.append(np.full(len(q), ROOF))
        u.append(np.zeros(len(q)))
        v.append(-q[:, 1])
        gap.append(np.full(len(q), step))
        own.append(np.ones(len(q), bool))
    return Blocks(np.concatenate(pts), np.concatenate(cols), np.concatenate(which), np.concatenate(edge),
                  np.concatenate(u), np.concatenate(v), np.concatenate(gap), edges, np.concatenate(own),
                  np.concatenate(normal), np.concatenate(kind))


WINDOW_U, WINDOW_V = (0.3, 0.7), (0.3, 0.8)     # window extent within a bay, within a floor


def facade_layout(form, h):
    """(bay, floor) metres for a building's windows, or None if it has none (styles.windows)."""
    from streetview_to_3d.postprocess.world import styles
    many = styles.windows(form.tags)
    if many == "none":
        return None
    if many == "few":
        eaves = max(h - form.roof.height, 1.0)
        wall = max(eaves - form.base_m, 1.0)
        row = wall / np.ceil(wall / styles.FEW_ROW_M)
        return styles.FEW_BAY_M, float(eaves / max(1, round(eaves / row)))
    kind = building_kind(form.tags)
    if kind == "yes":
        kind = "office" if h >= 20 else "house" if h <= 10 and form.roof.poly.area < 180 else "yes"
    bay = 2.8 if kind in ("house", "detached", "terrace", "semidetached_house") else \
          2.4 if kind in ("office", "commercial") else \
          6.0 if kind in ("industrial", "warehouse", "shed", "garage", "garages") else 3.0
    levels = _number(form.tags, "building:levels")
    floor = np.clip((h - form.roof.height) / levels, 2.4, 5.0) if levels else LEVEL_M
    return bay, float(floor)


def glass(wall):
    """A window's colour in a wall of colour wall (..., 3)."""
    return np.clip(np.asarray(wall) * GLASS_OWN + GLASS, 0, 1)


def windows(blocks, outlines, ground):
    """Paint wall points that fall on windows with glass, where points are dense enough
    (two per bay and floor); buildings with physical facades are skipped."""
    wall = np.flatnonzero((blocks.edge >= 0) & (blocks.kind == SURFACE))
    wall = wall[np.argsort(blocks.which[wall], kind="stable")]           # each building's together, once
    bounds = np.searchsorted(blocks.which[wall], np.arange(len(outlines) + 1))
    for owner, (xy, h, _, form, *_) in enumerate(outlines):
        i = wall[bounds[owner]:bounds[owner + 1]]
        if not len(i) or form.physical_facade:
            continue
        layout = facade_layout(form, h)
        if layout is None:
            continue
        bay, floor = layout
        up = blocks.v[i] - foot_of(xy, form, ground)
        qu, qv = blocks.u[i] / bay % 1, up / floor % 1
        on = ((qu >= WINDOW_U[0]) & (qu <= WINDOW_U[1]) & (qv >= WINDOW_V[0]) & (qv <= WINDOW_V[1])
              & (up >= 0.5) & (blocks.gap[i] * 2 <= min(bay, floor)))
        blocks.cols[i[on]] = glass(blocks.cols[i[on]])


def detail_quads(xy, h, form, ground, colour, planes=None):
    """(quad (4, 3) world, colour) for physical facades (Form.physical_facade), roof trim and,
    without a facade, a simple entrance. Walls in planes (fitted to DA3) get none."""
    from streetview_to_3d.postprocess.world.facade_geometry import facade_quads
    foot = foot_of(xy, form, ground)
    physical = form.physical_facade
    if physical:
        yield from facade_quads(xy, h, form, foot, colour, planes)
    eaves = foot + h - form.roof.height
    world = lambda en, up: np.c_[en[:, 0], -np.broadcast_to(up, len(en)), en[:, 1]]
    kind = building_kind(form.tags)
    if kind in ("roof", "carport", "hut"):
        return
    for j, (a, b) in enumerate(zip(xy[:-1], xy[1:])):
        length = np.linalg.norm(b - a)
        if planes and j in planes:
            continue
        if length < .5:
            continue
        t = (b - a) / length
        outward = np.array([t[1], -t[0]])
        if form.roof.shape == "flat":
            # parapet face and inward top cap
            z = eaves
            outer = np.array([a, b])
            inner = outer - outward * .22
            yield np.r_[world(outer, z - .4), world(outer[::-1], z)], np.clip(colour * 1.12, 0, 1)
            yield np.r_[world(outer, z), world(inner[::-1], z)], np.clip(colour * 1.2, 0, 1)
        else:
            z = eaves + form.roof.rise(np.array([a, b]))
            outer = np.array([a, b]) - outward * .02
            yield np.r_[world(outer, z - .18), world(outer[::-1], z[::-1])], colour * .75
        if not physical and j == form.front and length >= 3 and not form.base_m and not form.part \
                and kind not in ("shed", "garage", "garages", "warehouse", "industrial"):
            mid = (a + b) / 2
            ends = np.array([mid - t * .65, mid + t * .65]) + outward * .06
            top = min(foot + 2.35, eaves - .25)
            if top <= foot + 1.5:
                continue
            yield np.r_[world(ends, foot + .05), world(ends[::-1], top)], colour * .38 + [.02, .03, .04]
            canopy = ends + outward * .4
            yield np.r_[world(ends, top + .12), world(canopy[::-1], top + .12)], np.clip(colour * 1.15, 0, 1)


def reachable(outlines, scene):
    """True for each outline with a scene point within REACH_M of its box (seen from above); only these get points."""
    out = np.zeros(len(outlines), bool)
    if not len(scene) or not outlines:
        return out
    xz = scene[:, [0, 2]]
    lo = xz.min(0)
    cell = np.floor((xz - lo) / REACH_M).astype(int)
    grid = np.zeros(cell.max(0) + 2, int)
    grid[cell[:, 0] + 1, cell[:, 1] + 1] = 1
    seen = grid.cumsum(0).cumsum(1)                     # occupied cells up to (i, j)
    top = np.array(seen.shape) - 2
    for k, (xy, *_) in enumerate(outlines):
        a = np.clip(np.floor((xy.min(0) - REACH_M - lo) / REACH_M).astype(int), 0, None)
        b = np.minimum(np.floor((xy.max(0) + REACH_M - lo) / REACH_M).astype(int), top)
        if (a <= b).all():
            out[k] = seen[b[0] + 1, b[1] + 1] - seen[a[0], b[1] + 1] - seen[b[0] + 1, a[1]] + seen[a[0], a[1]] > 0
    return out


def reached(blocks, tree, n):
    """True for each of n buildings with a DA3 point (tree, or None) within REACH_M."""
    out = np.zeros(n, bool)
    if tree is not None and len(blocks.pts):
        d = tree.query(blocks.pts[::SAMPLE], distance_upper_bound=REACH_M, workers=-1)[0]
        out[np.unique(blocks.which[::SAMPLE][d < REACH_M])] = True
    return out


def _wall_facade(form, h, u, v, glass_colour):
    """(n, 7) facade rows: metres along and up, bay, floor, window colour; NO_FACADE without windows."""
    layout = facade_layout(form, h)
    if layout is None:
        return np.full((len(u), 7), NO_FACADE)
    return np.c_[u, v, np.full(len(u), layout[0]), np.full(len(u), layout[1]), np.tile(glass_colour, (len(u), 1))]


def solid(outlines, ground, colour, spacing=None):
    """Every outline as a mesh: (vertices, colours, faces, facade (n, 7)[, spacing]).

    Walls from ground(xy) (or the base for a part in the air) up to the roof,
    the roof's triangles, and a flat underside for parts in the air. facade
    rows as _wall_facade, NO_FACADE on roofs. spacing: one value per outline."""
    V, C, F, U, G = [], [], [], [], []
    count = 0

    def add(pts, col, faces, facade):
        nonlocal count
        V.append(pts)
        C.append(np.broadcast_to(col, pts.shape))
        F.append(faces + count)
        U.append(facade)
        G.append(np.full(len(pts), spacing[i] if spacing is not None else 0.0))
        count += len(pts)

    world = lambda en, up: np.c_[en[:, 0], -up, en[:, 1]]
    for i, (xy, h, _, form, *_) in enumerate(outlines):
        foot = foot_of(xy, form, ground)
        base, top = foot + form.base_m, foot + h
        low = base - (0.0 if form.base_m else form.skirt_m)      # wall bottom
        roof = form.roof
        eaves = max(base, top - roof.height)
        ring = xy if turning(xy) > 0 else xy[::-1]       # counter-clockwise: inside to the left
        for a, c in zip(ring[:-1], ring[1:]):
            length = float(np.linalg.norm(c - a))
            if length < 1e-6:
                continue
            t = (c - a) / length
            # corners where the roof bends and every SOLID_STEP_M, so the foot follows the ground
            u = np.unique(np.r_[length * roof.bends(a, c), np.arange(0, length, SOLID_STEP_M), length])
            en = a + t * u[:, None]
            up = eaves + roof.rise(en)
            if (up - base).max() < 1e-3:
                continue
            bottom = np.minimum(ground(en), up) if not form.base_m else np.full(len(u), low)
            pts = np.concatenate([world(en, bottom), world(en, up)])
            k = np.arange(len(u) - 1)
            n = len(u)
            faces = np.concatenate([np.c_[k, k + 1, n + k + 1], np.c_[k, n + k + 1, n + k]])
            add(pts, colour[i], faces, _wall_facade(form, h, np.r_[u, u], np.r_[bottom, up] - foot,
                                                   glass(colour[i])))
        tris = roof.triangles(SOLID_STEP_M)
        if form.base_m > 0:
            under = Roof(xy, "flat").triangles(SOLID_STEP_M)
            under[..., 2] = base - eaves
            tris = np.concatenate([tris, under[:, ::-1]])
        normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        keep = np.linalg.norm(normal, axis=1) > 1e-9
        tris, normal = tris[keep], normal[keep] / np.linalg.norm(normal[keep], axis=1, keepdims=True)
        if not len(tris):
            continue
        roof_colour = colour[i] if form.roof_colour is None else form.roof_colour
        pts = tris.reshape(-1, 3)
        add(world(pts[:, :2], eaves + pts[:, 2]), np.clip(roof_colour, 0, 1),
            np.arange(len(pts)).reshape(-1, 3), np.full((len(pts), 7), NO_FACADE))
    if not V:
        empty = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 7))
        return empty if spacing is None else (*empty, np.zeros(0))
    # merge vertices identical in position, colour, facade and spacing
    rows = np.c_[np.concatenate(V), np.round(np.concatenate(C) * 255), np.concatenate(U),
                 np.concatenate(G)].astype(np.float32)
    rows, index = np.unique(rows, axis=0, return_inverse=True)
    out = rows[:, :3], rows[:, 3:6] / 255, index.ravel()[np.concatenate(F)], rows[:, 6:13]
    return out if spacing is None else (*out, rows[:, 13])


def seam(blocks, da3, da3_normals, da3_cols, roofs_near):
    """Drop building points DA3 already covers and fade the rest into DA3; returns how many were dropped.

    On walls with a DA3 plane, a point within COVER x its gap of a DA3 point
    on the plane (in wall coordinates) goes; within SEAM_FADE_M the rest are
    pulled onto the plane (at most PULL_MAX_M) and tinted toward DA3. Roof
    points go by roofs_near, their distance to DA3."""
    from scipy.spatial import cKDTree
    from streetview_to_3d.postprocess.seams import ramp
    keep = np.ones(len(blocks.pts), bool)
    roof = blocks.edge == ROOF
    keep[roof] = roofs_near[roof] > COVER * blocks.gap[roof]
    order = np.argsort(blocks.edge, kind="stable")
    bounds = np.searchsorted(blocks.edge[order], np.arange(len(blocks.edges) + 1))
    xz = cKDTree(da3[:, [0, 2]]) if len(da3) else None
    for e, (a, t, length, plane) in enumerate(blocks.edges):
        mine = order[bounds[e]:bounds[e + 1]]
        if plane is None or not len(mine) or xz is None:
            continue
        n2, d2 = plane
        idx = near_box(xz, a + t * length / 2, length / 2 + SEAM_FADE_M + 1)
        idx = idx[on_plane(n2, d2, da3[idx], da3_normals[idx])]
        if len(idx) < SEAM_MIN:
            continue
        u = (da3[idx][:, [0, 2]] - a) @ t
        d, k = cKDTree(np.c_[u, -da3[idx, 1]]).query(np.c_[blocks.u[mine], blocks.v[mine]])
        band = ramp(d / SEAM_FADE_M)
        keep[mine] = d > COVER * blocks.gap[mine]
        off = np.clip(d2 - blocks.pts[mine] @ n2, -PULL_MAX_M, PULL_MAX_M)
        blocks.pts[mine] += ((1 - band) * off)[:, None] * n2
        w = (SEAM_TINT * (1 - band))[:, None]
        blocks.cols[mine] = blocks.cols[mine] * (1 - w) + da3_cols[idx[k]] * w
    blocks.take(keep)
    return int(len(keep) - keep.sum())


def pano_colours(pts, which, n, cameras, photos, occluders):
    """(n, 3) each building's median colour in the panos, NaN where under SEEN_MIN pixels are seen.

    A point is seen by a camera within SEE_M if nothing is more than BEHIND_M
    (+ BEHIND x distance) in front of it (so DA3's own copy of the wall does
    not hide it) and its pixel is labelled building or wall."""
    from streetview_to_3d.postprocess.fill.paint import NADIR_DEG, _at, depth_buffer, pixel
    from streetview_to_3d.models.segment import LABEL_IDS
    ids = [LABEL_IDS["building"], LABEL_IDS["wall"]]
    pts, which = pts[::SAMPLE], which[::SAMPLE]
    everything = np.concatenate([occluders[::SAMPLE_HIDING], pts])
    samples, owners = [], []
    for cam, ph in zip(cameras, photos):
        if ph is None:
            continue
        near = depth_buffer(cam, everything)
        u, v, r, below = cam.look(pts)
        seen = (r < SEE_M) & (below < NADIR_DEG) & (r <= near[pixel(u, v)] + BEHIND_M + BEHIND * r)
        seen &= np.isin(_at(ph[1], u, v), ids) & ~_at(ph[2], u, v)
        if seen.any():
            from PIL import Image
            img = np.asarray(Image.open(ph[0]).convert("RGB"))
            samples.append(_at(img, u[seen], v[seen]) / 255.0)
            owners.append(which[seen])
    out = np.full((n, 3), np.nan)
    if not samples:
        return out
    samples, owners = np.concatenate(samples), np.concatenate(owners)
    order = np.argsort(owners, kind="stable")
    samples, owners = samples[order], owners[order]
    starts = np.flatnonzero(np.r_[True, owners[1:] != owners[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(owners)]):
        if b - a >= max(3, SEEN_MIN / SAMPLE):
            out[owners[a]] = np.median(samples[a:b], 0)
    return out
