"""The buildings around the scene, from OpenStreetMap, stood on the terrain.

Footprints from osm.py, each raised into a block of points:

  - height: its own "height" tag, else building:levels (and roof:levels)
    x LEVEL_M, else a guess by kind (DEFAULT_M; most buildings carry
    neither -- NTU: 1864 of 2628)
  - a building mapped in parts (OSM's "building:part": a church's nave,
    tower and spire, the Eiffel Tower's floors) is drawn as its parts and
    not its outline, each from its own min_height (or building:min_level)
    up, so they stack; parts are left where OSM has them (not fitted)
  - its roof as "roof:shape" has it (roofs.py), the walls up to where it
    starts (height less roof:height); if it has none, as its region
    builds (styles.py: a Japanese house's hipped, a Swedish one's gabled,
    a block's flat), and a landmark its kind says -- a castle, a temple,
    a church -- drawn as one, out of parts
  - one DA3 built a wall of is slid onto it, whatever still stands in
    front of that wall cut away as from a solid block, a guessed height
    made DA3's (fit_to_scene); what DA3 already has of it is left to DA3 and the rest
    faded in next to it, judged on the wall itself (seam)
  - standing where the land meets it, the land fitted to it as a game's
    terrain is (settle): its foot the lowest land along its outline, the
    land cut down to that round it -- level within PAD_M, then rising at
    most CUT_SLOPE -- so no wall is ever buried (the map's own ground, a
    30 m blur with the buildings in it, buried a third of them by over a
    metre); else, past the land, the lowest ground under its outline
  - walls and a flat roof, spaced by how near the scene's cameras they
    are, as the land is (terrain.gap_at), and the walls again further in,
    sparser (INNER_M), so it is not seen through
  - one colour per building: what the scene's panos see of it where they
    see enough (pano_colours), else its own "building:colour" tag (or
    building:material's, MATERIAL), else one
    of the place's own building colours, softened (palette) -- the
    satellite's, from 10 m up, came out grey-brown; walls shaded by which
    way they face, the roof a little lighter and shaded the same way,
    "roof:colour" (or roof:material's) if it has one, else what the
    satellite sees of it (satellite_roofs) -- never a pano's own pixel:
    from the street they see its edge against the sky

Away from the cameras a building is written solid instead (solid; far,
terrain.NEAR_M): a few dozen triangles, not thousands of points -- walls
and roof shaded as the points are, each vertex carrying its place on its
wall. The viewer draws them as points all the same, scattered over the
triangles, with floors of windows from that place (effects/scatter.js).
Near the points, windows the same way (windows).

Called by terrain.build, which writes them to buildings.ply and the solid
ones to blocks.ply.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np

from .geometry import sample_quad
from scipy.spatial import cKDTree

from streetview_to_3d.postprocess.roofs import Roof

LEVEL_M = 3.2
DEFAULT_M = {"house": 7, "detached": 7, "semidetached_house": 7, "terrace": 7, "bungalow": 5,
             "roof": 4, "shed": 3, "garage": 3, "garages": 3, "carport": 3, "hut": 3, "kiosk": 3}
DEFAULT_OTHER_M = 12.0
SEE_M = 150.0                 # panos colour the buildings this close to them
BEHIND_M, BEHIND = 3.0, 0.05  # how far behind what is in front a wall may stand and still be seen
SEEN_MIN = 20                 # pixels of a building the panos must see to colour it
REACH_M = 3.0                 # only buildings DA3 has points this near are coloured by the panos
BLEND_M = 1.0                 # a building's points this near DA3's turn into them: their colour, their look
LOCAL_K = 6                   # DA3's spacing somewhere: how far its LOCAL_K-th nearest point is
FIT_M = 5.0                   # buildings this far past the scene's points are not looked at
WALL_SAMPLE_M = 1.0
SNAP_MAX_M = 2.0              # OSM's real offsets were 0.2-1.3 m (Stockholm, NTU)
ROAD_SLACK = 0.02             # a slide may put this much more of an outline onto roads
PULL_MAX_M = 1.0              # a wall's seam pulled onto DA3's plane by at most this
TRIM_TOL_M, TRIM_FRONT_M = 0.3, 30.0   # cut from this far in front of a DA3 wall, out to this
TRIM_MIN_M2, TRIM_MAX = 1.0, 0.2
CHEER_SAT, CHEER_LIFT = 1.3, 1.15
SAT_INSET_M, SAT_STEP_M = 3.0, 4.0   # (Sentinel-2's) a roof sampled this far in from its edge (a 10 m pixel on it is
                                     # half street), this far apart
# the colour of a building no pano sees enough of: the place's own (palette)
PALETTE_K, PALETTE_MIN, PALETTE_STRIDE = 8, 500, 8
SOFT_LIGHT, SOFT_SAT = (0.62, 0.88), (0.15, 0.45)   # softened: the satellite's grey-brown looked dull
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
ROOF, INNER = -1, -2          # Blocks.edge for a roof's points and an inner wall's
SURFACE, EDGE = 0, 1          # Blocks.kind: as effects/blocks.js draws them
GLASS, GLASS_OWN = (0.22, 0.43, 0.64), 0.12   # a window's colour: blue glass, a little of its wall's
# DA3's own copy of a wall (da3_copy, on_plane, stretches): its points within SAME_M of the wall's
# plane, facing within SAME_DEG of its way, within SAME_NEAR_M of it, SAME_MIN at least; agreeing when
# its fitted plane faces within AGREE_DEG; on it within ON_WALL_M; along it in SPAN_M steps of
# SPAN_PER_M points a metre, gaps up to BRIDGE_M closed
SAME_DEG, SAME_M, SAME_NEAR_M, SAME_MIN = 40, 2.5, 5.0, 200
AGREE_DEG = 15
ON_WALL_M, SPAN_M, BRIDGE_M, SPAN_PER_M = 0.5, 0.2, 0.6, 60
INNER_M, INNER_GAP = (0.6, 1.5), 2.0   # walls again this far inside, this many times sparser: gaps in the
                                       # outer wall show more building, not through it (a solid box behind
                                       # the points looked wrong)
TRIM_EDGE_M = 5.0             # only a straight wall this long trims: a curve's short edges found planes in
                              # its own curved, overhung walls and ate NTU's Hive
TOP_PERCENTILE = 97
SOLID_STEP_M = 2.0            # a hipped roof's triangles about this long, at least
NO_FACADE = -1e4              # a roof's place on a wall: none (a skirt's is below 0 too)
MIN_HEIGHT_M = 2.5
PAD_M, CUT_SLOPE = 1.0, 0.5   # the land level with a building's foot this far round it, then rising at most this
CUT_REACH_M = 20.0            # ... looked at this far out: the map's bumps are a few metres
CHUNK_M = 4.0                 # a wall is spaced in pieces this long, each as at its middle
ROOF_MIN_STEP_M = 0.5         # roofs are seen from above only
COVER = 0.75                  # an OSM point DA3 has a point within this much of its gap of is DA3's
SEAM_FADE_M = 3.0             # the seam's width
SEAM_TINT, SEAM_MIN = 0.8, 20


def _number(tags, key):
    try:
        v = float(str(tags[key]).split()[0].replace(",", "."))
        return v if v >= 0 else None
    except (KeyError, ValueError, IndexError):
        return None


def _height(tags):
    """(height m, whether it is only a guess)."""
    h = _number(tags, "height")
    if h:
        return h, False
    levels = _number(tags, "building:levels")
    if levels:
        return (levels + (_number(tags, "roof:levels") or 0)) * LEVEL_M, False
    return DEFAULT_M.get(tags.get("building", tags.get("building:part")), DEFAULT_OTHER_M), True


@dataclass
class Form:
    """What an outline stands as besides its height: its roof (roofs.Roof),
    where it starts above its ground (min_height), whether it is a part,
    and its own colours from its tags (None: none)."""
    roof: Roof
    base_m: float = 0.0
    part: bool = False
    colour: object = None
    roof_colour: object = None
    foot_m: object = None         # the height it stands at, once settle has fitted the land to it
    tags: dict = field(default_factory=dict)
    front: int = 0
    seed: int = 0
    shared_edges: frozenset = frozenset()
    geometry_cache: dict = field(default_factory=dict, repr=False)
    physical_facade: bool = False
    skirt_m: float = 0.0          # ... its walls on down this much further, to the land a neighbour cut lower
    stand_on: object = None       # a part standing on another's foot (styles: a castle's tiers on its base)


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
    """The height an outline stands at: its settled foot (settle), else
    the lowest ground under it (ground(xy))."""
    return form.foot_m if form.foot_m is not None else float(ground(xy).min())


def corners(outlines, spacing):
    """(points (n, 2), owner (n,)): along every outline, spacing(xy) of
    its middle apart -- the land's corners, so no triangle spans a wall --
    and which outline each is on."""
    import shapely
    pts, owner = [np.zeros((0, 2))], [np.zeros(0, int)]
    for i, (xy, *_) in enumerate(outlines):
        step = float(np.atleast_1d(spacing(xy[:-1].mean(0, keepdims=True)))[0])
        p = shapely.get_coordinates(shapely.segmentize(shapely.linearrings(xy), step))[:-1]
        pts.append(p)
        owner.append(np.full(len(p), i))
    return np.concatenate(pts), np.concatenate(owner)


def settle(outlines, xy, h, owner):
    """Every outline stood on the land, the land fitted to it: its foot
    (Form.foot_m) the lowest of the land's heights h at its own corners
    (xy[owner == i], corners), and the land within PAD_M of it no higher,
    rising at most CUT_SLOPE beyond -- only ever cut down, so what lies on
    the land (roads) stays over it. Feet from the land as it was: a
    terrace's houses step down a slope each on its own. Where a lower
    one's cut reaches a higher one, the higher one's walls go on down to
    it (Form.skirt_m), as a game's foundations do: never floating, its
    floors counted from its own foot all the same. Returns the land's h."""
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
    """(element, ring) for each building or building part outline among
    osm.fetch's elements."""
    for e in elements:
        tags = e.get("tags", {})
        if "building" not in tags and "building:part" not in tags:
            continue
        rings = [e.get("geometry")] if e["type"] == "way" else \
            [m.get("geometry") for m in e.get("members", []) if m.get("role") == "outer"]
        for ring in rings:
            if ring and len(ring) >= 4 and ring[0] == ring[-1]:   # a relation's outer split over ways: left out
                yield e, ring


def outlines(elements, to_xy):
    """[(outline (n, 2) east/north metres, closed, height m, guessed, Form)]
    of the buildings and building parts among osm.fetch's elements, a
    building with parts left out for them; to_xy(lat, lon) -> east,
    north."""
    import shapely
    from shapely.geometry import Polygon
    road_xy = [to_xy(e["geometry"]) for e in elements if e.get("tags", {}).get("highway")
               and len(e.get("geometry", [])) >= 2]
    road_tree = cKDTree(np.concatenate(road_xy)) if road_xy else None
    out = []
    for e, ring in _rings(elements):
        xy = to_xy(ring)
        if (xy[:-1, 0] * xy[1:, 1] - xy[1:, 0] * xy[:-1, 1]).sum() < 0:   # counter-clockwise: normals face out
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
    # Parts inherit semantic hints from their containing outline, without replacing
    # their own heights/roofs/materials. A church's untyped nave must not become apartments.
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
    # as the place builds: its region's roofs, landmarks as theirs (styles.py)
    from streetview_to_3d.postprocess import styles
    ring = next((r for _, r in _rings(elements)), None)
    out = styles.apply(out, styles.region(ring[0]["lat"], ring[0]["lon"]) if ring else "other", Form)
    # Shared walls should not sprout balconies inside adjacent buildings.
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
    """RGB 0-1 of an OSM colour tag ("#c8a060", "#ca6", or a plain name),
    or None."""
    t = str(text).strip().lower()
    if t.startswith("#") and len(t) in (4, 7):
        h = t[1:] if len(t) == 7 else "".join(ch * 2 for ch in t[1:])
        try:
            return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255
        except ValueError:
            return None
    return np.array(NAMED[t]) if t in NAMED else None


def soften(rgb):
    """rgb made soft and bright, the paint style's: lightness into
    SOFT_LIGHT, saturation into SOFT_SAT, its hue kept."""
    import colorsys
    h, l, s = colorsys.rgb_to_hls(*np.clip(rgb, 0, 1))
    l = SOFT_LIGHT[0] + (SOFT_LIGHT[1] - SOFT_LIGHT[0]) * l
    s = float(np.clip(s * 1.4, *SOFT_SAT))
    return np.array(colorsys.hls_to_rgb(h, l, s))


def palette(photos):
    """(colours (k, 3), shares (k,)): the place's own building colours,
    softened -- PALETTE_K groups of what the scene's panos label building
    or wall -- or PASTEL, evenly, if they see under PALETTE_MIN such
    pixels. photos: as pano_colours has them (image, class map, mask)."""
    from PIL import Image
    from scipy.cluster.vq import kmeans2
    from streetview_to_3d.services.segment import LABEL_IDS
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
    """Each roof without its own colour (Form.roof_colour) given the
    satellite's: the median over it, inset in from its edge, sampled step
    apart, made livelier (cheer) if lively -- a 10 m pixel's is dull, a
    sharp one's is the roof's own; colour_at(east/north (n, 2)) -> RGB. A
    roof too small for that keeps its building's colour. Returns how many."""
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
        with warnings.catch_warnings():     # all NaN: warned, then skipped below
            warnings.simplefilter("ignore", RuntimeWarning)
            c = np.nanmedian(colour_at(at), 0)
        if not np.isfinite(c).all():        # the map has none there
            continue
        form.roof_colour = cheer(c) if lively else c
        n += 1
    return n


def colours(outlines, palette_):
    """(n, 3) each building's colour: its own tags' (Form.colour), else one
    of the palette's, picked by where it stands (the same every run) as
    often as that colour is among the place's buildings."""
    cols, share = palette_
    cum = np.cumsum(share) / share.sum()
    out = np.empty((len(outlines), 3))
    for i, (xy, _, _, form, *_) in enumerate(outlines):
        tag = form.colour
        if tag is not None:
            out[i] = tag
            continue
        c = xy.mean(0)
        pick = (np.sin(c[0] * 12.9898 + c[1] * 78.233) * 43758.5453) % 1.0
        out[i] = cols[min(int(np.searchsorted(cum, pick)), len(cols) - 1)]
    return out


def da3_copy(n, d, x, da3, da3_normals):
    """DA3's own copy of the wall plane (n, d) that the points x lie on:
    (n2, d2, agrees) fitted to DA3's points within SAME_M of it, facing
    within SAME_DEG of its way and within SAME_NEAR_M of x -- trees face
    every way and the ground up, so neither counts -- agrees when it faces
    within AGREE_DEG of n; None if DA3 has under SAME_MIN such points."""
    lo, hi = x.min(0) - SAME_NEAR_M, x.max(0) + SAME_NEAR_M
    cand = np.flatnonzero(np.all((da3 >= lo) & (da3 <= hi), axis=1))
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
    """True for DA3's points on the plane (n2, d2): within ON_WALL_M of it,
    facing its way."""
    return (np.abs(da3 @ n2 - d2) < ON_WALL_M) & (np.abs(da3_normals @ n2) > np.cos(np.radians(SAME_DEG)))


def stretches(a):
    """[(start, end), ...] along a wall where DA3 has it: SPAN_M steps with
    at least SPAN_PER_M points per metre, gaps up to BRIDGE_M closed."""
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


def _walls(xy, base, h, da3, da3_normals):
    """{edge index: (n2, d2, DA3's points on that plane)} for the walls of
    outline xy DA3 has a copy of (da3_copy: DA3's points near
    its plane that face its way, so trees, poles and the ground never
    count), n2 facing out of the building."""
    walls = {}
    for j, (a, c) in enumerate(zip(xy[:-1], xy[1:])):
        length = float(np.linalg.norm(c - a))
        if length < 1e-6:
            continue
        t = (c - a) / length
        n = np.array([t[1], 0.0, -t[0]])                 # outward: outlines run counter-clockwise
        copy = da3_copy(n, float(n @ [a[0], 0, a[1]]), _wall_samples(a, c, base, base + h), da3, da3_normals)
        if copy is None or not copy[2]:
            continue
        n2, d2, _ = copy
        near = np.all(np.abs(da3[:, [0, 2]] - (a + c) / 2) < length / 2 + FIT_M, axis=1)
        idx = np.flatnonzero(near)
        walls[j] = (n2, d2, idx[on_plane(n2, d2, da3[idx], da3_normals[idx])])
    return walls


def _trim(xy, walls, da3):
    """xy with whatever stands in front of a DA3 wall cut away, as a solid
    block sliced straight down (the cut face a new wall, the outline
    re-drawn round it): DA3's wall is the building's real face, and the
    camera saw it, so nothing of the building stands between them. Only in
    front of the stretches along it where DA3 has that wall
    (stretches) -- its plane runs on past them, maybe through
    another wing -- from TRIM_TOL_M out, past DA3's own noise, and only by
    straight walls (TRIM_EDGE_M). None
    when that cuts under TRIM_MIN_M2 or over TRIM_MAX of it."""
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
    """outlines, each building DA3 built a wall of moved onto it, trimmed
    to it, and where its height was a guess given DA3's; each outline
    gains {edge: (n2, d2)}, DA3's plane for each of its walls DA3 has.

    Each wall looks for DA3's copy of it (_walls). A copy says how far
    that wall is off, along its own facing; one slide of the whole
    building (never turned: OSM's shape is good, its place a metre or two
    off) meets all of them in least squares -- with one wall, or parallel
    ones, it only moves towards them, never sideways along them. Not moved
    if that is over SNAP_MAX_M, or if it moves the outline further onto a
    road (on_road(xy): the share of it there) -- buildings do not stand in
    roads. Then whatever of it still stands in front of a DA3 wall is cut
    away (_trim). A guessed height becomes the top of DA3's points on its
    walls (TOP_PERCENTILE). Returns the new outlines, how many moved and
    how many were trimmed."""
    if not len(da3):
        return [o + ({},) for o in outlines], 0, 0
    lo, hi = da3[:, [0, 2]].min(0) - FIT_M, da3[:, [0, 2]].max(0) + FIT_M
    out, moved, trimmed = [], 0, 0
    for xy, h, guessed, form in outlines:
        if form.part or not ((xy.max(0) >= lo) & (xy.min(0) <= hi)).all():
            out.append((xy, h, guessed, form, {}))
            continue
        base = foot_of(xy, form, ground)
        walls = _walls(xy, base, h, da3, da3_normals)
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
            walls = _walls(xy, base, h, da3, da3_normals)
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
    """Every building's points, and for each where on its building it is:
    which building, which wall edge (ROOF on a roof, INNER on an inner wall)
    and its place on that wall, u metres along and v metres up (height
    above sea level), how far it is from its neighbours (gap) and whether
    it keeps its own colour (own: roof:colour). Its colour is its own,
    unlit: the viewer lights it. edges[e]: (start, along, outward, length,
    DA3's plane for it (n2, d2) or None). For the viewer, which draws each as a brush
    stroke (effects/blocks.js): which way it faces (normal), which way its stroke goes
    (along: a wall's way, down a roof's slope, up a corner) and what it is (kind:
    SURFACE, or EDGE: a corner's or the eaves')."""

    def __init__(self, pts, cols, which, edge, u, v, gap, edges, own, normal, along, kind):
        self.pts, self.cols, self.which = pts, cols, which
        self.edge, self.u, self.v, self.gap, self.edges = edge, u, v, gap, edges
        self.own, self.normal, self.along, self.kind = own, normal, along, kind

    def take(self, keep):
        for k in ("pts", "cols", "which", "edge", "u", "v", "gap", "own", "normal", "along", "kind"):
            setattr(self, k, getattr(self, k)[keep])


def cheer(rgb):
    """The satellite's colour of a roof livelier: from above and
    in the shade they come out dull -- CHEER_SAT more saturated, CHEER_LIFT
    brighter."""
    rgb = np.asarray(rgb, float)
    grey = rgb.mean(-1, keepdims=True)
    return np.clip((grey + (rgb - grey) * CHEER_SAT) * CHEER_LIFT, 0, 1)


def _inset(xy, d):
    """Outline xy moved d metres in, counter-clockwise, or None if nothing
    of it is left."""
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient
    p = Polygon(xy).buffer(-d, join_style="mitre")
    if p.is_empty:
        return None
    if p.geom_type != "Polygon":
        p = max(p.geoms, key=lambda g: g.area)
    return np.asarray(orient(p, 1.0).exterior.coords)


def points(outlines, spacing, ground, colour):
    """Blocks for every outline.

    spacing(xy): point spacing at east/north points -- a wall is laid in
    pieces up to CHUNK_M long, each spaced as at its middle, so only what
    is near gets dense; ground(xy): the ground's height there; colour:
    (n, 3) each building's (colours), as it is: the viewer lights it. Each
    stands from its Form's base_m over its ground up to the roof, the roof
    (roofs.py) up to its height, the walls reaching up to meet it (a
    gable's end); a part that starts in the air has a flat underside."""
    roofs, walls, edge_of, us, vs, gaps, roof_gaps, edges = [], [], [], [], [], [], [], []
    roof_ways, wall_ways, kinds = [], [], []      # (normal, along) a point each; a wall's points' kinds
    for xy, h, _, form, planes in outlines:
        foot = foot_of(xy, form, ground)
        base, top = foot + form.base_m, foot + h
        low = base - (0.0 if form.base_m else form.skirt_m)      # its walls' bottom
        roof = form.roof
        eaves = max(base, top - roof.height)
        # roof, spaced as at its middle; an underside if it starts in the air
        s = float(max(spacing(xy.mean(0)[None])[0], ROOF_MIN_STEP_M))
        rp, rn = roof.surface(s)
        rp[:, 2] += eaves
        if form.base_m > 0:
            under = Roof(xy, "flat").surface(s)[0]
            rp = np.concatenate([rp, np.c_[under[:, :2], np.full(len(under), base)]])
            rn = np.concatenate([rn, np.tile([0.0, 0.0, -1.0], (len(under), 1))])
        roofs.append(rp[:, [0, 2, 1]] * [1, -1, 1])
        roof_ways.append(_roof_ways(rn))
        roof_gaps.append(np.full(len(rp), s))
        # walls: along each edge and up it, to where the roof meets it
        w, eo, uu, vv, gg, nn, aa, kk = [], [], [], [], [], [], [], []
        for j, (a, c) in enumerate(zip(xy[:-1], xy[1:])):
            length = float(np.linalg.norm(c - a))
            if length < 1e-6:
                continue
            t = (c - a) / length
            out = np.array([t[1], -t[0]])                    # outward for a counter-clockwise ring
            e = len(edges)
            edges.append((a, t, out, length, planes.get(j)))
            for c0 in np.arange(0, length, CHUNK_M):
                c1 = min(length, c0 + CHUNK_M)
                s = float(spacing((a + t * (c0 + c1) / 2)[None])[0])
                along = np.arange(c0, c1, s)
                rise = roof.rise(a + t * along[:, None])
                levels = np.arange(low, eaves + (rise.max() if len(rise) else 0), s)
                U, V = np.repeat(along, len(levels)), np.tile(levels, len(along))
                ok = V <= eaves + np.repeat(rise, len(levels)) + 1e-6
                U, V = U[ok], V[ok]
                kind = np.full(len(U), SURFACE)
                # its edges, as strokes of their own: its corner (where it starts) and the eaves
                corner = np.arange(low, eaves + (rise[0] if len(rise) else 0), s) if c0 == 0 else np.zeros(0)
                top = eaves + rise
                U = np.r_[U, np.zeros(len(corner)), along]
                V = np.r_[V, corner, top]
                kind = np.r_[kind, np.full(len(corner) + len(along), EDGE)]
                way = np.tile([t[0], 0.0, t[1]], (len(U), 1))
                way[len(U) - len(corner) - len(along):len(U) - len(along)] = [0.0, -1.0, 0.0]   # up the corner
                w.append(np.column_stack([a[0] + t[0] * U, -V, a[1] + t[1] * U]))
                eo.append(np.full(len(U), e))
                uu.append(U)
                vv.append(V)
                gg.append(np.full(len(U), s))
                nn.append(np.tile([out[0], 0.0, out[1]], (len(U), 1)))
                aa.append(way)
                kk.append(kind)
        # inner walls: the same, further in, sparser; not the building's own edges
        for inset in INNER_M:
            ring = _inset(xy, inset)
            for a, c in zip(ring[:-1], ring[1:]) if ring is not None else ():
                length = float(np.linalg.norm(c - a))
                if length < 1e-6:
                    continue
                t = (c - a) / length
                for c0 in np.arange(0, length, CHUNK_M):
                    c1 = min(length, c0 + CHUNK_M)
                    s = INNER_GAP * float(spacing((a + t * (c0 + c1) / 2)[None])[0])
                    along, levels = np.arange(c0, c1, s), np.arange(low, eaves, s)
                    U, V = np.repeat(along, len(levels)), np.tile(levels, len(along))
                    w.append(np.column_stack([a[0] + t[0] * U, -V, a[1] + t[1] * U]))
                    eo.append(np.full(len(U), INNER))
                    uu.append(np.zeros(len(U)))
                    vv.append(V)
                    gg.append(np.full(len(U), s))
                    nn.append(np.tile([t[1], 0.0, -t[0]], (len(U), 1)))
                    aa.append(np.tile([t[0], 0.0, t[1]], (len(U), 1)))
                    kk.append(np.full(len(U), SURFACE))
        walls.append(np.concatenate(w) if w else np.zeros((0, 3)))
        edge_of.append(np.concatenate(eo) if eo else np.zeros(0, int))
        us.append(np.concatenate(uu) if uu else np.zeros(0))
        vs.append(np.concatenate(vv) if vv else np.zeros(0))
        gaps.append(np.concatenate(gg) if gg else np.zeros(0))
        wall_ways.append((np.concatenate(nn), np.concatenate(aa)) if nn else (np.zeros((0, 3)), np.zeros((0, 3))))
        kinds.append(np.concatenate(kk) if kk else np.zeros(0, int))
    if not roofs:
        z = np.zeros((0, 3))
        return Blocks(z, z, np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0), [],
                      np.zeros(0, bool), z, z, np.zeros(0, int))
    pts, cols, which, edge, u, v, gap, own = [], [], [], [], [], [], [], []
    normal, way, kind = [], [], []
    for i, (roof, wall, eo, uu, vv, rg, gg) in enumerate(zip(roofs, walls, edge_of, us, vs, roof_gaps, gaps)):
        normal += [roof_ways[i][0], wall_ways[i][0]]
        way += [roof_ways[i][1], wall_ways[i][1]]
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
    for i, (xy, h, _, form, planes) in enumerate(outlines):
        step = float(max(spacing(xy).min(), .12))
        from streetview_to_3d.postprocess.facade_geometry import enabled
        form.physical_facade = enabled(form, h, step)
        for quad, col in detail_quads(xy, h, form, ground, colour[i], planes, gap=step):
            q = sample_quad(quad, step)
            facing = np.cross(quad[1] - quad[0], quad[-1] - quad[0])
            side = quad[1] - quad[0]
            normal.append(np.tile(facing / max(np.linalg.norm(facing), 1e-9), (len(q), 1)))
            way.append(np.tile(side / max(np.linalg.norm(side), 1e-9), (len(q), 1)))
            kind.append(np.full(len(q), SURFACE))
            pts.append(q)
            cols.append(np.tile(col, (len(q), 1)))
            which.append(np.full(len(q), i))
            edge.append(np.full(len(q), ROOF))
            u.append(np.zeros(len(q)))
            v.append(-q[:, 1])
            gap.append(np.full(len(q), step))
            own.append(np.ones(len(q), bool))
    return Blocks(np.concatenate(pts), np.concatenate(cols), np.concatenate(which), np.concatenate(edge),
                  np.concatenate(u), np.concatenate(v), np.concatenate(gap), edges, np.concatenate(own),
                  np.concatenate(normal), np.concatenate(way), np.concatenate(kind))


def _roof_ways(n):
    """A roof's points' (normals, along) in Blocks' frame (east, down, north),
    from its normals (east, north, up): along down its slope, a flat one's east."""
    normal = n[:, [0, 2, 1]] * [1, -1, 1]
    down = np.c_[n[:, 0], np.zeros(len(n)), n[:, 1]]                  # downhill on the map
    flat = np.linalg.norm(down, axis=1) < 0.05
    down[flat] = [1.0, 0.0, 0.0]
    down -= normal * (down * normal).sum(1, keepdims=True)             # on the roof
    return normal, down / np.maximum(np.linalg.norm(down, axis=1, keepdims=True), 1e-9)


FLOOR_M, BAY_M = LEVEL_M, 3.0       # windows: one row a floor, one a bay (as effects/blocks.js)
WINDOW_U, WINDOW_V = (0.3, 0.7), (0.3, 0.8)     # of a bay, of a floor


def facade_layout(form, h):
    """Actual metres per bay/floor, constrained by tagged storeys and building
    kind; None for a building with no windows (styles.windows). One with few
    has tall rows (each styles.FEW_ROW_M at most), styles.FEW_BAY_M apart,
    the last ending at its eaves -- a part up in the air: rows on its walls."""
    from streetview_to_3d.postprocess import styles
    many = styles.windows(form.tags)
    if many == "none":
        return None
    if many == "few":
        eaves = max(h - form.roof.height, 1.0)
        wall = max(eaves - form.base_m, 1.0)
        row = wall / np.ceil(wall / styles.FEW_ROW_M)
        return styles.FEW_BAY_M, float(eaves / max(1, round(eaves / row)))
    kind = form.tags.get("building", form.tags.get("building:part", "yes"))
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
    """A wall's points on its windows its glass's colour (glass): a window a
    bay and a floor, over the ground's first half metre, where points are
    close enough to tell them (two to a bay and a floor) -- as the viewer
    lays the far buildings' (effects/blocks.js, from blocks.ply's facade).
    Nothing of their own: the wall's points that fall on them."""
    for owner, (xy, h, _, form, *_) in enumerate(outlines):
        i = np.flatnonzero((blocks.edge >= 0) & (blocks.which == owner) & (blocks.kind == SURFACE))
        if not len(i) or form.physical_facade:
            continue  # real window modules, without a second painted window grid
        layout = facade_layout(form, h)
        if layout is None:
            continue
        bay, floor = layout
        up = blocks.v[i] - foot_of(xy, form, ground)
        qu, qv = blocks.u[i] / bay % 1, up / floor % 1
        on = ((qu >= WINDOW_U[0]) & (qu <= WINDOW_U[1]) & (qv >= WINDOW_V[0]) & (qv <= WINDOW_V[1])
              & (up >= 0.5) & (blocks.gap[i] * 2 <= min(bay, floor)))
        blocks.cols[i[on]] = glass(blocks.cols[i[on]])


def detail_quads(xy, h, form, ground, colour, planes=None, gap=2.5):
    """Physical facades, roof trim and a cheap entrance fallback at coarse spacing.

    Returns (quad world positions, colour). Roof trim stays within the footprint,
    avoiding overhang into neighbours. Airborne parts and utility buildings omit doors.
    """
    from streetview_to_3d.postprocess.facade_geometry import enabled, facade_quads
    foot = foot_of(xy, form, ground)
    physical = enabled(form, h, gap)
    if physical:
        yield from facade_quads(xy, h, form, foot, colour, planes)
    eaves = foot + h - form.roof.height
    world = lambda en, up: np.c_[en[:, 0], -np.broadcast_to(up, len(en)), en[:, 1]]
    kind = form.tags.get("building", form.tags.get("building:part", "yes"))
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
            # A visible parapet face and inward top cap, no change to mapped height.
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


def toward(pts, cols, gap, tree, da3_cols, every=1):
    """(cols, near, keep): points pts (spaced gap) turning into DA3's as they
    come within BLEND_M of them -- near 0 that far off or more, 1 on one --
    their colour mixed toward their nearest DA3 point's by it, and as few
    of them kept as DA3's are there (keep: each its own chance, fixed in
    the world, from all of them far off to as sparse as DA3's points on
    one; never more than were). tree: a cKDTree of DA3's points (every
    every-th of them), da3_cols their colours. The viewer turns their look
    into DA3's points' by near too (effects/blocks.js)."""
    from streetview_to_3d.postprocess.seams import ramp
    if tree is None or not len(pts):
        return cols, np.zeros(len(pts)), np.ones(len(pts), bool)
    d, k = tree.query(pts, distance_upper_bound=BLEND_M)
    near = 1 - ramp(d / BLEND_M)                               # d is inf past BLEND_M: 0
    k = np.minimum(k, len(da3_cols) - 1)
    cols = cols + (da3_cols[k] - cols) * near[:, None]
    # DA3's spacing round its nearest point: its LOCAL_K-th neighbour's distance, as
    # if all its points were there, not every every-th
    keep = np.ones(len(pts), bool)
    close = np.flatnonzero(near > 0)
    if len(close):
        far = tree.query(tree.data[k[close]], k=LOCAL_K + 1)[0][:, -1]
        spacing = far / np.sqrt(LOCAL_K / np.pi) / np.sqrt(every)
        share = np.minimum(1, (gap[close] / np.maximum(spacing, 1e-6)) ** 2)    # of these, as many as DA3's
        chance = np.sin(pts[close] @ [12.9898, 78.233, 37.719]) * 43758.5453 % 1
        keep[close] = chance < 1 - near[close] * (1 - share)
    return cols, near, keep


def reached(blocks, tree, n):
    """True for each of n buildings DA3 has a point within REACH_M of
    (tree: a cKDTree of DA3's points, or None). Only these take the panos'
    colour -- it is there to meet DA3's; one it never reaches keeps its
    palette's."""
    out = np.zeros(n, bool)
    if tree is not None and len(blocks.pts):
        d = tree.query(blocks.pts, distance_upper_bound=REACH_M)[0]
        out[np.unique(blocks.which[d < REACH_M])] = True
    return out


def _turning(xy):
    """Twice the signed area of closed outline xy: over 0 counter-clockwise."""
    return float(np.sum(xy[:-1, 0] * xy[1:, 1] - xy[1:, 0] * xy[:-1, 1]))


def _wall_facade(form, h, u, v, glass_colour):
    """A wall's facade (n, 7): its points' metres along and up, its bay and
    floor, its windows' colour; NO_FACADE if it has no windows (facade_layout)."""
    layout = facade_layout(form, h)
    if layout is None:
        return np.full((len(u), 7), NO_FACADE)
    return np.c_[u, v, np.full(len(u), layout[0]), np.full(len(u), layout[1]), np.tile(glass_colour, (len(u), 1))]


def solid(outlines, ground, colour, spacing=None):
    """Every outline as triangles: (vertices (n, 3) world frame, colours
    (n, 3), as they are -- the viewer lights them --, faces (m, 3), facade
    (n, 7)): walls from its base (Form.base_m) to where the roof meets
    them; its roof (roofs.Roof.triangles); a flat underside if it starts
    in the air. facade: a wall vertex's metres along its wall and up from
    the ground, its bay/floor sizes and its windows' colour (glass);
    NO_FACADE on roofs and on walls with no windows. No details
    (balconies, trims, doors): drawn far off as paint (effects/blocks.js),
    its windows are paint too, never geometry. With spacing (one per
    outline), also each vertex's: one building's points all alike."""
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
        low = base - (0.0 if form.base_m else form.skirt_m)      # its walls' bottom
        roof = form.roof
        eaves = max(base, top - roof.height)
        ring = xy if _turning(xy) > 0 else xy[::-1]      # counter-clockwise: each wall's inside to its left
        for j, (a, c) in enumerate(zip(ring[:-1], ring[1:])):
            length = float(np.linalg.norm(c - a))
            if length < 1e-6:
                continue
            t = (c - a) / length
            u = length * roof.bends(a, c)
            en = a + t * u[:, None]
            up = eaves + roof.rise(en)
            if (up - base).max() < 1e-3:
                continue
            pts = np.concatenate([world(en, np.full(len(u), low)), world(en, up)])
            k = np.arange(len(u) - 1)
            n = len(u)
            faces = np.concatenate([np.c_[k, k + 1, n + k + 1], np.c_[k, n + k + 1, n + k]])
            add(pts, colour[i], faces, _wall_facade(form, h, np.r_[u, u], np.r_[np.full(n, low), up] - foot,
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
    # corners alike in place, colour, facade and spacing are one (a flat roof's, a wall's strip)
    rows = np.c_[np.concatenate(V), np.round(np.concatenate(C) * 255), np.concatenate(U),
                 np.concatenate(G)].astype(np.float32)
    rows, index = np.unique(rows, axis=0, return_inverse=True)
    out = rows[:, :3], rows[:, 3:6] / 255, index.ravel()[np.concatenate(F)], rows[:, 6:13]
    return out if spacing is None else (*out, rows[:, 13])


def seam(blocks, da3, da3_normals, da3_cols, roofs_near):
    """Leave to DA3 what it has of each building, and fade the rest in.

    Only walls DA3 has a copy of (fit_to_scene's planes) are touched, and
    only against DA3's points on that plane (on_plane), seen
    on the wall as (along, up): an OSM point goes where DA3 has a point
    within COVER of its own gap there -- however far in front or behind
    DA3's copy stands -- and stays, every one, where it has not: OSM fills
    exactly what DA3 lacks, its ragged edge included. (Thinning OSM near
    DA3 to interleave left a sparse strip: DA3 thins out at its edges too.)
    Over SEAM_FADE_M next to DA3 they are pulled onto its plane (by at
    most PULL_MAX_M) and towards its colour, so the two meet. A roof goes
    where DA3 has a point within COVER of its gap (roofs_near: their
    distance). Returns how many points DA3 already had."""
    from scipy.spatial import cKDTree
    from streetview_to_3d.postprocess.seams import ramp
    keep = np.ones(len(blocks.pts), bool)
    roof = blocks.edge == ROOF
    keep[roof] = roofs_near[roof] > COVER * blocks.gap[roof]
    order = np.argsort(blocks.edge, kind="stable")
    bounds = np.searchsorted(blocks.edge[order], np.arange(len(blocks.edges) + 1))
    for e, (a, t, out, length, plane) in enumerate(blocks.edges):
        mine = order[bounds[e]:bounds[e + 1]]
        if plane is None or not len(mine):
            continue
        n2, d2 = plane
        box = np.all(np.abs(da3[:, [0, 2]] - (a + t * length / 2)) < length / 2 + SEAM_FADE_M + 1, axis=1)
        idx = np.flatnonzero(box)
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
    """(n, 3) each building's colour as the panos see it, NaN where too
    little of it is seen.

    A wall point is seen by a camera within SEE_M when nothing is more than
    BEHIND_M (+ BEHIND of the distance) in front of it -- DA3's own copy of
    the wall stands a metre or two off OSM's outline and must not hide it,
    a tree or a nearer building further off does -- and its pixel is
    labelled building or wall (never sky over a building guessed too tall,
    nor a car). A building's colour is the median over every such pixel,
    SEEN_MIN of them at least: one flat colour, robust to a stray one."""
    from streetview_to_3d.fill.paint import NADIR_DEG, ZB_W, _at
    from streetview_to_3d.services.segment import LABEL_IDS
    ids = [LABEL_IDS["building"], LABEL_IDS["wall"]]
    everything = np.concatenate([occluders, pts])
    samples, owners = [], []
    h, w = ZB_W // 2, ZB_W
    for cam, ph in zip(cameras, photos):
        if ph is None:
            continue
        u, v, r, _ = cam.look(everything)
        near = np.full(h * w, np.inf)
        o = np.argsort(-r)
        iu, iv = (u[o] * w).astype(int), (v[o] * h).astype(int)
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                near[np.clip(iv + dv, 0, h - 1) * w + (iu + du) % w] = r[o]
        u, v, r, below = cam.look(pts)
        px = np.clip((v * h).astype(int), 0, h - 1) * w + (u * w).astype(int) % w
        seen = (r < SEE_M) & (below < NADIR_DEG) & (r <= near[px] + BEHIND_M + BEHIND * r)
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
        if b - a >= SEEN_MIN:
            out[owners[a]] = np.median(samples[a:b], 0)
    return out
