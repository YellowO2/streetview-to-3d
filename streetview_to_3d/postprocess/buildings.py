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
    starts (height less roof:height); flat if it has none
  - one DA3 built a wall of is slid onto it, whatever still stands in
    front of that wall cut away as from a solid block, a guessed height
    made DA3's (fit_to_scene); what DA3 already has of it is left to DA3 and the rest
    faded in next to it, judged on the wall itself (seam)
  - standing on the lowest ground under its outline (terrain.py's map)
  - walls and a flat roof, spaced by how near the scene's cameras they
    are, as the land is (terrain.gap_at), and the walls again further in,
    sparser (INNER_M), so it is not seen through
  - one colour per building: what the scene's panos see of it where they
    see enough (pano_colours), else its own "building:colour" tag (or
    building:material's, MATERIAL), else one
    of the place's own building colours, softened (palette) -- the
    satellite's, from 10 m up, came out grey-brown; walls shaded by which
    way they face, the roof a little lighter and shaded the same way,
    "roof:colour" (or roof:material's) if it has one (never a pano's own
    pixel: from the street they see its edge against the sky)

Called by terrain.build, which writes them to buildings.ply.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from streetview_to_3d.postprocess.roofs import Roof

LEVEL_M = 3.2
DEFAULT_M = {"house": 7, "detached": 7, "semidetached_house": 7, "terrace": 7, "bungalow": 5,
             "roof": 4, "shed": 3, "garage": 3, "garages": 3, "carport": 3, "hut": 3, "kiosk": 3}
DEFAULT_OTHER_M = 12.0
SEE_M = 150.0                 # panos colour the buildings this close to them
BEHIND_M, BEHIND = 3.0, 0.05  # how far behind what is in front a wall may stand and still be seen
SEEN_MIN = 20                 # pixels of a building the panos must see to colour it
FIT_M = 5.0                   # buildings this far past the scene's points are not looked at
WALL_SAMPLE_M = 1.0
SNAP_MAX_M = 2.0              # OSM's real offsets were 0.2-1.3 m (Stockholm, NTU)
ROAD_SLACK = 0.02             # a slide may put this much more of an outline onto roads
PULL_MAX_M = 1.0              # a wall's seam pulled onto DA3's plane by at most this
TRIM_TOL_M, TRIM_FRONT_M = 0.3, 30.0   # cut from this far in front of a DA3 wall, out to this
TRIM_MIN_M2, TRIM_MAX = 1.0, 0.2
CHEER_SAT, CHEER_LIFT = 1.3, 1.15
ROOF_LIFT = 1.08              # a roof faces the sky: a little lighter than its walls
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
WALL_SHADE = 0.85             # a wall facing away from the sun, of one facing it (0.6 looked dull)
ROOF, INNER = -1, -2          # Blocks.edge for a roof's points and an inner wall's
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
MIN_HEIGHT_M = 2.5
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
                colour("roof:colour", "roof:material"))


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
    out = []
    for e, ring in _rings(elements):
        xy = to_xy(ring)
        if (xy[:-1, 0] * xy[1:, 1] - xy[1:, 0] * xy[:-1, 1]).sum() < 0:   # counter-clockwise: normals face out
            xy = xy[::-1]
        h, guessed = _height(e["tags"])
        form = _form(e["tags"], xy, h)
        out.append((xy, max(h, form.base_m + MIN_HEIGHT_M), guessed, form))
    parts = [Polygon(o[0]).buffer(0).representative_point() for o in out if o[3].part]
    if not parts:
        return out
    tree = shapely.STRtree(parts)
    return [o for o in out if o[3].part or not len(tree.query(Polygon(o[0]).buffer(0), "contains"))]


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
        base = ground(xy).min()
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
    which building, how lit (NaN on a roof), which wall edge (ROOF on a
    roof, INNER on an inner wall) and its place on that wall, u metres along and v metres up
    (height above sea level), and how far it is from its neighbours (gap); on a roof, what its
    building's colour is multiplied by there (shade: lighter, lit by which way it faces) and
    whether it keeps its own (own: roof:colour). edges[e]: (start, along, outward, length,
    DA3's plane for it (n2, d2) or None)."""

    def __init__(self, pts, cols, which, light, edge, u, v, gap, edges, shade, own):
        self.pts, self.cols, self.which, self.light = pts, cols, which, light
        self.edge, self.u, self.v, self.gap, self.edges = edge, u, v, gap, edges
        self.shade, self.own = shade, own

    def take(self, keep):
        for k in ("pts", "cols", "which", "light", "edge", "u", "v", "gap", "shade", "own"):
            setattr(self, k, getattr(self, k)[keep])


def cheer(rgb):
    """A map's colour (satellite, a pano's median) livelier: from above and
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


def _lit(normals, sun):
    """How a roof facing normals (m, 3: east, north, up) is shaded, of a
    flat one: shaded as the walls are, and a little lighter (ROOF_LIFT)."""
    up = lambda n: WALL_SHADE + (1 - WALL_SHADE) * np.clip(n @ sun, 0, 1)
    world = normals[:, [0, 2, 1]] * [1, -1, 1]                  # -> x east, y down, z north
    return ROOF_LIFT * up(world) / up(np.array([0.0, -1.0, 0.0]))


def points(outlines, spacing, ground, colour, sun):
    """Blocks for every outline.

    spacing(xy): point spacing at east/north points -- a wall is laid in
    pieces up to CHUNK_M long, each spaced as at its middle, so only what
    is near gets dense; ground(xy): the ground's height there; colour:
    (n, 3) each building's (colours) -- a roof a little lighter
    (ROOF_LIFT), roofs and walls shaded by which way they face. Each
    stands from its Form's base_m over its ground up to the roof, the roof
    (roofs.py) up to its height, the walls reaching up to meet it (a
    gable's end); a part that starts in the air has a flat underside."""
    roofs, roof_shades, walls, lights, edge_of, us, vs, gaps, roof_gaps, edges = [], [], [], [], [], [], [], [], [], []
    for xy, h, _, form, planes in outlines:
        foot = ground(xy).min()
        base, top = foot + form.base_m, foot + h
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
        roof_shades.append(_lit(rn, sun))
        roof_gaps.append(np.full(len(rp), s))
        # walls: along each edge and up it, to where the roof meets it; lit by how squarely it faces the sun
        w, light, eo, uu, vv, gg = [], [], [], [], [], []
        for j, (a, c) in enumerate(zip(xy[:-1], xy[1:])):
            length = float(np.linalg.norm(c - a))
            if length < 1e-6:
                continue
            t = (c - a) / length
            out = np.array([t[1], -t[0]])                    # outward for a counter-clockwise ring
            e = len(edges)
            edges.append((a, t, out, length, planes.get(j)))
            lit = abs(np.array([out[0], 0.0, out[1]]) @ sun)
            for c0 in np.arange(0, length, CHUNK_M):
                c1 = min(length, c0 + CHUNK_M)
                s = float(spacing((a + t * (c0 + c1) / 2)[None])[0])
                along = np.arange(c0, c1, s)
                rise = roof.rise(a + t * along[:, None])
                levels = np.arange(base, eaves + (rise.max() if len(rise) else 0), s)
                U, V = np.repeat(along, len(levels)), np.tile(levels, len(along))
                ok = V <= eaves + np.repeat(rise, len(levels)) + 1e-6
                U, V = U[ok], V[ok]
                w.append(np.column_stack([a[0] + t[0] * U, -V, a[1] + t[1] * U]))
                light.append(np.full(len(U), lit))
                eo.append(np.full(len(U), e))
                uu.append(U)
                vv.append(V)
                gg.append(np.full(len(U), s))
        # inner walls: the same, further in, sparser, unlit (so darker); not the building's own edges
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
                    along, levels = np.arange(c0, c1, s), np.arange(base, eaves, s)
                    U, V = np.repeat(along, len(levels)), np.tile(levels, len(along))
                    w.append(np.column_stack([a[0] + t[0] * U, -V, a[1] + t[1] * U]))
                    light.append(np.zeros(len(U)))
                    eo.append(np.full(len(U), INNER))
                    uu.append(np.zeros(len(U)))
                    vv.append(V)
                    gg.append(np.full(len(U), s))
        walls.append(np.concatenate(w) if w else np.zeros((0, 3)))
        lights.append(np.concatenate(light) if light else np.zeros(0))
        edge_of.append(np.concatenate(eo) if eo else np.zeros(0, int))
        us.append(np.concatenate(uu) if uu else np.zeros(0))
        vs.append(np.concatenate(vv) if vv else np.zeros(0))
        gaps.append(np.concatenate(gg) if gg else np.zeros(0))
    if not roofs:
        z = np.zeros((0, 3))
        return Blocks(z, z, np.zeros(0, int), np.zeros(0), np.zeros(0, int), np.zeros(0), np.zeros(0),
                      np.zeros(0), [], np.zeros(0), np.zeros(0, bool))
    pts, cols, which, lit, edge, u, v, gap, shade, own = [], [], [], [], [], [], [], [], [], []
    for i, (roof, rs, wall, light, eo, uu, vv, rg, gg) in enumerate(zip(roofs, roof_shades, walls, lights, edge_of,
                                                                         us, vs, roof_gaps, gaps)):
        mine = outlines[i][3].roof_colour
        pts += [roof, wall]
        cols += [np.clip((colour[i] if mine is None else mine) * rs[:, None], 0, 1),
                 colour[i] * (WALL_SHADE + (1 - WALL_SHADE) * light[:, None])]
        which.append(np.full(len(roof) + len(wall), i))
        lit += [np.full(len(roof), np.nan), light]
        edge += [np.full(len(roof), ROOF), eo.astype(int)]
        u += [np.zeros(len(roof)), uu]
        v += [-roof[:, 1], vv]
        gap += [rg, gg]
        shade += [rs, np.ones(len(wall))]
        own += [np.full(len(roof), mine is not None), np.zeros(len(wall), bool)]
    return Blocks(np.concatenate(pts), np.concatenate(cols), np.concatenate(which), np.concatenate(lit),
                  np.concatenate(edge), np.concatenate(u), np.concatenate(v), np.concatenate(gap), edges,
                  np.concatenate(shade), np.concatenate(own))


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
