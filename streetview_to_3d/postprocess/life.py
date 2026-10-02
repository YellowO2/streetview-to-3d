"""What moves in the world around a scene, for the viewer to move
(effects/traffic.js, birds.js, boats.js): its cars' roads, its birds, its
boats' courses. A quiet place: a car on a street at a time, a few birds, a
boat or two.

Cars. The car roads (CARS) on the ground and over its bridges within
REACH_M of the cameras: further off, nobody watches a car. Through the
scene too, on its own road at its own ground's height (terrain's road_h
meets it there), in their lanes beside whatever DA3 parked at its kerb. There OSM's line
can lie metres to one side of the road DA3 has (at Lake Como, along its
pavement), so where the scene's own ground (seams.SceneGround) says what
is road, each point of the line is moved across onto the middle of that
road, its room the road's half-width (onto_road), easing back onto OSM's
line over EASE_M past it.
A car never drives into a building: OSM's widths are guesses, and its
buildings (as fitted onto DA3's walls) can stand nearer a road's line than
its lane. Each point keeps how much room it has to the nearest wall
(room: half the road's width at most); a car keeps to its lane only as far
as that lets it, and where a car's width does not fit (CLEAR_M) the road
is cut. The ground's roads are split where they cross, so a car can turn
there; a bridge joins only at its ends, as a road under it is no junction.
Their height is the road's own (terrain's road_h; a bridge's deck). Cars
keep to the side the country drives on (its panos' country code: LEFT).

Birds. The kinds that pass over the scene (its middle, its ground's height)
now and then, a small group at a time: gulls and swallows where there is
water within GULLS_M of the cameras, pigeons and swallows elsewhere.

Boats. On each water near the scene (within BOATS_M of its middle, at
least BOAT_WATER_M2 of it), a course round it SHORE_M off its shore: a
river's up one side and down the other, a lake's round it.

Ducks. A few on the water nearest the scene (within DUCKS_M of its middle):
a home DUCK_SHORE_M in from the shore where it comes nearest, clear of the
scene's own ground (DUCK_CLEAR_M: not under a bridge it stands on), and how
far they paddle round it (clear of both).

Cats. One or two on the scene's own ground -- a pavement, a verge: off the
car roads and what its panos call road, CAT_EDGE_M in from its edge,
CAT_NEAR_M of a camera -- each with
a few spots near together it sits at and strolls between, at the ground's
height there.

Written to life.json beside scene.json (its "life"), east/north metres:
    {"cars": {"side": "left" | "right",
              "colours": {"body": [[r, g, b], ...], "glass": .., "tyre": .., "head": .., "tail": ..},
              "roads": [{"width": m, "points": [[e, n, h], ...], "room": [m, ...]}, ...]},
     "birds": {"centre": [e, n, h],
               "flocks": [{"kind": "gull" | "pigeon" | "swallow",
                           "colours": {"body": [r, g, b], "wing": .., "tip": ..}}, ...]},
     "boats": {"colours": {"hull": .., "deck": .., "cabin": .., "glass": ..},
               "courses": [{"level": m, "points": [[e, n], ...]}, ...]},
     "ducks": {"colours": [{"body": .., "head": .., "tail": .., "bill": ..}, ...],
               "homes": [{"level": m, "at": [e, n], "reach": m}, ...]},
     "cats": {"colours": [{"body": .., "tail": ..}, ...],
              "cats": [{"spots": [[e, n, h], ...]}, ...]}}
each car road a stretch between junctions or ends, its points STEP_M apart;
two roads meet where an end of one is within JOIN_M of an end of the other
(traffic.js); a course closed, its last point its first's neighbour.
"""
import json
import os

import numpy as np

from .roads import KEPT, PATHS, _area, _on_ground, _width

FILENAME = "life.json"
CARS = tuple(k for k in KEPT if k not in PATHS)
REACH_M = 100.0
STEP_M = 4.0
MIN_M = 20.0              # a stretch shorter than this is left out
JOIN_M = 1.0
CLEAR_M = 1.1             # a road this near a wall is too narrow for a car (half its width and some)
ACROSS_M = 8.0            # the scene's road looked for this far either side of OSM's line
ROAD_MIN_M, EASE_M = 3.0, 20.0   # a road narrower than this is none; moved back onto OSM's over this
# the countries that drive on the left (ISO 3166-1 alpha-2)
LEFT = {"GB", "IE", "IM", "JE", "GG", "MT", "CY", "JP", "SG", "MY", "BN", "HK", "MO", "TH", "ID", "TL",
        "IN", "LK", "BD", "BT", "NP", "PK", "MV", "AU", "NZ", "PG", "FJ", "WS", "TO", "SB", "KI", "TV",
        "NR", "CK", "NU", "ZA", "NA", "BW", "ZW", "ZM", "MW", "MZ", "TZ", "KE", "UG", "LS", "SZ", "MU",
        "SC", "JM", "TT", "BB", "BS", "GY", "SR", "AG", "DM", "GD", "KN", "LC", "VC", "VG", "VI", "KY",
        "BM", "FK", "AI", "MS", "TC", "SH"}
GULLS_M = 200.0
DUCKS_M, DUCK_SHORE_M, DUCK_REACH_M, DUCK_CLEAR_M = 150.0, 10.0, 8.0, 5.0
CAT_NEAR_M = (2.0, 15.0)      # a cat's spots between these from the nearest camera
CAT_EDGE_M, CAT_ROAD_M = 1.0, 1.5   # ... this far in from the ground's edge, and past a car road's side
CAT_SPOT_M, CAT_SPOTS, CATS_APART_M = 6.0, 8, 10.0
BOATS_M, SHORE_M = 300.0, 12.0
BOAT_WATER_M2 = 3000.0
# the colours cars come in, each car its own while they last (traffic.js);
# their glass, tyres and lights
BODY = [(0.88, 0.88, 0.86), (0.08, 0.08, 0.09), (0.68, 0.69, 0.70), (0.32, 0.33, 0.35),
        (0.55, 0.09, 0.08), (0.12, 0.20, 0.40), (0.18, 0.30, 0.22), (0.72, 0.66, 0.55),
        (0.42, 0.58, 0.72)]
GLASS = (0.14, 0.17, 0.21)
TYRE = (0.06, 0.06, 0.06)
HEAD = (0.95, 0.93, 0.82)
TAIL = (0.62, 0.08, 0.06)
# a gull's: white, grey wings, black tips; a pigeon's: blue-grey, darker tips
GULL = {"body": (0.93, 0.93, 0.92), "wing": (0.66, 0.68, 0.71), "tip": (0.1, 0.1, 0.11)}
PIGEON = {"body": (0.52, 0.53, 0.57), "wing": (0.6, 0.61, 0.65), "tip": (0.2, 0.2, 0.23)}
# a swallow's: dark steel-blue all over, its tips darker
SWALLOW = {"body": (0.12, 0.14, 0.24), "wing": (0.1, 0.11, 0.19), "tip": (0.06, 0.06, 0.1)}
# a mallard's: a drake's grey body and green head, a hen's brown all over
DUCKS = [{"body": (0.62, 0.6, 0.56), "head": (0.1, 0.3, 0.18), "tail": (0.12, 0.12, 0.13), "bill": (0.85, 0.75, 0.25)},
         {"body": (0.5, 0.38, 0.26), "head": (0.45, 0.35, 0.25), "tail": (0.35, 0.27, 0.2), "bill": (0.75, 0.5, 0.2)}]
# cats: a ginger, a black, a grey tabby, a white one
CATS = [{"body": (0.8, 0.48, 0.2), "tail": (0.7, 0.4, 0.16)}, {"body": (0.08, 0.08, 0.09), "tail": (0.08, 0.08, 0.09)},
        {"body": (0.5, 0.48, 0.45), "tail": (0.35, 0.33, 0.31)}, {"body": (0.9, 0.89, 0.86), "tail": (0.85, 0.84, 0.8)}]
# a small motorboat's
BOAT = {"hull": (0.92, 0.92, 0.9), "deck": (0.56, 0.42, 0.28), "cabin": (0.88, 0.88, 0.86),
        "glass": (0.14, 0.17, 0.21)}


def side(pano_id):
    """"left" or "right": the side cars keep to where pano_id was taken."""
    import aiohttp
    from streetlevel import streetview
    from streetview_to_3d.services.http_headers import BROWSER_HEADERS
    from streetview_to_3d.services.streetview_fetch import run_async

    async def country():
        async with aiohttp.ClientSession(headers=BROWSER_HEADERS) as session:
            pano = await streetview.find_panorama_by_id_async(pano_id, session=session)
            return pano.country_code if pano else None
    return "left" if run_async(country()) in LEFT else "right"


def roads(elements, to_xy, height, decks, keep, walls=(), ground=None):
    """[(width m, (n, 3) east/north/height every STEP_M, room (n,) m)]: the
    car roads (elements: osm.fetch's; to_xy their geometry's), on the ground
    at height(xy) -- onto the scene's own road where its ground (a
    seams.SceneGround) has one -- and over decks (roads.Deck, those of car
    roads), kept only where keep(xy) (n, 2) is true and CLEAR_M clear of
    walls (buildings' outlines, (m, 2) each); room: how far the nearest wall
    is, half the width at most (the scene's road's, on it)."""
    import shapely
    wall = shapely.union_all([shapely.make_valid(shapely.Polygon(w)) for w in walls if len(w) >= 3])
    shapely.prepare(wall)

    def clear(xy):
        return shapely.distance(wall, shapely.points(xy)) if not wall.is_empty else np.full(len(xy), np.inf)

    def runs(width, xy, h=None):
        c = clear(xy)
        if h is None and ground is not None:
            xy, half = onto_road(xy, ground)
            c = np.fmin(clear(xy), half)
        return [(width, np.c_[xy[i], height(xy[i]) if h is None else h[i]], np.minimum(c[i], width / 2))
                for i in _runs(xy, keep(xy) & (c >= CLEAR_M))]
    ways = [(shapely.LineString(to_xy(e["geometry"])), _width(e["tags"])) for e in elements
            if e.get("type") == "way" and e.get("tags", {}).get("highway") in CARS
            and _on_ground(e["tags"]) and not _area(e) and len(e.get("geometry") or []) >= 2]
    out = []
    if ways:
        tree = shapely.STRtree([w for w, _ in ways])
        for piece in shapely.get_parts(shapely.line_merge(shapely.union_all([w for w, _ in ways]))):
            xy = _every(shapely.get_coordinates(piece))
            width = ways[tree.nearest(shapely.Point(xy[len(xy) // 2]))][1]
            out += runs(width, xy)
    for deck in decks:
        xy = _every(deck.xy)
        out += runs(deck.width, xy, np.interp(_along(xy), deck.at, deck.h))
    return out


def onto_road(xy, ground):
    """(xy moved, half-width (n,)): a line's points (n, 2), STEP_M apart,
    each moved across onto the middle of the road the scene's ground
    (seams.SceneGround) has there -- the run of it its panos call road
    nearest the line, ROAD_MIN_M wide at least, within ACROSS_M -- easing
    back onto the line over EASE_M where it has none; the road's half-width
    where it has one (NaN elsewhere)."""
    from .seams import CELL_M
    n = len(xy)
    ahead = np.gradient(xy, axis=0) if n > 1 else np.array([[1.0, 0.0]])
    side = np.c_[-ahead[:, 1], ahead[:, 0]] / np.maximum(np.linalg.norm(ahead, axis=1), 1e-9)[:, None]
    off = np.arange(-ACROSS_M, ACROSS_M + 1e-9, CELL_M)
    share = ground.road_at((xy[:, None, :] + off[None, :, None] * side[:, None, :]).reshape(-1, 2))
    on = share.reshape(n, len(off)) >= 0.5
    move, half = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        edges = np.flatnonzero(np.diff(np.r_[0, on[i].astype(int), 0]))
        spans = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if (b - a) * CELL_M >= ROAD_MIN_M]
        if spans:
            a, b = min(spans, key=lambda s: 0 if off[s[0]] <= 0 <= off[s[1] - 1] else
                       min(abs(off[s[0]]), abs(off[s[1] - 1])))
            move[i], half[i] = (off[a] + off[b - 1]) / 2, (b - a) * CELL_M / 2
    have = np.flatnonzero(np.isfinite(move))
    if not len(have):
        return xy, half
    # where it has no road, back onto the line over EASE_M from the nearest that has
    at = _along(xy)
    nearest = have[np.abs(at[:, None] - at[have][None, :]).argmin(1)]
    eased = move[nearest] * np.clip(1 - np.abs(at - at[nearest]) / EASE_M, 0, 1)
    moved = np.where(np.isfinite(move), move, eased)
    moved = np.convolve(np.r_[moved[0], moved, moved[-1]], np.ones(3) / 3, "valid")   # no jolt point to point
    return xy + side * moved[:, None], half


def _along(xy):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]


def _every(xy):
    """The line xy (n, 2) as points STEP_M apart, its ends kept."""
    at = _along(xy)
    s = np.linspace(0, at[-1], max(2, int(np.ceil(at[-1] / STEP_M)) + 1))
    return np.c_[np.interp(s, at, xy[:, 0]), np.interp(s, at, xy[:, 1])]


def _runs(xy, ok):
    """The indices of each run of xy (n, 2) where ok (n,), each at least MIN_M long."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(ok, int), 0]))
    return [np.arange(a, b) for a, b in zip(edges[::2], edges[1::2])
            if b - a >= 2 and _along(xy[a:b])[-1] >= MIN_M]


def _water(surfaces):
    """water.Water's surfaces as [(level, polygon east/north)]."""
    import shapely
    return [(s["level"], shapely.make_valid(shapely.Polygon(s["outer"], s["holes"]))) for s in surfaces]


def birds(surfaces, cams, ground_h):
    """life.json's birds: gulls and swallows if any of surfaces
    (water.Water's) lies within GULLS_M of cams (east/north (n, 2)), else
    pigeons and swallows, over the cameras' middle at ground_h."""
    import shapely
    near = shapely.MultiPoint(cams).buffer(GULLS_M)
    gulls = any(w.intersects(near) for _, w in _water(surfaces))
    kinds = {"gull": GULL, "swallow": SWALLOW} if gulls else {"pigeon": PIGEON, "swallow": SWALLOW}
    return {"centre": [round(float(v), 2) for v in (*np.mean(cams, axis=0), ground_h)],
            "flocks": [{"kind": k, "colours": _rgb(colours)} for k, colours in kinds.items()]}


def boats(surfaces, cams):
    """life.json's boats' courses: round each water within BOATS_M of the
    cams' (east/north (n, 2)) middle, SHORE_M off its shore, at its level."""
    import shapely
    reach = shapely.Point(np.mean(cams, axis=0)).buffer(BOATS_M)
    courses = []
    for level, w in _water(surfaces):
        for part in shapely.get_parts(w.intersection(reach).buffer(-SHORE_M)):
            if part.geom_type == "Polygon" and part.area >= BOAT_WATER_M2:
                ring = shapely.get_coordinates(part.exterior.simplify(1.0))[:-1]
                courses.append({"level": round(float(level), 2), "points": np.round(ring, 2).tolist()})
    return {"colours": _rgb(BOAT), "courses": courses}


def ducks(surfaces, cams, ground=None):
    """life.json's ducks: a home on the water nearest the cams' (east/north
    (n, 2)) middle, DUCK_SHORE_M in from its shore and DUCK_CLEAR_M off the
    scene's ground (a seams.SceneGround), if within DUCKS_M."""
    import shapely
    middle = np.mean(cams, axis=0)
    homes = []
    for level, w in _water(surfaces):
        inner = w.buffer(-DUCK_SHORE_M)
        if inner.is_empty:
            continue
        # its edge, every 2 m: DUCK_SHORE_M from the shore
        at = shapely.get_coordinates(shapely.segmentize(inner.boundary, 2.0))
        off = ground.at(at)[0] if ground is not None else np.full(len(at), np.inf)
        at, off = at[off >= DUCK_CLEAR_M], off[off >= DUCK_CLEAR_M]
        if not len(at):
            continue
        k = np.argmin(np.hypot(*(at - middle).T))
        if np.hypot(*(at[k] - middle)) > DUCKS_M:
            continue
        reach = min(DUCK_REACH_M, w.boundary.distance(shapely.Point(at[k])) - 2, off[k] - 2)
        homes.append((np.hypot(*(at[k] - middle)), {"level": round(float(level), 2),
                                                     "at": np.round(at[k], 2).tolist(), "reach": round(float(reach), 2)}))
    return {"colours": [_rgb(c) for c in DUCKS], "homes": [h for _, h in sorted(homes, key=lambda x: x[0])[:1]]}


def cats(ground, stretches, cams, seed=0):
    """life.json's cats: one or two, each spots on the scene's own ground
    (seams.SceneGround) off the car roads (roads()'s stretches) near the
    cams (east/north (n, 2)), CATS_APART_M apart."""
    import shapely
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import cKDTree
    from .seams import CELL_M
    have = np.isfinite(ground.height)
    if not have.any():
        return {"colours": [_rgb(c) for c in CATS], "cats": []}
    deep = distance_transform_edt(have) * CELL_M                 # how far in from the ground's edge
    i, j = np.nonzero(deep >= CAT_EDGE_M)
    xy = (np.c_[i, j] + ground.lo + 0.5) * CELL_M
    d = cKDTree(cams).query(xy)[0]
    ok = (d >= CAT_NEAR_M[0]) & (d <= CAT_NEAR_M[1]) & ~(ground.road_at(xy) >= 0.5)
    if stretches:
        roads_ = shapely.union_all([shapely.LineString(p[:, :2]).buffer(w / 2 + CAT_ROAD_M) for w, p, _ in stretches])
        ok &= ~shapely.contains_xy(roads_, xy[:, 0], xy[:, 1])
    xy, h = xy[ok], ground.height[i[ok], j[ok]]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(2):
        far = np.ones(len(xy), bool)
        for c in out:
            far &= np.hypot(*(xy - c["spots"][0][:2]).T) >= CATS_APART_M
        if not far.any():
            break
        home = xy[rng.choice(np.flatnonzero(far))]
        near = np.flatnonzero(np.hypot(*(xy - home).T) <= CAT_SPOT_M)
        pick = rng.choice(near, min(CAT_SPOTS, len(near)), replace=False)
        pick = pick[np.argsort(np.hypot(*(xy[pick] - home).T))]          # its home first
        out.append({"spots": np.round(np.c_[xy[pick], h[pick]], 2).tolist()})
    return {"colours": [_rgb(c) for c in CATS], "cats": out}


def cars(stretches, side_):
    """life.json's cars: stretches (roads()), side_ (side())."""
    return {"side": side_, "colours": {"body": [list(c) for c in BODY], "glass": list(GLASS), "tyre": list(TYRE),
                                       "head": list(HEAD), "tail": list(TAIL)},
            "roads": [{"width": round(float(w), 2), "points": np.round(p, 2).tolist(),
                       "room": np.round(r, 2).tolist()} for w, p, r in stretches]}


def _rgb(colours):
    return {k: list(c) for k, c in colours.items()}


def save(scene_dir, **parts):
    """Write FILENAME (see the module) of its parts -- cars (cars()), birds,
    boats, ducks, cats; its name."""
    with open(os.path.join(scene_dir, FILENAME), "w") as f:
        json.dump(parts, f, separators=(",", ":"))
    return FILENAME
