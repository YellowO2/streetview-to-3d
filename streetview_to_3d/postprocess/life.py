"""What moves around a scene, for the viewer to animate: cars on OSM car roads (moved onto the
scene's own road, clear of walls, on the country's side), birds, boats along nearby shores,
ducks and cats.

Written to life.json beside scene.json (east/north metres):
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
Car road points are STEP_M apart; room is the clearance to the nearest wall.
"""
import json
import os

import numpy as np

from .roads import KEPT, PATHS, _area, _on_ground, _width

FILENAME = "life.json"
CARS = tuple(k for k in KEPT if k not in PATHS)
REACH_M = 100.0           # car roads this near a camera
STEP_M = 4.0
MIN_M = 20.0              # shorter stretches are dropped
CLEAR_M = 1.1             # min clearance to a wall for a car
ACROSS_M = 8.0            # the scene's road is searched this far either side of OSM's line
ROAD_MIN_M, EASE_M = 3.0, 20.0   # min scene road width; ease back onto OSM's line over this
# countries driving on the left (ISO 3166-1 alpha-2)
LEFT = {"GB", "IE", "IM", "JE", "GG", "MT", "CY", "JP", "SG", "MY", "BN", "HK", "MO", "TH", "ID", "TL",
        "IN", "LK", "BD", "BT", "NP", "PK", "MV", "AU", "NZ", "PG", "FJ", "WS", "TO", "SB", "KI", "TV",
        "NR", "CK", "NU", "ZA", "NA", "BW", "ZW", "ZM", "MW", "MZ", "TZ", "KE", "UG", "LS", "SZ", "MU",
        "SC", "JM", "TT", "BB", "BS", "GY", "SR", "AG", "DM", "GD", "KN", "LC", "VC", "VG", "VI", "KY",
        "BM", "FK", "AI", "MS", "TC", "SH"}
GULLS_M = 200.0
DUCKS_M, DUCK_SHORE_M, DUCK_REACH_M, DUCK_CLEAR_M = 150.0, 10.0, 8.0, 5.0
CAT_NEAR_M = (2.0, 15.0)      # cat spots between these distances from the nearest camera
CAT_EDGE_M, CAT_ROAD_M = 1.0, 1.5   # ... this far in from the ground's edge and past a car road's side
CAT_SPOT_M, CAT_SPOTS, CATS_APART_M = 6.0, 8, 10.0
BOATS_M, SHORE_M = 300.0, 12.0
BOAT_WATER_M2 = 3000.0
# car body colours; glass, tyres and lights
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
    """[(width m, (n, 3) east/north/height every STEP_M, room (n,) m)] of the car roads and decks.

    At height(xy), moved onto the scene's road where ground (seams.SceneGround)
    has one, kept where keep(xy) and CLEAR_M from walls; room: clearance to
    the nearest wall, at most half the width."""
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
    """(xy moved, half-width (n,), NaN off road): each point moved sideways onto the middle of the
    scene ground's nearest road run (ROAD_MIN_M wide, within ACROSS_M), easing back over EASE_M."""
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
    # without road, ease back onto the line from the nearest point that has one
    at = _along(xy)
    nearest = have[np.abs(at[:, None] - at[have][None, :]).argmin(1)]
    eased = move[nearest] * np.clip(1 - np.abs(at - at[nearest]) / EASE_M, 0, 1)
    moved = np.where(np.isfinite(move), move, eased)
    moved = np.convolve(np.r_[moved[0], moved, moved[-1]], np.ones(3) / 3, "valid")   # smooth jolts
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
    """life.json's birds over the cameras' middle: gulls if water is within GULLS_M, else pigeons; and swallows."""
    import shapely
    near = shapely.MultiPoint(cams).buffer(GULLS_M)
    gulls = any(w.intersects(near) for _, w in _water(surfaces))
    kinds = {"gull": GULL, "swallow": SWALLOW} if gulls else {"pigeon": PIGEON, "swallow": SWALLOW}
    return {"centre": [round(float(v), 2) for v in (*np.mean(cams, axis=0), ground_h)],
            "flocks": [{"kind": k, "colours": _rgb(colours)} for k, colours in kinds.items()]}


def boats(surfaces, cams):
    """life.json's boat courses SHORE_M off the shore of each water within BOATS_M of the cameras' middle."""
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
    """life.json's ducks: one home DUCK_SHORE_M in from the nearest shore within DUCKS_M, clear of the scene's ground."""
    import shapely
    middle = np.mean(cams, axis=0)
    homes = []
    for level, w in _water(surfaces):
        inner = w.buffer(-DUCK_SHORE_M)
        if inner.is_empty:
            continue
        # candidate spots every 2 m along the inset shore
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
    """life.json's cats: up to two, with spots on the scene's non-road ground near the cameras."""
    import shapely
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import cKDTree
    from .seams import CELL_M
    have = np.isfinite(ground.height)
    if not have.any():
        return {"colours": [_rgb(c) for c in CATS], "cats": []}
    deep = distance_transform_edt(have) * CELL_M                 # distance in from the ground's edge
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
        pick = pick[np.argsort(np.hypot(*(xy[pick] - home).T))]          # home first
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
    """Write life.json from its parts; returns its name."""
    with open(os.path.join(scene_dir, FILENAME), "w") as f:
        json.dump(parts, f, separators=(",", ":"))
    return FILENAME
