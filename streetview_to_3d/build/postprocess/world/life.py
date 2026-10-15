"""What moves round the scene, for the viewer to animate.

Cars on OSM's car roads (car_roads.py: moved onto the scene's own road, clear of walls, on the
country's side), birds, boats along nearby shores, ducks and cats.

Saved as life.json beside scene.json (east/north metres):
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
Car road points are car_roads.STEP_M apart; room is the clearance to the nearest wall.
"""

import numpy as np

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


def life(site, stretches, water, elevation, side):
    """life.json's contents round site (Site): cars on stretches (car_roads) keeping to side
    (driving_side), birds over the cameras at the ground's mean height (Elevation), boats and
    ducks on the water (Water), cats on the scene's ground."""
    cams, surfaces = site.cameras, water.surfaces
    return dict(cars=_cars(stretches, side),
                birds=_birds(surfaces, cams, float(np.mean(elevation(cams)))),
                boats=_boats(surfaces, cams), ducks=_ducks(surfaces, cams, site.ground),
                cats=_cats(site.ground, stretches, cams))


def _water(surfaces):
    """water.Water's surfaces as [(level, polygon east/north)]."""
    import shapely
    return [(s["level"], shapely.make_valid(shapely.Polygon(s["outer"], s["holes"]))) for s in surfaces]


def _birds(surfaces, cams, ground_h):
    """life.json's birds over the cameras' middle: gulls if water is within GULLS_M, else pigeons; and swallows."""
    import shapely
    near = shapely.MultiPoint(cams).buffer(GULLS_M)
    gulls = any(w.intersects(near) for _, w in _water(surfaces))
    kinds = {"gull": GULL, "swallow": SWALLOW} if gulls else {"pigeon": PIGEON, "swallow": SWALLOW}
    return {"centre": [round(float(v), 2) for v in (*np.mean(cams, axis=0), ground_h)],
            "flocks": [{"kind": k, "colours": _rgb(colours)} for k, colours in kinds.items()]}


def _boats(surfaces, cams):
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


def _ducks(surfaces, cams, ground=None):
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


def _cats(ground, stretches, cams, seed=0):
    """life.json's cats: up to two, with spots on the scene's non-road ground near the cameras."""
    import shapely
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import cKDTree
    from streetview_to_3d.build.postprocess.SceneGround import CELL_M
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


def _cars(stretches, side_):
    """life.json's cars: stretches (car_roads), side_ (driving_side)."""
    return {"side": side_, "colours": {"body": [list(c) for c in BODY], "glass": list(GLASS), "tyre": list(TYRE),
                                       "head": list(HEAD), "tail": list(TAIL)},
            "roads": [{"width": round(float(w), 2), "points": np.round(p, 2).tolist(),
                       "room": np.round(r, 2).tolist()} for w, p, r in stretches]}


def _rgb(colours):
    return {k: list(c) for k, c in colours.items()}
