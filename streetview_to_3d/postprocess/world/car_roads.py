"""The stretches of road near the scene that cars drive along."""
import numpy as np

from streetview_to_3d.postprocess.world.tools.osm_roads import KEPT, PATHS, is_area, on_ground, width as _width

CARS = tuple(k for k in KEPT if k not in PATHS)
REACH_M = 100.0           # car roads this near a camera
STEP_M = 4.0
MIN_M = 20.0              # shorter stretches are dropped
CLEAR_M = 1.1             # min clearance to a wall for a car
ACROSS_M = 8.0            # the scene's road is searched this far either side of OSM's line
ROAD_MIN_M, EASE_M = 3.0, 20.0   # min scene road width; ease back onto OSM's line over this


def car_roads(elements, roads, outlines, site):
    """[(width m, (n, 3) east/north/height every STEP_M, room (n,) m)] of OSM's car roads and
    their bridges (_roads) within REACH_M of site's cameras (Site), at the roads' height (Roads),
    clear of the buildings' walls (Outlines), on the scene's own road where it has one."""
    cars = {e["id"] for e in elements if e.get("tags", {}).get("highway") in CARS}
    return _roads(elements, site.to_xy, roads.height, [d for d in roads.height.decks if d.ids & cars],
                  lambda xy: site.from_cameras(xy) < REACH_M, [o[0] for o in outlines.items], site.ground)


def _roads(elements, to_xy, height, decks, keep, walls=(), ground=None):
    """[(width m, (n, 3) east/north/height every STEP_M, room (n,) m)] of the car roads and decks.

    At height(xy), moved onto the scene's road where ground (postprocess/SceneGround.py)
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
            xy, half = _onto_road(xy, ground)
            c = np.fmin(clear(xy), half)
        return [(width, np.c_[xy[i], height(xy[i]) if h is None else h[i]], np.minimum(c[i], width / 2))
                for i in _runs(xy, keep(xy) & (c >= CLEAR_M))]
    ways = [(shapely.LineString(to_xy(e["geometry"])), _width(e["tags"])) for e in elements
            if e.get("type") == "way" and e.get("tags", {}).get("highway") in CARS
            and on_ground(e["tags"]) and not is_area(e) and len(e.get("geometry") or []) >= 2]
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


def _onto_road(xy, ground):
    """(xy moved, half-width (n,), NaN off road): each point moved sideways onto the middle of the
    scene ground's nearest road run (ROAD_MIN_M wide, within ACROSS_M), easing back over EASE_M."""
    from streetview_to_3d.postprocess.SceneGround import CELL_M
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
