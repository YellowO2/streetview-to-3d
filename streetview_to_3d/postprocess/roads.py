"""The roads around the scene, from OpenStreetMap, laid on the terrain.

OSM draws a road as its centreline, so each is given a width -- its own
"width" tag, else "lanes" x LANE_M, else a guess by kind (WIDTH_M) -- and
laid as a strip of points LIFT_M above the ground under it, spaced as the
terrain is at that distance, and only where that spacing leaves it a point
wide at least: further out it would be a blob, and the satellite shows it.
Paths, steps and the like are narrow ones. Coloured by its "surface" tag
(SURFACE), else asphalt for a road and paving for a path.

Bridges (bridges, bridge_points) are laid apart: not on the ground but
from the ground at one end to the ground at the other, arched a little
(ARCH_PER of their length, ARCH_MAX at most), with a DECK_M edge down
each side, so seen from low they are a deck, not a sheet. Tunnels are not
drawn.

Called by terrain.build, whose points they join in terrain.ply.
"""
import numpy as np

LANE_M = 3.2
WIDTH_M = {"motorway": 14, "trunk": 12, "primary": 10, "secondary": 9, "tertiary": 8,
           "motorway_link": 6, "trunk_link": 6, "primary_link": 6, "secondary_link": 6,
           "tertiary_link": 6, "residential": 6, "unclassified": 6, "living_street": 5,
           "service": 4, "pedestrian": 4, "track": 3}
NARROW_M = 2.0            # footways, paths, cycleways, steps, anything else
SKIP = ("proposed", "construction", "raceway", "bus_stop", "platform", "elevator", "corridor")
LIFT_M = 0.15
ARCH_PER, ARCH_MAX, DECK_M = 0.03, 2.0, 1.0
SIDE = np.array([0.60, 0.60, 0.58]) * 0.85      # a bridge's edge: concrete, in shade
SURFACE = {"asphalt": (0.33, 0.33, 0.35), "concrete": (0.60, 0.60, 0.58), "paving_stones": (0.58, 0.51, 0.45),
           "sett": (0.50, 0.47, 0.44), "tiles": (0.63, 0.54, 0.47), "wood": (0.50, 0.38, 0.26),
           "gravel": (0.58, 0.55, 0.49), "fine_gravel": (0.62, 0.58, 0.50), "compacted": (0.58, 0.54, 0.46),
           "ground": (0.48, 0.41, 0.31), "dirt": (0.48, 0.41, 0.31), "earth": (0.48, 0.41, 0.31),
           "sand": (0.72, 0.66, 0.52), "grass": (0.36, 0.50, 0.30)}
PATHS = ("footway", "path", "pedestrian", "steps", "cycleway", "bridleway")


def _width(tags):
    for key, per in (("width", 1.0), ("lanes", LANE_M)):
        try:
            v = float(str(tags[key]).split()[0].replace(",", "."))
            if v > 0:
                return v * per
        except (KeyError, ValueError):
            pass
    return WIDTH_M.get(tags["highway"], NARROW_M)


def _colour(tags):
    s = tags.get("surface")
    if s in SURFACE:
        return np.array(SURFACE[s])
    return np.array(SURFACE["concrete" if tags["highway"] in PATHS or s == "paving" else "asphalt"])


def lines(elements, to_xy):
    """[(centreline (n, 2) east/north metres, width m, colour)] of the roads among
    osm.fetch's elements that are not tunnels or bridges -- neither lies on
    the ground."""
    out = []
    for e in elements:
        tags = e.get("tags", {})
        if e["type"] != "way" or "highway" not in tags or tags["highway"] in SKIP:
            continue
        if tags.get("tunnel", "no") != "no" or tags.get("bridge", "no") != "no" or not e.get("geometry"):
            continue
        if len(e["geometry"]) >= 2:
            out.append((to_xy(e["geometry"]), _width(tags), _colour(tags)))
    return out


def bridges(elements, to_xy):
    """As lines, the roads among osm.fetch's elements that are bridges."""
    out = []
    for e in elements:
        tags = e.get("tags", {})
        if e["type"] != "way" or "highway" not in tags or tags["highway"] in SKIP:
            continue
        if tags.get("bridge", "no") == "no" or len(e.get("geometry") or []) < 2:
            continue
        out.append((to_xy(e["geometry"]), _width(tags), _colour(tags)))
    return out


def bridge_points(bridges, step, ground):
    """(points (n, 3) world, colours (n, 3)): each bridge's deck, from the
    ground at its first end to the ground at its last, arched, and its edge
    down each side. step, ground as points'."""
    pts, cols = [], []
    for xy, width, colour in bridges:
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        length = seg.sum()
        if length < 1e-6:
            continue
        s = float(step(xy.mean(0)[None])[0])
        if s > width:
            continue
        # the centreline every s, and how far along it each is (0-1)
        at = np.arange(0, length + 1e-9, s)
        cum = np.r_[0, np.cumsum(seg)]
        centre = np.stack([np.interp(at, cum, xy[:, 0]), np.interp(at, cum, xy[:, 1])], 1)
        t = at / length
        h0, h1 = ground(xy[[0, -1]]) + LIFT_M
        h = h0 + (h1 - h0) * t + min(ARCH_PER * length, ARCH_MAX) * 4 * t * (1 - t)
        ahead = np.gradient(centre, axis=0)
        side = np.stack([-ahead[:, 1], ahead[:, 0]], 1) / np.maximum(np.linalg.norm(ahead, axis=1), 1e-9)[:, None]
        across = np.arange(-width / 2, width / 2 + 1e-9, s)
        deck = (centre[:, None] + across[None, :, None] * side[:, None]).reshape(-1, 2)
        pts.append(np.column_stack([deck[:, 0], -np.repeat(h, len(across)), deck[:, 1]]))
        cols.append(np.tile(colour, (len(deck), 1)))
        down = np.arange(s, DECK_M + 1e-9, s)
        for edge in (-width / 2, width / 2):
            e = centre + edge * side
            pts.append(np.column_stack([np.repeat(e[:, 0], len(down)),
                                        -(h[:, None] - down[None]).ravel(), np.repeat(e[:, 1], len(down))]))
            cols.append(np.tile(SIDE, (len(e) * len(down), 1)))
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(pts), np.concatenate(cols)


def coverage(roads, step_m=0.5):
    """f(outline (n, 2)) -> the share of its edges lying on a road (within
    the road's half width, less half a metre: a pavement edge is not in it)."""
    from scipy.spatial import cKDTree
    pts, half = [], []
    for xy, width, _ in roads:
        for a, b in zip(xy[:-1], xy[1:]):
            n = max(2, int(np.linalg.norm(b - a) / step_m))
            pts.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
            half.append(np.full(n, width / 2 - 0.5))
    if not pts:
        return lambda xy: 0.0
    tree, half = cKDTree(np.concatenate(pts)), np.concatenate(half)

    def f(xy):
        e = np.concatenate([a + (c - a) * np.linspace(0, 1, max(2, int(np.linalg.norm(c - a) / step_m)))[:, None]
                            for a, c in zip(xy[:-1], xy[1:])])
        d, k = tree.query(e)
        return float((d < half[k]).mean())
    return f


def near(roads, margin_m, step_m=1.0):
    """f(east/north (n, 2)) -> whether within margin_m of a road's side."""
    from scipy.spatial import cKDTree
    pts, half = [], []
    for xy, width, _ in roads:
        for a, b in zip(xy[:-1], xy[1:]):
            n = max(2, int(np.linalg.norm(b - a) / step_m))
            pts.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
            half.append(np.full(n, width / 2 + margin_m))
    if not pts:
        return lambda xy: np.zeros(len(xy), bool)
    tree, half = cKDTree(np.concatenate(pts)), np.concatenate(half)
    reach = float(half.max())

    def f(xy):
        d, k = tree.query(xy, distance_upper_bound=reach)
        return d < half[np.minimum(k, len(half) - 1)]
    return f


def points(roads, step, ground):
    """(points (n, 3) world, colours (n, 3)) for every road.

    step(xy): point spacing at east/north points; ground(xy): height of the
    ground there."""
    strips, cols = [], []
    for xy, width, colour in roads:
        for a, b in zip(xy[:-1], xy[1:]):
            length = np.linalg.norm(b - a)
            if length < 1e-6:
                continue
            s = float(step(((a + b) / 2)[None])[0])
            if s > width:
                continue
            along = a + (b - a) * (np.arange(max(1, int(length / s)) + 1) / max(1, int(length / s)))[:, None]
            side = np.array([-(b - a)[1], (b - a)[0]]) / length
            across = np.arange(-width / 2, width / 2 + 1e-9, s)
            strips.append((along[:, None, :] + across[None, :, None] * side).reshape(-1, 2))
            cols.append(np.tile(colour, (len(strips[-1]), 1)))
    if not strips:
        return np.zeros((0, 3)), np.zeros((0, 3))
    xy = np.concatenate(strips)
    h = ground(xy) + LIFT_M
    return np.column_stack([xy[:, 0], -h, xy[:, 1]]), np.concatenate(cols)
