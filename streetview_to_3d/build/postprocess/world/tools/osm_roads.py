"""Which OSM ways are roads, how wide, what colour, and where.

ways and lines give the ground-level roads; coverage and near say how much of an outline, or
whether a place, lies on or beside them.
"""
import numpy as np

from streetview_to_3d.build.postprocess.world.tools.osm_tags import tag_number

LANE_M = 3.2
WIDTH_M = {"motorway": 14, "trunk": 12, "primary": 10, "secondary": 9, "tertiary": 8,
           "motorway_link": 6, "trunk_link": 6, "primary_link": 6, "secondary_link": 6,
           "tertiary_link": 6, "residential": 6, "unclassified": 6, "living_street": 5,
           "service": 4, "pedestrian": 4, "track": 3}
NARROW_M = 2.0            # footways, paths and anything not in WIDTH_M
SKIP = ("proposed", "construction", "raceway", "bus_stop", "platform", "elevator", "corridor")
SURFACE = {"asphalt": (0.33, 0.33, 0.35), "concrete": (0.60, 0.60, 0.58), "paving_stones": (0.58, 0.51, 0.45),
           "sett": (0.50, 0.47, 0.44), "tiles": (0.63, 0.54, 0.47), "wood": (0.50, 0.38, 0.26),
           "gravel": (0.58, 0.55, 0.49), "fine_gravel": (0.62, 0.58, 0.50), "compacted": (0.58, 0.54, 0.46),
           "ground": (0.48, 0.41, 0.31), "dirt": (0.48, 0.41, 0.31), "earth": (0.48, 0.41, 0.31),
           "sand": (0.72, 0.66, 0.52), "grass": (0.36, 0.50, 0.30)}
PATHS = ("footway", "path", "pedestrian", "steps", "cycleway", "bridleway")
KEPT = ("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "road",
        "living_street", "pedestrian", "motorway_link", "trunk_link", "primary_link", "secondary_link",
        "tertiary_link")


def width(tags):
    """A road's width (m): its width tag, else its lanes, else its kind's."""
    for key, per in (("width", 1.0), ("lanes", LANE_M)):
        v = tag_number(tags, key)
        if v is not None and v > 0:
            return v * per
    return WIDTH_M.get(tags["highway"], NARROW_M)


def colour(tags):
    """A road's surface colour."""
    s = tags.get("surface")
    if s in SURFACE:
        return np.array(SURFACE[s])
    return np.array(SURFACE["concrete" if tags["highway"] in PATHS or s == "paving" else "asphalt"])


def on_ground(tags):
    """Whether a way is on open ground: no tunnel, bridge, indoor way or non-zero layer."""
    if tags.get("tunnel", "no") != "no" or tags.get("bridge", "no") != "no" or tags.get("indoor", "no") != "no":
        return False
    try:
        return float(str(tags.get("layer", "0")).split(";")[0]) == 0
    except ValueError:
        return True


def is_area(e):
    """Whether a way is a square (area=yes, closed) rather than a line."""
    g = e.get("geometry") or []
    return e.get("tags", {}).get("area") == "yes" and len(g) >= 4 and g[0] == g[-1]


def ways(elements, to_xy):
    """(centreline (n, 2) east/north, width m, colour, tags) of the ground-level roads, squares excluded."""
    for e in elements:
        tags = e.get("tags", {})
        if e["type"] != "way" or "highway" not in tags or tags["highway"] in SKIP:
            continue
        if not on_ground(tags) or is_area(e) or len(e.get("geometry") or []) < 2:
            continue
        yield to_xy(e["geometry"]), width(tags), colour(tags), tags


def kept_ways(elements, to_xy):
    """[(OSM id, centreline (n, 2), width, node ids, is a bridge)] of kept ground ways and bridges."""
    out = []
    for e in elements:
        tags = e.get("tags", {})
        if e["type"] != "way" or tags.get("highway") not in KEPT or is_area(e) or len(e.get("geometry") or []) < 2:
            continue
        bridge = tags.get("bridge", "no") != "no"
        if bridge or on_ground(tags):
            out.append((e["id"], to_xy(e["geometry"]), width(tags), e.get("nodes") or [], bridge))
    return out


def lines(elements, to_xy):
    """[(centreline, width, colour)] of the ground-level roads."""
    return [(xy, w, c) for xy, w, c, _ in ways(elements, to_xy)]


def coverage(roads, step_m=0.5):
    """f(outline (n, 2)) -> share of its edges on a road (within half width less 0.5 m)."""
    from scipy.spatial import cKDTree
    pts, half = [], []
    for xy, width_, _ in roads:
        for a, b in zip(xy[:-1], xy[1:]):
            n = max(2, int(np.linalg.norm(b - a) / step_m))
            pts.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
            half.append(np.full(n, width_ / 2 - 0.5))
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
    for xy, width_, _ in roads:
        for a, b in zip(xy[:-1], xy[1:]):
            n = max(2, int(np.linalg.norm(b - a) / step_m))
            pts.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
            half.append(np.full(n, width_ / 2 + margin_m))
    if not pts:
        return lambda xy: np.zeros(len(xy), bool)
    tree, half = cKDTree(np.concatenate(pts)), np.concatenate(half)
    reach = float(half.max())

    def f(xy):
        d, k = tree.query(xy, distance_upper_bound=reach)
        return d < half[np.minimum(k, len(half) - 1)]
    return f
