"""The roads around the scene, from OpenStreetMap, laid on the terrain.

OSM draws a road as its centreline, so each is given a width -- its own
"width" tag, else "lanes" x LANE_M, else a guess by kind (WIDTH_M) -- and
laid as a strip of points LIFT_M above the ground under it, spaced as the
terrain is at that distance. Paths, steps and the like are narrow ones.

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
COLOUR = np.array([1.0, 0.2, 0.8])     # bright, to judge the fit against DA3's own roads


def _width(tags):
    for key, per in (("width", 1.0), ("lanes", LANE_M)):
        try:
            v = float(str(tags[key]).split()[0].replace(",", "."))
            if v > 0:
                return v * per
        except (KeyError, ValueError):
            pass
    return WIDTH_M.get(tags["highway"], NARROW_M)


def lines(elements, to_xy):
    """[(centreline (n, 2) east/north metres, width m)] of the roads among
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
            out.append((to_xy(e["geometry"]), _width(tags)))
    return out


def points(roads, step, ground):
    """(points (n, 3) world, colours (n, 3)) for every road.

    step(d): spacing at d metres from the centre; ground(xy): height of the
    ground at east/north points."""
    strips = []
    for xy, width in roads:
        for a, b in zip(xy[:-1], xy[1:]):
            length = np.linalg.norm(b - a)
            if length < 1e-6:
                continue
            s = step(float(np.linalg.norm((a + b) / 2)))
            along = a + (b - a) * (np.arange(max(1, int(length / s)) + 1) / max(1, int(length / s)))[:, None]
            side = np.array([-(b - a)[1], (b - a)[0]]) / length
            across = np.arange(-width / 2, width / 2 + 1e-9, s) if width > s else np.array([0.0])
            strips.append((along[:, None, :] + across[None, :, None] * side).reshape(-1, 2))
    if not strips:
        return np.zeros((0, 3)), np.zeros((0, 3))
    xy = np.concatenate(strips)
    h = ground(xy) + LIFT_M
    return np.column_stack([xy[:, 0], -h, xy[:, 1]]), np.tile(COLOUR, (len(xy), 1))
