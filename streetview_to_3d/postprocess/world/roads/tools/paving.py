"""How paved streets look: pavements, kerbs, paint, and their colours."""
import numpy as np

from streetview_to_3d.postprocess.world.tools.geometry import hash01
from streetview_to_3d.postprocess.world.tools.osm_tags import tag_number

LIFT_M = 0.15             # roads stand this far over the ground
STRIP_M = 2.5             # mesh corner spacing along roads
PAVEMENT = (0.64, 0.62, 0.57)
KERB = (0.74, 0.72, 0.67)
PAINT = (0.82, 0.81, 0.73)
KERB_M = .14
PAVING_JOINT = (.43, .42, .39)


def _positive(tags, key, default):
    value = tag_number(tags, key)
    return value if value is not None and np.isfinite(value) and value > 0 else default


def sidewalks(xy, width, tags):
    """Tagged left/right ribbons; 'separate' is handled by mapped footway ways."""
    import shapely
    line = shapely.LineString(xy)
    for side, sign in (("left", 1), ("right", -1)):
        value = tags.get("sidewalk:" + side, tags.get("sidewalk:both", tags.get("sidewalk", "no")))
        if value not in ("yes", "both", side):
            continue
        w = min(6.0, _positive(tags, f"sidewalk:{side}:width",
                            _positive(tags, "sidewalk:both:width", 1.8)))
        # one-sided buffers keep turns joined without overlapping the road
        outer = line.buffer(sign * (width / 2 + w), single_sided=True, join_style="mitre")
        inner = line.buffer(sign * width / 2, single_sided=True, join_style="mitre")
        yield shapely.difference(outer, inner), line.offset_curve(sign * (width / 2 + w / 2)), w


def lane_dividers(xy, width, tags):
    """Conservative dashed dividers from explicit lane counts, clear of segment ends."""
    import shapely
    if tags.get("lane_markings") == "no" or tags.get("surface", "asphalt") not in ("asphalt", "concrete"):
        return
    lanes = _positive(tags, "lanes", 1)
    if lanes != int(lanes) or not 2 <= lanes <= 8 or width / lanes < 2.4:
        return
    line = shapely.LineString(xy)
    for k in range(1, int(lanes)):
        offset = width * (k / lanes - .5)
        stripe = line if abs(offset) < 1e-6 else line.offset_curve(offset)
        if stripe.geom_type != "LineString":
            continue
        for start in np.arange(width, stripe.length - width - 2.5, 6):
            from shapely.ops import substring
            yield substring(stripe, start, start + 2.5).buffer(.09, cap_style="flat")


def surface_colours(xy, colour):
    """Quiet, deterministic variation without high-contrast noise."""
    colour = np.asarray(colour)
    cell = np.floor(xy / .8)
    n = hash01(cell[:, 0], cell[:, 1])
    return np.clip(colour * (1 + (n[:, None] - .5) * .055), 0, 1)


def curb_faces(net, ground, where):
    """Vertical kerb faces along raised pavements within where."""
    import shapely
    pavement = shapely.union_all([net.shapes.get(PAVEMENT, shapely.Polygon()),
                                  net.shapes.get(KERB, shapely.Polygon()), net.shapes.get(PAVING_JOINT, shapely.Polygon())])
    boundary = shapely.intersection(shapely.boundary(pavement), where)
    for part in shapely.get_parts(shapely.segmentize(boundary, STRIP_M)):
        if part.geom_type not in ("LineString", "LinearRing"):
            continue
        xy = shapely.get_coordinates(part)
        foot = ground(xy) + LIFT_M
        for k in range(len(xy) - 1):
            en, bottom = xy[k:k + 2], foot[k:k + 2]
            yield np.r_[np.c_[en[:, 0], -bottom, en[:, 1]],
                        np.c_[en[::-1, 0], -(bottom[::-1] + KERB_M), en[::-1, 1]]]
