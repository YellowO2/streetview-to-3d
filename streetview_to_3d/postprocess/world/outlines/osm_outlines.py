"""The buildings OSM maps round the scene, as outlines with their forms."""
import numpy as np
from scipy.spatial import cKDTree

from streetview_to_3d.postprocess.world.Form import MIN_HEIGHT_M, Form
from streetview_to_3d.postprocess.world.Roof import Roof
from streetview_to_3d.postprocess.world.tools import styles
from streetview_to_3d.postprocess.world.tools.geometry import turning
from streetview_to_3d.postprocess.world.tools.osm_tags import LEVEL_M, building_kind, tag_number

DEFAULT_M = {"house": 7, "detached": 7, "semidetached_house": 7, "terrace": 7, "bungalow": 5,
             "roof": 4, "shed": 3, "garage": 3, "garages": 3, "carport": 3, "hut": 3, "kiosk": 3}
DEFAULT_OTHER_M = 12.0
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


def osm_outlines(elements, site):
    """[(closed outline (n, 2) east/north, height m, guessed, Form)] of OSM's buildings and
    building parts round site (Site): heights and roofs from tags or regional guesses
    (tools/styles.py); a building with parts is replaced by them; shared walls marked."""
    return _outlines(elements, site.to_xy)


def _number(tags, key):
    v = tag_number(tags, key)
    return v if v is not None and v >= 0 else None


def _height(tags):
    """(height m, whether it is only a guess)."""
    h = _number(tags, "height")
    if h:
        return h, False
    levels = _number(tags, "building:levels")
    if levels:
        return (levels + (_number(tags, "roof:levels") or 0)) * LEVEL_M, False
    return DEFAULT_M.get(building_kind(tags), DEFAULT_OTHER_M), True


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


def _rings(elements):
    """(element, closed ring) for each building or building:part outline."""
    for e in elements:
        tags = e.get("tags", {})
        if "building" not in tags and "building:part" not in tags:
            continue
        rings = [e.get("geometry")] if e["type"] == "way" else \
            [m.get("geometry") for m in e.get("members", []) if m.get("role") == "outer"]
        for ring in rings:
            if ring and len(ring) >= 4 and ring[0] == ring[-1]:   # outers split over several ways are skipped
                yield e, ring


def _outlines(elements, to_xy):
    """[(closed outline (n, 2) east/north, height m, guessed, Form)]; a building with parts is replaced by them."""
    import shapely
    from shapely.geometry import Polygon
    road_xy = [to_xy(e["geometry"]) for e in elements if e.get("tags", {}).get("highway")
               and len(e.get("geometry", [])) >= 2]
    road_tree = cKDTree(np.concatenate(road_xy)) if road_xy else None
    out = []
    for e, ring in _rings(elements):
        xy = to_xy(ring)
        if turning(xy) < 0:                       # counter-clockwise, so normals face out
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
    # parts inherit style hints from their containing outline (an untyped church nave is not apartments)
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
    # regional roofs and landmarks (tools/styles.py)
    ring = next((r for _, r in _rings(elements)), None)
    out = styles.apply(out, styles.region(ring[0]["lat"], ring[0]["lon"]) if ring else "other", Form)
    # shared walls get no balconies
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
    """RGB 0-1 of an OSM colour tag ("#c8a060", "#ca6" or a name), or None."""
    t = str(text).strip().lower()
    if t.startswith("#") and len(t) in (4, 7):
        h = t[1:] if len(t) == 7 else "".join(ch * 2 for ch in t[1:])
        try:
            return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255
        except ValueError:
            return None
    return np.array(NAMED[t]) if t in NAMED else None
