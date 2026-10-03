"""How a building looks from what OSM says of it and where it stands: regional roofs and facades
for untagged buildings, small guessed ones as houses, and landmarks built from parts."""
import math

import numpy as np
from shapely.geometry import Polygon

from streetview_to_3d.postprocess.world.geometry import turning
from streetview_to_3d.postprocess.world.osm import building_kind
from streetview_to_3d.postprocess.world.roofs import Roof

# (region, lat from, to, lon from, to): the first match wins
REGIONS = [
    ("korea", 33.0, 38.7, 124.5, 129.6),
    ("japan", 24.0, 46.0, 122.9, 146.0),
    ("southeast_asia", -11.0, 21.5, 92.0, 141.0),
    ("south_asia", 5.0, 35.0, 60.0, 89.0),
    ("east_asia", 18.0, 54.0, 89.0, 135.0),
    ("europe_south", 34.0, 45.5, -10.0, 30.0),
    ("europe_north", 45.5, 72.0, -25.0, 45.0),
    ("middle_east", 12.0, 42.0, -18.0, 60.0),
    ("north_america", 24.0, 72.0, -170.0, -50.0),
    ("latin_america", -56.0, 24.0, -118.0, -34.0),
]
EAST_ASIA = ("japan", "korea", "east_asia")
# untagged roof by region: (small building's, big building's, pitch in degrees)
ROOFS = {
    "japan": ("hipped", "flat", 25), "korea": ("hipped", "flat", 25), "east_asia": ("hipped", "flat", 25),
    "southeast_asia": ("hipped", "flat", 25), "europe_south": ("hipped", "hipped", 20),
    "europe_north": ("gabled", "flat", 38), "north_america": ("gabled", "flat", 30),
}
OTHER_ROOF = ("flat", "flat", 30)
SMALL_M2, SMALL_LEVELS = 150.0, 3     # a small building: under this area, at most this many storeys
HOUSE_M = 7.0                         # guessed height of a small building
FLAT_KINDS = {"industrial", "warehouse", "retail", "commercial", "office", "supermarket", "garage",
              "garages", "carport", "parking", "service", "roof", "construction", "greenhouse"}
# facade_geometry.Profile fields by (region, kind):
# bay, window width, window share of a floor, frame depth, balconies, ledge depth, pilasters
FACADES = {
    ("east_asia", "house"): (1.8, 1.6, .5, .12, False, .25, False),     # a ken apart, wide low windows
    ("east_asia", "apartments"): (3.0, 1.7, .6, .18, True, .3, False),  # a balcony every flat, no ornament
}
# landmarks
STONE = (0.56, 0.54, 0.5)
PLASTER = (0.93, 0.92, 0.88)
CASTLE_BASE, CASTLE_TIER_M, CASTLE_SHRINK = 0.22, 4.5, 0.12  # base share of height; tier height; shrink per tier
PAGODA_TIER_M, PAGODA_SHRINK = 3.2, 0.11
TIER_WALL = 0.55              # wall share of a tier; its roof runs into the next
TIER_PITCH, TOP_PITCH = 28, 35
SPIRE_MIN_M = 0.7             # min pagoda spire radius
TEMPLE_ROOF = 0.45            # temple roof share of its height
CHURCH_PITCH, CHURCH_WALL_M = 45, 10.0   # roof pitch; wall height of a guessed nave
TOWER_SIDE, TOWER_MIN_M, TOWER_MAX_M = 0.35, 4.0, 10.0  # tower side as share of nave width, limits
SPIRE = {"north": ("pyramidal", 2.5), "south": ("pyramidal", 0.6), "orthodox": ("onion", 1.2)}  # height per side
DOME, MINARET_M, MINARET_TOP = 0.32, 1.6, 3.0  # dome radius share of width; minaret radius; cone height ratio
# windows by kind: none, few (halls: tall rows at most FEW_ROW_M, FEW_BAY_M apart) or many
NO_WINDOWS = {"shed", "garage", "garages", "carport", "roof", "hut", "greenhouse", "storage_tank", "silo",
              "bunker", "ruins", "container", "transformer_tower", "water_tower", "service", "construction",
              "bridge", "base", "spire", "dome", "minaret"}
FEW_WINDOWS = {"church", "cathedral", "chapel", "mosque", "temple", "shrine", "industrial", "warehouse",
               "manufacture", "barn", "stable", "cowshed", "farm_auxiliary", "sports_hall", "hangar", "tier",
               "tower"}
FEW_BAY_M, FEW_ROW_M = 5.0, 6.0


def region(lat, lon):
    for name, a, b, c, d in REGIONS:
        if a <= lat <= b and c <= lon <= d:
            return name
    return "other"


def facade(tags, kind):
    """The region's facade profile for kind ("house", "apartments"), or None."""
    r = tags.get("style:region")
    return FACADES.get(("east_asia" if r in EAST_ASIA else r, kind))


def windows(tags):
    """"none", "few" or "many": how many windows a building of these tags has."""
    kind = building_kind(tags)
    if kind in NO_WINDOWS:
        return "none"
    if kind in FEW_WINDOWS or tags.get("amenity") == "place_of_worship":
        return "few"
    return "many"


def landmark(tags, place):
    b, rel = tags.get("building"), tags.get("religion")
    worship = tags.get("amenity") == "place_of_worship"
    if tags.get("historic") == "castle" and (tags.get("castle_type") in ("shiro", "japanese") or place == "japan") \
            or (b == "castle" and place == "japan"):
        return "castle"
    if b == "pagoda" or tags.get("tower:type") == "pagoda":
        return "pagoda"
    if b == "shrine" or worship and rel == "shinto":
        return "shrine"
    if b == "temple" or worship and rel == "buddhist":
        return "temple"
    if b in ("church", "cathedral", "chapel") or worship and rel == "christian":
        return "church"
    if b == "mosque" or worship and rel == "muslim":
        return "mosque"
    return None


def _box(xy):
    """(centre, long axis, short axis (unit), long, short) of xy's smallest box."""
    with np.errstate(invalid="ignore", divide="ignore"):    # a square's: shapely divides by 0
        box = np.asarray(Polygon(xy).buffer(0).minimum_rotated_rectangle.exterior.coords)[:4]
    a, b = box[1] - box[0], box[2] - box[1]
    if np.linalg.norm(a) < np.linalg.norm(b):
        a, b = b, a
    la, lb = np.linalg.norm(a), np.linalg.norm(b)
    return box.mean(0), a / max(la, 1e-9), b / max(lb, 1e-9), la, lb


def _rect(centre, u, v, long_, short):
    """A closed counter-clockwise rectangle."""
    c = [centre + s * u * long_ / 2 + t * v * short / 2 for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    ring = np.array(c + c[:1])
    if turning(ring) < 0:
        ring = ring[::-1]
    return ring


def _circle(centre, r, n=16):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = centre + r * np.c_[np.cos(a), np.sin(a)]
    return np.vstack([ring, ring[:1]])


def _pitched(xy, shape, pitch, cap):
    roof = Roof(xy, shape)
    if roof.shape != "flat":
        roof.height = min(math.tan(math.radians(pitch)) * roof.half, cap)
    return roof


def _part(form_cls, xy, top, base, roof, colour, tags, under, seed):
    """A part as an outline tuple, standing on under's foot (a Form) if given."""
    roof.height = min(roof.height, max(top - base, 0.0))
    form = form_cls(roof, base, True, None if colour is None else np.array(colour), tags=tags,
                    seed=seed, stand_on=under)
    return xy, top, False, form


def _tiers(xy, h, form, n, tier_m, shrink, base_frac, spire, form_cls):
    """A stone base (base_frac of the height) and n roofed tiers, each shrink smaller; a spire if asked."""
    centre, u, v, long_, short = _box(xy)
    base_m = base_frac * n * tier_m / (1 - base_frac)
    if form.tags.get("height"):                     # tagged height: tiers fill it
        base_m = base_frac * h
        tier_m = (h - base_m) / n
    out = []
    if base_m:
        out.append(_part(form_cls, xy, base_m, 0.0, Roof(xy, "flat"), STONE, {"building:part": "base"},
                         None, form.seed))
    z, first = base_m, 0.92 if base_m else 1.0     # walls set back from the stone's edge
    for i in range(n):
        s = first * (1 - shrink * i)
        ring = _rect(centre, u, v, long_ * s, short * s)
        roof = _pitched(ring, "hipped", TOP_PITCH if i == n - 1 else TIER_PITCH, 1e9)
        out.append(_part(form_cls, ring, z + tier_m * TIER_WALL + roof.height, z, roof,
                         PLASTER if form.colour is None else form.colour, {"building:part": "tier"},
                         out[0][3] if out else None, form.seed))
        z += tier_m
    if spire:
        top, r = out[-1][1], max(short * s * 0.08, SPIRE_MIN_M)
        ring = _circle(centre, r, 8)
        out.append(_part(form_cls, ring, top + short * s * 0.9, top - 0.5, Roof(ring, "cone", 3 * r), (0.35, 0.33, 0.3),
                         {"building:part": "spire"}, out[0][3], form.seed))
    return out


def _church(xy, h, guessed, form, place, form_cls):
    centre, u, v, long_, short = _box(xy)
    if not form.tags.get("roof:shape"):
        form.roof = _pitched(xy, "gabled", CHURCH_PITCH, 1e9)
        if guessed:
            h = max(h, form.base_m + form.roof.height + CHURCH_WALL_M)
        form.roof.height = min(form.roof.height, h - form.base_m)
    side = float(np.clip(TOWER_SIDE * short * 2, TOWER_MIN_M, min(TOWER_MAX_M, short)))
    if short > 0.9 * long_ and abs(u[0]) < abs(v[0]):  # near square: run east-west
        u, v, long_, short = v, u, short, long_
    west = -1 if u[0] > 0 else 1                     # tower at the west end
    at = centre + west * u * (long_ / 2 - side / 2)
    ring = _rect(at, u, v, side, side)
    orthodox = "orthodox" in str(form.tags.get("denomination", ""))
    shape, rise = SPIRE["orthodox" if orthodox else "south" if place in ("europe_south", "latin_america")
                        else "north"]
    walls = max(h * 1.6, h + 8)
    roof = Roof(ring, shape)
    roof.height = side * rise
    tower = _part(form_cls, ring, walls + roof.height, 0.0, roof, form.colour, {"building:part": "tower"},
                  form, form.seed)
    return [(xy, h, guessed, form), tower]


def _mosque(xy, h, guessed, form, form_cls):
    centre, u, v, long_, short = _box(xy)
    r = DOME * short
    ring = _circle(centre, r)
    dome = Roof(ring, "dome")
    dome.height = r
    corner = centre + u * (long_ / 2 - MINARET_M * 1.5) + v * (short / 2 - MINARET_M * 1.5)
    shaft = _circle(corner, MINARET_M, 8)
    cone = Roof(shaft, "cone")
    cone.height = MINARET_M * MINARET_TOP
    return [(xy, h, guessed, form),
            _part(form_cls, ring, h + r + 1.5, h - 0.5, dome, form.colour, {"building:part": "dome"}, form, form.seed),
            _part(form_cls, shaft, max(2.2 * h, h + 15) + cone.height, 0.0, cone, form.colour,
                  {"building:part": "minaret"}, form, form.seed)]


def apply(outlines, place, form_cls):
    """outlines (buildings.outlines') with regional roofs and heights and landmarks built; form_cls: buildings.Form."""
    shape_small, shape_big, pitch = ROOFS.get(place, OTHER_ROOF)
    out = []
    for xy, h, guessed, form in outlines:
        form.tags["style:region"] = place
        kind = building_kind(form.tags)
        what = None if form.part else landmark(form.tags, place)
        if what in ("castle", "pagoda"):
            levels = form.tags.get("building:levels")
            try:
                n = int(float(levels))
            except (TypeError, ValueError):
                n = 4 if what == "castle" else 5
            n = int(np.clip(n, 2, 7))
            if what == "castle":
                out += _tiers(xy, h, form, n, CASTLE_TIER_M, CASTLE_SHRINK, CASTLE_BASE, False, form_cls)
            else:
                out += _tiers(xy, h, form, n, PAGODA_TIER_M, PAGODA_SHRINK, 0.0, True, form_cls)
            continue
        if what in ("temple", "shrine") and not form.tags.get("roof:shape"):
            h = max(h, 10.0)
            shape = "hipped" if what == "temple" else "gabled"
            form.roof = Roof(xy, shape, TEMPLE_ROOF * (h - form.base_m))
            out.append((xy, h, guessed, form))
            continue
        if what == "church":
            out += _church(xy, h, guessed, form, place, form_cls)
            continue
        if what == "mosque":
            out += _mosque(xy, h, guessed, form, form_cls)
            continue
        area = Polygon(xy).buffer(0).area
        small = area < SMALL_M2
        if guessed and small and kind == "yes":
            h = HOUSE_M
        levels = form.tags.get("building:levels")
        try:
            small = small and float(levels) <= SMALL_LEVELS
        except (TypeError, ValueError):
            pass
        shape = shape_small if small else shape_big
        if not form.tags.get("roof:shape") and kind not in FLAT_KINDS and shape != "flat" and not form.part:
            form.roof = _pitched(xy, shape, pitch, max(h - form.base_m - 2.5, 0.0))
        out.append((xy, h, guessed, form))
    return out
