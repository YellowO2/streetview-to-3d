"""Buildings as the place builds them: what a building looks like from
what OpenStreetMap says of it -- its kind first, where it stands for the
rest. Most outlines say nothing (Matsumoto, Tokyo, New York: nine in ten
are just building=yes; Stockholm's old town, where they say most, three
in four say more), so where it stands decides those:

  - its region (region: the world in a few boxes) shapes an untagged
    roof (ROOFS: a small building's and a big one's, at its pitch) and
    its facade's rhythm (FACADES: facade_geometry.profile_for)
  - a small one (under SMALL_M2) of guessed height a house's (HOUSE_M),
    not a block's
  - a landmark its kind says (landmark) drawn as one, out of building
    parts standing on its foot, so roofs, points and colours stay the
    building code's own:
      castle (a Japanese one: historic=castle and castle_type=shiro, or
      in Japan): a stone base and tiers, each smaller, each roofed
      pagoda: tiers, a spire on top
      temple, shrine: one great roof, hipped or gabled
      church: a steep roof, a tower at its west end with a spire (an
      onion if Orthodox, low if southern)
      mosque: a dome in its middle, a minaret at a corner

All of it a likeness, not a survey: a building's own tags (roof:shape,
height, parts) always stand. East/north metres, heights above its foot.
"""
import math

import numpy as np
from shapely.geometry import Polygon

from streetview_to_3d.postprocess.roofs import Roof

# (region, lat from, to, lon from, to): the first that holds a place
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
# an untagged roof by region: (a small building's, a big one's), its pitch in degrees
ROOFS = {
    "japan": ("hipped", "flat", 25), "korea": ("hipped", "flat", 25), "east_asia": ("hipped", "flat", 25),
    "southeast_asia": ("hipped", "flat", 25), "europe_south": ("hipped", "hipped", 20),
    "europe_north": ("gabled", "flat", 38), "north_america": ("gabled", "flat", 30),
}
OTHER_ROOF = ("flat", "flat", 30)
SMALL_M2, SMALL_LEVELS = 150.0, 3     # a small building: under this, no more storeys than this
HOUSE_M = 7.0                         # a small one's guessed height
FLAT_KINDS = {"industrial", "warehouse", "retail", "commercial", "office", "supermarket", "garage",
              "garages", "carport", "parking", "service", "roof", "construction", "greenhouse"}
# a facade's rhythm by region and kind (facade_geometry.Profile's fields after its name):
# bay, window width, window of a floor, frame depth, balconies, ledge depth, pilasters
FACADES = {
    ("east_asia", "house"): (1.8, 1.6, .5, .12, False, .25, False),     # a ken apart, wide low windows
    ("east_asia", "apartments"): (3.0, 1.7, .6, .18, True, .3, False),  # a balcony every flat, no ornament
}
# landmarks
STONE = (0.56, 0.54, 0.5)
PLASTER = (0.93, 0.92, 0.88)
CASTLE_BASE, CASTLE_TIER_M, CASTLE_SHRINK = 0.22, 4.5, 0.12  # its base, of its height; a tier; each smaller
PAGODA_TIER_M, PAGODA_SHRINK = 3.2, 0.11
TIER_WALL = 0.55              # of a tier, its walls; its roof runs on into the next one's
TIER_PITCH, TOP_PITCH = 28, 35
SPIRE_MIN_M = 0.7             # a pagoda's spire's radius at least (a roof needs a square metre)
TEMPLE_ROOF = 0.45            # of its height, a temple's roof
CHURCH_PITCH, CHURCH_WALL_M = 45, 10.0   # a guessed nave's walls this tall
TOWER_SIDE, TOWER_MIN_M, TOWER_MAX_M = 0.35, 4.0, 10.0  # of the nave's width
SPIRE = {"north": ("pyramidal", 2.5), "south": ("pyramidal", 0.6), "orthodox": ("onion", 1.2)}  # of the side
DOME, MINARET_M, MINARET_TOP = 0.32, 1.6, 3.0  # of its width, a dome across; a minaret's radius; its cone
# windows by kind (building, else building:part): none on what has none, few on a hall -- one
# tall row a storey (FEW_ROW_M at most), FEW_BAY_M apart -- the rest a window a bay, a row a floor
NO_WINDOWS = {"shed", "garage", "garages", "carport", "roof", "hut", "greenhouse", "storage_tank", "silo",
              "bunker", "ruins", "container", "transformer_tower", "water_tower", "service", "construction",
              "bridge", "base", "spire", "dome", "minaret"}
FEW_WINDOWS = {"church", "cathedral", "chapel", "mosque", "temple", "shrine", "industrial", "warehouse",
               "manufacture", "barn", "stable", "cowshed", "farm_auxiliary", "sports_hall", "hangar", "tier",
               "tower"}
FEW_BAY_M, FEW_ROW_M = 5.0, 6.0   # a hall's windows this far apart; a row at most this tall


def region(lat, lon):
    for name, a, b, c, d in REGIONS:
        if a <= lat <= b and c <= lon <= d:
            return name
    return "other"


def facade(tags, kind):
    """Its region's facade rhythm for kind ("house", "apartments", ...),
    or None: the building code's own."""
    r = tags.get("style:region")
    return FACADES.get(("east_asia" if r in EAST_ASIA else r, kind))


def windows(tags):
    """"none", "few" or "many": how many windows a building of these tags has."""
    kind = tags.get("building", tags.get("building:part", "yes"))
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
    if (ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1]).sum() < 0:
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
    """One more part (the building code's outline tuple), standing on
    under's (a Form's) foot, else its own."""
    roof.height = min(roof.height, max(top - base, 0.0))
    form = form_cls(roof, base, True, None if colour is None else np.array(colour), tags=tags,
                    seed=seed, stand_on=under)
    return xy, top, False, form


def _tiers(xy, h, form, n, tier_m, shrink, base_frac, spire, form_cls):
    """A stone base (base_frac of it, if any) and n tiers, each shrink
    smaller than the first, each roofed, its roof running on into the
    next one's walls; a spire on top if spire. All stand on the lowest
    part's foot."""
    centre, u, v, long_, short = _box(xy)
    base_m = base_frac * n * tier_m / (1 - base_frac)
    if form.tags.get("height"):                     # its own height: the tiers fill it
        base_m = base_frac * h
        tier_m = (h - base_m) / n
    out = []
    if base_m:
        out.append(_part(form_cls, xy, base_m, 0.0, Roof(xy, "flat"), STONE, {"building:part": "base"},
                         None, form.seed))
    z, first = base_m, 0.92 if base_m else 1.0     # the walls set back from the stone's edge
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
        if guessed:                                  # its nave's walls, not a barn's
            h = max(h, form.base_m + form.roof.height + CHURCH_WALL_M)
        form.roof.height = min(form.roof.height, h - form.base_m)
    side = float(np.clip(TOWER_SIDE * short * 2, TOWER_MIN_M, min(TOWER_MAX_M, short)))
    if short > 0.9 * long_ and abs(u[0]) < abs(v[0]):  # near square: along east-west
        u, v, long_, short = v, u, short, long_
    west = -1 if u[0] > 0 else 1                     # the long axis's west end: a tower faces west
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
    """outlines (buildings.outlines') as the place builds: untagged roofs
    and heights by region, landmarks drawn as theirs. form_cls:
    buildings.Form. Returns the new outlines."""
    shape_small, shape_big, pitch = ROOFS.get(place, OTHER_ROOF)
    out = []
    for xy, h, guessed, form in outlines:
        form.tags["style:region"] = place
        kind = form.tags.get("building", form.tags.get("building:part", "yes"))
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
