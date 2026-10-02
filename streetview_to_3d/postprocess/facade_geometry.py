"""OSM facade profiles adapted to the vendored Apache-2.0 building grammar.

Upstream provides bay placement, window/sill geometry, balcony slabs and rails,
entrances and semantic facade motifs. This adapter adds point-readable depth,
keeps our mapped roofs, and returns coloured quads in our x-east/y-down/z-north frame.
"""
from dataclasses import dataclass
import re

import numpy as np

from ._vendor.osm_building_grammar.config import (
    BuildingGrammarConfig, FacadeStyleConfig, RoofStyleConfig,
)
from ._vendor.osm_building_grammar.grammar import generate_building_spec, _oriented_box, window_offsets


MAX_DETAIL_GAP = 1.6
SUPPORTED = {"yes", "apartments", "residential", "house", "detached", "semidetached_house",
             "terrace", "office", "commercial", "retail", "hotel", "industrial", "warehouse"}


@dataclass(frozen=True)
class Profile:
    name: str
    bay: float
    window_width: float
    window_ratio: float
    frame_depth: float
    balcony: bool
    ledge_depth: float
    pilasters: bool


def profile_for(tags, height, area):
    kind = tags.get("building", tags.get("building:part", "yes"))
    material = tags.get("building:material", "").lower()
    architecture = tags.get("building:architecture", "").lower()
    year = re.match(r"\d{4}", str(tags.get("start_date", "")))
    historic = (year and int(year[0]) < 1945) or any(
        term in architecture for term in ("baroque", "renaissance", "classic", "art_nouveau", "jugendstil"))
    from streetview_to_3d.postprocess.styles import facade
    regional = facade(tags, "house" if kind in ("house", "detached", "semidetached_house", "terrace") or historic
                      or (kind == "yes" and height < 10 and area < 180) else "apartments")
    if regional and kind not in ("industrial", "warehouse", "office", "commercial", "retail"):
        return Profile("regional", *regional)
    if kind in ("industrial", "warehouse"):
        return Profile("industrial", 5.5, 3.2, .48, .22, False, .3, False)
    if kind in ("office", "commercial") or material == "glass":
        return Profile("office", 3.2, 2.35, .73, .28, False, .45, True)
    if kind == "retail" or tags.get("shop"):
        return Profile("retail", 3.4, 2.35, .7, .32, False, .45, True)
    if historic or material in ("stone", "sandstone", "limestone", "brick"):
        return Profile("historic_urban", 3.15, 1.45, .61, .38, False, .5, True)
    if kind in ("house", "detached", "semidetached_house", "terrace") or (kind == "yes" and height < 10 and area < 180):
        return Profile("house", 3.0, 1.45, .56, .32, False, .3, False)
    return Profile("urban_apartments", 3.35, 1.75, .62, .35, True, .4, True)


def enabled(form, height, gap):
    kind = form.tags.get("building", form.tags.get("building:part", "yes"))
    edge_lengths = np.linalg.norm(np.diff(np.asarray(form.roof.poly.exterior.coords), axis=0), axis=1)
    return (kind in SUPPORTED and gap <= MAX_DETAIL_GAP and edge_lengths.max(initial=0) >= 3 and
            4 <= height - form.base_m - form.roof.height <= 90 and
            12 <= form.roof.poly.area <= 6000)


def _faces(vertices, faces):
    v = np.asarray(vertices, float)
    for face in faces:
        if len(face) == 4:
            yield v[face]


def facade_quads(xy, height, form, foot, colour, planes=None):
    """Build repeatable facade volumes without changing the existing roof or footprint.

    Fitted DA3 walls and shared walls receive no guessed ornaments. Balcony faces
    form real slabs and three-sided guards, not a window-colour pattern.
    """
    cache_key = (height, foot, tuple(colour), tuple(sorted((planes or {}).keys())))
    if cache_key in form.geometry_cache:
        yield from form.geometry_cache[cache_key]
        return
    profile = profile_for(form.tags, height, form.roof.poly.area)
    wall_height = height - form.roof.height - form.base_m
    raw_levels = _positive_number(form.tags.get("building:levels"))
    min_levels = _positive_number(form.tags.get("building:min_level")) or 0
    levels = int(np.clip(round(raw_levels - min_levels) if raw_levels else round(wall_height / 3.2), 1, 28))
    floor = wall_height / levels
    style = FacadeStyleConfig(name=profile.name, wall_color=(*colour, 1.0))
    trim = np.clip(np.asarray(colour) * .78 + .2, 0, 1)
    metal = np.clip(np.asarray(colour) * .35 + .04, 0, 1)
    style.window.spacing = profile.bay
    style.window.width = profile.window_width
    style.window.height = floor * profile.window_ratio
    style.window.sill_height = floor * .23
    style.window.depth = .06
    style.window.frame_width = .18
    style.window.frame_depth = profile.frame_depth
    style.window.frame_color = (*trim, 1.0)
    style.window.vertical_mullions = 1
    style.window.sill_depth = .48
    style.window.sill_thickness = .16
    style.window.sill_color = (*trim, 1.0)
    style.balcony.enabled = profile.balcony and levels >= 3
    style.balcony.width = min(profile.bay - .45, 2.6)
    style.balcony.depth = 1.15
    style.balcony.slab_height = .22
    style.balcony.every_n_floors = 2
    style.balcony.railing_height = 1.0
    style.balcony.railing_bar_count = 0  # broad guard faces survive point sampling
    style.balcony.railing_bar_depth = .12
    style.balcony.color = (*trim, 1.0)
    style.balcony.railing_color = (*metal, 1.0)
    style.ledge.enabled = False  # upstream shelves are planes; below we emit thick slabs
    style.door.enabled = not form.base_m and not form.part
    style.door.width = min(2.1, profile.bay - .5)
    style.door.height = min(2.7, floor - .35)
    style.door.frame_width = .28
    style.door.frame_depth = .38
    style.door.frame_color = (*trim, 1.0)
    style.door.handle_enabled = False
    style.door.canopy_enabled = True
    style.door.canopy_width = 2.8
    style.door.canopy_depth = 1.2
    style.door.canopy_thickness = .22
    style.door.canopy_color = (*trim, 1.0)
    style.antenna.enabled = False
    config = BuildingGrammarConfig(styles=[style], roof=RoofStyleConfig(type="flat", edge_enabled=False))
    tags = dict(form.tags, height=str(wall_height), **{"building:levels": str(levels),
                "roof:shape": "flat", "roof:height": "0", "grammar:street_facing_side": str(form.front)})
    # Upstream remains responsible for facade modules; our original roof implementation stays authoritative.
    footprint = [(float(x), float(y), 0.0) for x, y in xy[:-1]]
    spec = generate_building_spec(footprint, tags, config, source_name="osm")
    allowed = {"window", "window_frame", "window_mullion", "window_sill", "balcony", "balcony_rail",
               "door", "door_frame", "door_canopy", "awning", "signboard", "garage_door", "loading_dock"}
    skipped = set(planes or {}) | set(form.shared_edges)
    rows = []
    a, b = xy[:-1], xy[1:]
    lengths = np.linalg.norm(b - a, axis=1)
    tangents = (b - a) / np.maximum(lengths, 1e-9)[:, None]
    normals = np.c_[tangents[:, 1], -tangents[:, 0]]

    def emit(quad, rgb):
        # its colour as it is: the viewer lights it by which way it faces
        q = np.asarray(quad, float).copy()
        q[:, 2] += foot + form.base_m
        rows.append((q[:, [0, 2, 1]] * [1, -1, 1], np.clip(np.asarray(rgb), 0, 1)))

    for mesh in spec.meshes:
        if mesh.role not in allowed:
            continue
        verts = np.asarray(mesh.vertices, float)
        mid = verts[:, :2].mean(0)
        # Side identification by footprint projection also handles upstream naming variants.
        along = np.clip(np.sum((mid - a) * tangents, axis=1), 0, lengths)
        side = int(np.argmin(np.linalg.norm(mid - (a + along[:, None] * tangents), axis=1)))
        if side in skipped or lengths[side] < 2.5:
            continue
        if mesh.role in ("balcony", "balcony_rail"):
            match = re.search(r"_(\d+)_(\d+)_(\d+)$", mesh.name)
            if match and int(match[3]) % 2 != form.seed % 2:
                continue  # alternate bays instead of balconies on every window
        rgb = np.asarray(mesh.color[:3])
        if mesh.role == "window":
            rgb = np.asarray(colour) * .12 + [.22, .43, .64]
        elif mesh.role in ("door", "garage_door"):
            rgb = np.asarray(colour) * .2 + [.03, .07, .1]
        if mesh.role == "awning":
            rgb = np.asarray(colour) * .55
        for quad in _faces(mesh.vertices, mesh.faces):
            emit(quad, rgb)
            if mesh.role in ("window_frame", "window_mullion", "door_frame"):
                # Close the sides of the raised surround; upstream's front panels become volumes.
                back = quad.copy()
                back[:, :2] -= normals[side] * max(profile.frame_depth, .2)
                for k in range(4):
                    j = (k + 1) % 4
                    emit(np.array([quad[k], quad[j], back[j], back[k]]), rgb)

    for side, (start, tangent, normal, length) in enumerate(zip(a, tangents, normals, lengths)):
        if side in skipped or length < 2.5:
            continue
        for row in range(1, levels + 1):
            top = min(wall_height, row * floor)
            depth = profile.ledge_depth * (1.4 if row == levels else 1)
            center = start + tangent * length / 2 + normal * depth / 2
            vertices, faces = _oriented_box(tuple(center), tuple(tangent), tuple(normal),
                                             length, depth, .24 if row < levels else .4, max(0, top - .24))
            # The top cornice must not exceed the mapped eaves.
            vertices = [(x, y, min(z, wall_height)) for x, y, z in vertices]
            for quad in _faces(vertices, faces):
                emit(quad, trim)
        if profile.pilasters:
            offsets = window_offsets(length, .38, profile.bay * 2, .5)
            for offset in offsets:
                center = start + tangent * offset + normal * .17
                vertices, faces = _oriented_box(tuple(center), tuple(tangent), tuple(normal),
                                                 .38, .34, wall_height, 0)
                for quad in _faces(vertices, faces):
                    emit(quad, trim * .94)
    # A few roof volumes give the top silhouette something beyond a roof surface.
    # Keep them wholly inside the mapped roof and omit stacked building parts.
    if not form.part:
        from shapely.geometry import Polygon
        inset = form.roof.poly.buffer(-2)
        if not inset.is_empty:
            center_point = inset.representative_point()
            center = np.array([center_point.x, center_point.y])
            longest = int(np.argmax(lengths))
            width, depth, extra_height = (2.8, 2.0, 1.25) if form.roof.shape == "flat" else (.8, .7, 1.25)
            base = wall_height + float(form.roof.rise(center[None])[0])
            vertices, faces = _oriented_box(tuple(center), tuple(tangents[longest]), tuple(normals[longest]),
                                           width, depth, extra_height, base)
            if form.roof.poly.covers(Polygon(np.asarray(vertices)[:4, :2])):
                for quad in _faces(vertices, faces):
                    emit(quad, np.asarray(colour) * .72)
    form.geometry_cache[cache_key] = rows
    yield from rows


def _positive_number(value):
    try:
        number = float(value)
        return number if np.isfinite(number) and number > 0 else None
    except (ValueError, TypeError):
        return None
