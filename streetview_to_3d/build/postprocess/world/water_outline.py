"""OSM's water near the scene, as one shape."""
import numpy as np
import shapely

from streetview_to_3d.build.postprocess.world.tools.reach import OSM_M


def water_outline(elements, site, osm, wet):
    """OSM's water within OSM_M of the scene (_outline), or None without OSM elements.

    site: Site; osm: OpenStreetMap, asked where a place is water; wet: WaterMap, its stand-in."""
    if not elements:
        return None
    return _outline(elements, site.to_xy, OSM_M, lambda xy: osm.water_at(np.stack(site.to_ll(xy), 1)), wet)


def _is_water(e):
    """Whether an element is a water outline: a water way, or an untagged water relation member."""
    if e.get("type") != "way":
        return False
    tags = e.get("tags", {})
    if tags.get("natural") in ("water", "coastline") or tags.get("waterway") == "riverbank" \
            or tags.get("landuse") in ("reservoir", "basin"):
        return True
    return not any(k in tags for k in ("building", "building:part", "highway"))


def _outline(elements, to_xy, box_m, water_at, wet):
    """OSM's water within the box of half-width box_m, as one shapely geometry.

    Water outlines split the box into faces; a face is water if water_at(xy)
    (OpenStreetMap.water_at, falling back to wet) says so, or if a coastline has it on
    its right. With a single face, wet (the JRC map) decides."""
    inner = shapely.box(-box_m, -box_m, box_m, box_m)
    ways = [(e, to_xy(e["geometry"])) for e in elements if _is_water(e) and len(e.get("geometry") or []) >= 2]
    lines = [inner.intersection(shapely.LineString(xy)) for _, xy in ways]
    faces = list(shapely.get_parts(shapely.polygonize([shapely.union_all(lines + [inner.exterior])])))
    if not faces:
        return shapely.Polygon()
    inside = np.array([f.representative_point().coords[0] for f in faces])
    if len(faces) == 1:
        return faces[0] if wet(inside).all() else shapely.Polygon()
    try:
        water = np.array(water_at(inside), bool)
    except (OSError, ValueError):          # Overpass busy: the JRC map decides
        water = np.asarray(wet(inside), bool)
    # the sea lies to the right of the coastline
    right, left = [], []
    for e, xy in ways:
        if e.get("tags", {}).get("natural") != "coastline":
            continue
        a, b = xy[:-1], xy[1:]
        d = b - a
        n = np.stack([d[:, 1], -d[:, 0]], 1) / np.maximum(np.linalg.norm(d, axis=1), 1e-9)[:, None]
        mid = (a + b) / 2
        right.append(mid + n)
        left.append(mid - n)
    if right:
        right, left = np.concatenate(right), np.concatenate(left)
        for k, f in enumerate(faces):
            water[k] |= shapely.contains_xy(f, *right.T).sum() > shapely.contains_xy(f, *left.T).sum()
    return shapely.union_all([f for f, w in zip(faces, water) if w]) if water.any() else shapely.Polygon()
