"""Pure rule-based building grammar expansion."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import hypot

from .config import BuildingGrammarConfig, FacadeStyleConfig, RoofStyleConfig, building_value_is_excluded, selectable_styles_for_tags


Point3D = tuple[float, float, float]
Point2D = tuple[float, float]


@dataclass(frozen=True, slots=True)
class MeshSpec:
    name: str
    role: str
    material: str
    color: tuple[float, float, float, float]
    texture_path: str | None
    vertices: list[Point3D]
    faces: list[list[int]]
    instance_count: int = 1
    texture_scale: float = 1.0
    export_parts: tuple["MeshSpec", ...] = ()


@dataclass(frozen=True, slots=True)
class BuildingSpec:
    source_name: str
    levels: int
    height: float
    floor_heights: list[float]
    meshes: list[MeshSpec] = field(default_factory=list)

    def meshes_by_role(self, role: str) -> list[MeshSpec]:
        return [mesh for mesh in self.meshes if mesh.role == role]


@dataclass(frozen=True, slots=True)
class RoofFrame:
    center: Point2D
    direction: Point2D
    normal: Point2D
    min_long: float
    max_long: float
    min_side: float
    max_side: float
    eave_z: float
    ridge_z: float


def generate_building_spec(
    footprint: list[Point3D],
    tags: dict[str, str],
    config: BuildingGrammarConfig,
    *,
    source_name: str = "building",
) -> BuildingSpec:
    if building_value_is_excluded(tags, config):
        raise ValueError(f"building value {tags.get('building')!r} is excluded")

    clean = orient_footprint_ccw(clean_footprint(footprint))
    if len(clean) < 3:
        raise ValueError("A building footprint needs at least three points")

    styles = [
        _facade_style_from_tags(style, tags)
        for style in selectable_styles_for_tags(tags, config.styles or [FacadeStyleConfig()])
    ]
    primary_style = styles[0] if styles else FacadeStyleConfig()
    roof_style = _roof_style_for_building(source_name, styles)
    levels = infer_levels(tags, config, primary_style)
    floor_heights = floor_height_sequence(levels, tags, config, primary_style)
    total_height = sum(floor_heights)
    roof = roof_style.roof if roof_style and roof_style.roof is not None else primary_style.roof or config.roof
    roof = _roof_from_tags(roof, tags)
    meshes: list[MeshSpec] = []
    street_side_index = _street_facing_side_index(clean, tags)

    floor_bottoms = _floor_bottoms(floor_heights)
    for side_index, (start, end) in enumerate(_segments(clean)):
        style = styles[(side_index - street_side_index) % len(styles)]
        normal = outward_normal(start, end, polygon_is_ccw(clean))
        length = distance_2d(start, end)
        if length == 0:
            continue
        window_positions = window_offsets(length, style.window.width, style.window.spacing, style.window.min_margin)
        door_offset = length / 2.0
        if style.wall_row_colors:
            for floor_index, floor_bottom in enumerate(floor_bottoms):
                meshes.append(
                    _wall_row_mesh(
                        source_name,
                        side_index,
                        floor_index,
                        start,
                        end,
                        floor_bottom,
                        floor_heights[floor_index],
                        style,
                    )
                )
        else:
            meshes.append(_wall_mesh(source_name, side_index, start, end, total_height, style, source_name=source_name))
        if _door_applies(style, side_index, street_side_index, tags):
            meshes.append(_door_mesh(source_name, side_index, start, end, normal, door_offset, floor_heights[0], style))
            meshes.extend(_door_detail_meshes(source_name, side_index, start, end, normal, door_offset, floor_heights[0], style))
        for floor_index, floor_bottom in enumerate(floor_bottoms):
            floor_height = floor_heights[floor_index]
            for window_index, offset in enumerate(window_positions):
                if floor_index == 0 and _door_applies(style, side_index, street_side_index, tags) and _window_overlaps_door(offset, door_offset, style):
                    continue
                meshes.append(
                    _window_mesh(
                        source_name,
                        side_index,
                        floor_index,
                        window_index,
                        start,
                        end,
                        normal,
                        offset,
                        floor_bottom,
                        floor_height,
                        style,
                    )
                )
                meshes.extend(
                    _window_detail_meshes(
                        source_name,
                        side_index,
                        floor_index,
                        window_index,
                        start,
                        end,
                        normal,
                        offset,
                        floor_bottom,
                        floor_height,
                        style,
                    )
                )
                if _balcony_applies(style, floor_index):
                    meshes.append(
                        _balcony_mesh(
                            source_name,
                            side_index,
                            floor_index,
                            window_index,
                            start,
                            end,
                            normal,
                            offset,
                            floor_bottom,
                            style,
                        )
                    )
                    meshes.extend(
                        _balcony_detail_meshes(
                            source_name,
                            side_index,
                            floor_index,
                            window_index,
                            start,
                            end,
                            normal,
                            offset,
                            floor_bottom,
                            style,
                        )
                    )
            if _ledge_applies(style, floor_index):
                meshes.append(_ledge_mesh(source_name, side_index, floor_index, start, end, normal, floor_bottom, style))
        meshes.extend(
            _facade_depth_meshes(
                source_name,
                side_index,
                start,
                end,
                normal,
                street_side_index,
                floor_heights,
                total_height,
                style,
                tags,
            )
        )

    meshes.append(_roof_mesh(source_name, clean, total_height, roof, tags))
    meshes.extend(_roof_edge_meshes(source_name, clean, total_height, roof))
    meshes.extend(_gutter_meshes(source_name, clean, total_height, roof))
    meshes.extend(_roof_detail_meshes(source_name, clean, total_height, roof, tags))
    roof_style = styles[_stable_index(source_name, len(styles))]
    meshes.extend(_roof_service_meshes(source_name, clean, total_height, roof, roof_style, tags))
    meshes.extend(_antenna_meshes(source_name, clean, total_height, roof, roof_style))
    return BuildingSpec(
        source_name=source_name,
        levels=levels,
        height=total_height,
        floor_heights=floor_heights,
        meshes=meshes,
    )


def infer_levels(tags: dict[str, str], config: BuildingGrammarConfig, style: FacadeStyleConfig | None = None) -> int:
    for key in ("building:levels", "levels"):
        value = _parse_int(tags.get(key))
        if value is not None:
            return max(value, 1)
    height = parse_meters(tags.get("height"))
    floor_height = style.default_floor_height if style and style.default_floor_height is not None else config.default_floor_height
    if height is not None:
        return max(1, round(height / floor_height))
    if style and style.default_levels is not None:
        return max(style.default_levels, 1)
    return max(config.default_levels, 1)


def floor_height_sequence(
    levels: int,
    tags: dict[str, str],
    config: BuildingGrammarConfig,
    style: FacadeStyleConfig | None = None,
) -> list[float]:
    explicit_height = parse_meters(tags.get("height"))
    default_floor_height = style.default_floor_height if style and style.default_floor_height is not None else config.default_floor_height
    heights = [
        config.irregular_floor_heights.get(index, default_floor_height)
        for index in range(levels)
    ]
    if explicit_height is not None:
        scale = explicit_height / sum(heights)
        heights = [height * scale for height in heights]
    return heights


def effective_part_min_height(tags: dict[str, str], config: BuildingGrammarConfig, style: FacadeStyleConfig | None = None) -> float:
    explicit = parse_meters(tags.get("min_height") or tags.get("building:min_height"))
    if explicit is not None:
        return max(explicit, 0.0)
    min_level = _parse_int(tags.get("building:min_level") or tags.get("min_level"))
    if min_level is None:
        return 0.0
    floor_height = style.default_floor_height if style and style.default_floor_height is not None else config.default_floor_height
    return max(min_level, 0) * floor_height


def tags_for_building_part_volume(
    tags: dict[str, str],
    config: BuildingGrammarConfig,
    style: FacadeStyleConfig | None = None,
) -> tuple[dict[str, str], float]:
    """Return generation tags and base height for an OSM building:part volume."""
    min_height = effective_part_min_height(tags, config, style)
    if min_height <= 0:
        return dict(tags), 0.0

    volume_tags = dict(tags)
    explicit_height = parse_meters(volume_tags.get("height"))
    if explicit_height is not None:
        volume_tags["grammar:part:absolute_height"] = f"{explicit_height:.6f}"
        volume_tags["height"] = f"{max(explicit_height - min_height, 0.1):.6f}"
    else:
        levels = _parse_int(volume_tags.get("building:levels") or volume_tags.get("levels"))
        min_level = _parse_int(volume_tags.get("building:min_level") or volume_tags.get("min_level"))
        if levels is not None and min_level is not None:
            effective_levels = max(levels - max(min_level, 0), 1)
            volume_tags["building:levels"] = str(effective_levels)
            volume_tags.pop("levels", None)
    volume_tags["grammar:part:min_height"] = f"{min_height:.6f}"
    volume_tags["grammar:disable_ground_entrance"] = "yes"
    return volume_tags, min_height


def parse_meters(value: str | None) -> float | None:
    if not value:
        return None
    normalized = value.strip().lower().replace("meters", "m").replace("metres", "m")
    if normalized.endswith("m"):
        normalized = normalized[:-1].strip()
    try:
        return float(normalized)
    except ValueError:
        return None


def _roof_from_tags(roof: RoofStyleConfig, tags: dict[str, str]) -> RoofStyleConfig:
    updates: dict[str, object] = {}
    roof_shape = (tags.get("roof:shape") or tags.get("roof:type") or tags.get("grammar:roof:type") or "").strip().lower()
    shape_aliases = {
        "flat": "flat",
        "gabled": "gabled",
        "gable": "gabled",
        "pitched": "gabled",
        "skillion": "gabled",
        "saltbox": "gabled",
        "gambrel": "gabled",
        "mansard": "gabled",
        "hipped": "hipped",
        "hip": "hipped",
        "half-hipped": "hipped",
        "half hipped": "hipped",
        "side_hipped": "hipped",
        "pyramid": "pyramid",
        "pyramidal": "pyramid",
    }
    if roof_shape in shape_aliases:
        updates["type"] = shape_aliases[roof_shape]
    roof_height = parse_meters(tags.get("roof:height") or tags.get("roof:levels"))
    if roof_height is not None:
        updates["height"] = max(roof_height, 0.0)
    roof_material = tags.get("roof:material") or tags.get("roof:material:name")
    if roof_material:
        updates["material"] = _semantic_material_name("Roof", roof_material)
    roof_color = _parse_osm_color(tags.get("roof:colour") or tags.get("roof:color"))
    if roof_color is not None:
        updates["color"] = roof_color
    return replace(roof, **updates) if updates else roof


def _facade_style_from_tags(style: FacadeStyleConfig, tags: dict[str, str]) -> FacadeStyleConfig:
    updates: dict[str, object] = {}
    material = tags.get("facade:material") or tags.get("building:facade:material") or tags.get("building:material")
    if material:
        updates["wall_material"] = _semantic_material_name("Facade", material)
    color = _parse_osm_color(
        tags.get("facade:colour")
        or tags.get("facade:color")
        or tags.get("building:colour")
        or tags.get("building:color")
    )
    if color is not None:
        updates["wall_color"] = color
        updates["wall_color_variants"] = []
        updates["wall_row_colors"] = []
    return replace(style, **updates) if updates else style


def _semantic_material_name(role: str, value: str) -> str:
    normalized = value.strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "brick": "Brick",
        "bricks": "Brick",
        "plaster": "Plaster",
        "render": "Render",
        "stucco": "Stucco",
        "glass": "Glass",
        "metal": "Metal",
        "steel": "Steel",
        "concrete": "Concrete",
        "wood": "Wood",
        "timber": "Timber",
        "stone": "Stone",
        "sandstone": "Sandstone",
        "tile": "Tile",
        "tiles": "Tile",
        "roof tiles": "Tile",
        "slate": "Slate",
        "copper": "Copper",
        "zinc": "Zinc",
        "asphalt": "Asphalt",
        "membrane": "Membrane",
    }
    label = aliases.get(normalized, " ".join(part.capitalize() for part in normalized.split()))
    return f"OSM {role} {label}" if label else f"OSM {role}"


def _parse_osm_color(value: str | None) -> tuple[float, float, float, float] | None:
    if not value:
        return None
    color = value.strip().lower().replace("grey", "gray")
    if ";" in color:
        color = color.split(";", 1)[0].strip()
    if color.startswith("#"):
        hex_value = color[1:]
        if len(hex_value) == 3:
            hex_value = "".join(char * 2 for char in hex_value)
        if len(hex_value) == 6:
            try:
                return (
                    int(hex_value[0:2], 16) / 255.0,
                    int(hex_value[2:4], 16) / 255.0,
                    int(hex_value[4:6], 16) / 255.0,
                    1.0,
                )
            except ValueError:
                return None
    named = {
        "black": (0.02, 0.02, 0.02, 1.0),
        "white": (0.92, 0.9, 0.86, 1.0),
        "gray": (0.45, 0.45, 0.43, 1.0),
        "silver": (0.68, 0.68, 0.66, 1.0),
        "red": (0.58, 0.12, 0.08, 1.0),
        "brown": (0.38, 0.22, 0.12, 1.0),
        "beige": (0.72, 0.66, 0.54, 1.0),
        "cream": (0.84, 0.78, 0.62, 1.0),
        "yellow": (0.78, 0.64, 0.22, 1.0),
        "orange": (0.72, 0.38, 0.12, 1.0),
        "green": (0.24, 0.42, 0.25, 1.0),
        "blue": (0.12, 0.24, 0.46, 1.0),
        "anthracite": (0.12, 0.13, 0.13, 1.0),
        "terracotta": (0.55, 0.2, 0.11, 1.0),
    }
    return named.get(color)


def window_offsets(length: float, width: float, spacing: float, margin: float) -> list[float]:
    usable = length - margin * 2.0
    if usable < width:
        return []
    count = max(1, int((usable + spacing - width) // spacing))
    total_width = (count - 1) * spacing + width
    start = (length - total_width) / 2.0 + width / 2.0
    return [start + index * spacing for index in range(count)]


def clean_footprint(footprint: list[Point3D]) -> list[Point3D]:
    clean: list[Point3D] = []
    for point in footprint:
        normalized = (float(point[0]), float(point[1]), float(point[2]))
        if not clean or normalized[:2] != clean[-1][:2]:
            clean.append(normalized)
    if len(clean) > 1 and clean[0][:2] == clean[-1][:2]:
        clean.pop()
    return clean


def polygon_is_ccw(points: list[Point3D]) -> bool:
    return signed_polygon_area(points) > 0


def signed_polygon_area(points: list[Point3D]) -> float:
    clean = clean_footprint(points)
    if len(clean) < 3:
        return 0.0
    area = 0.0
    for start, end in _segments(clean):
        area += start[0] * end[1] - end[0] * start[1]
    return area / 2.0


def orient_footprint_ccw(points: list[Point3D]) -> list[Point3D]:
    clean = clean_footprint(points)
    if len(clean) < 3 or polygon_is_ccw(clean):
        return clean
    return list(reversed(clean))


def outward_normal(start: Point3D, end: Point3D, ccw: bool) -> Point2D:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = hypot(dx, dy)
    if length == 0:
        return (0.0, 0.0)
    if ccw:
        return (dy / length, -dx / length)
    return (-dy / length, dx / length)


def _wall_mesh(
    mesh_source_name: str,
    side_index: int,
    start: Point3D,
    end: Point3D,
    height: float,
    style: FacadeStyleConfig,
    *,
    source_name: str,
) -> MeshSpec:
    color_index, color = _variant_wall_color(style, source_name, side_index)
    vertices = [
        (start[0], start[1], 0.0),
        (end[0], end[1], 0.0),
        (end[0], end[1], height),
        (start[0], start[1], height),
    ]
    return MeshSpec(
        name=f"{mesh_source_name}.facade_{side_index}",
        role="facade",
        material=_wall_material_name(style.wall_material, "variant", color_index),
        color=color,
        texture_path=style.wall_texture_path,
        vertices=vertices,
        faces=[[0, 1, 2, 3]],
        texture_scale=style.wall_texture_scale,
    )


def _wall_row_mesh(
    source_name: str,
    side_index: int,
    floor_index: int,
    start: Point3D,
    end: Point3D,
    floor_bottom: float,
    floor_height: float,
    style: FacadeStyleConfig,
) -> MeshSpec:
    color_index, color = _row_wall_color(style, floor_index)
    vertices = [
        (start[0], start[1], floor_bottom),
        (end[0], end[1], floor_bottom),
        (end[0], end[1], floor_bottom + floor_height),
        (start[0], start[1], floor_bottom + floor_height),
    ]
    return MeshSpec(
        name=f"{source_name}.facade_{side_index}_row_{floor_index}",
        role="facade",
        material=_wall_material_name(style.wall_material, "row", color_index),
        color=color,
        texture_path=style.wall_texture_path,
        vertices=vertices,
        faces=[[0, 1, 2, 3]],
        texture_scale=style.wall_texture_scale,
    )


def _window_mesh(
    source_name: str,
    side_index: int,
    floor_index: int,
    window_index: int,
    start: Point3D,
    end: Point3D,
    normal: Point2D,
    offset: float,
    floor_bottom: float,
    floor_height: float,
    style: FacadeStyleConfig,
) -> MeshSpec:
    tangent = _tangent(start, end)
    window = style.window
    center = _point_on_segment(start, tangent, normal, offset, window.depth)
    height = min(window.height, max(0.2, floor_height - window.sill_height - 0.25))
    bottom = floor_bottom + min(window.sill_height, max(0.15, floor_height - height - 0.15))
    left = _move(center, tangent, -window.width / 2.0)
    right = _move(center, tangent, window.width / 2.0)
    vertices = [
        (left[0], left[1], bottom),
        (right[0], right[1], bottom),
        (right[0], right[1], bottom + height),
        (left[0], left[1], bottom + height),
    ]
    return MeshSpec(
        name=f"{source_name}.window_{side_index}_{floor_index}_{window_index}",
        role="window",
        material=window.material,
        color=window.color,
        texture_path=window.texture_path,
        vertices=vertices,
        faces=[[0, 1, 2, 3]],
        texture_scale=window.texture_scale,
    )


def _window_detail_meshes(
    source_name: str,
    side_index: int,
    floor_index: int,
    window_index: int,
    start: Point3D,
    end: Point3D,
    normal: Point2D,
    offset: float,
    floor_bottom: float,
    floor_height: float,
    style: FacadeStyleConfig,
) -> list[MeshSpec]:
    tangent = _tangent(start, end)
    window = style.window
    frame_width = max(window.frame_width, 0.0)
    height = min(window.height, max(0.2, floor_height - window.sill_height - 0.25))
    bottom = floor_bottom + min(window.sill_height, max(0.15, floor_height - height - 0.15))
    depth = window.depth + max(window.frame_depth, 0.0)
    meshes: list[MeshSpec] = []

    if frame_width > 0:
        frame_height = height + frame_width * 2.0
        frame_bottom = bottom - frame_width
        frame_span = window.width + frame_width * 2.0
        for name, lateral in (("left", -window.width / 2.0 - frame_width / 2.0), ("right", window.width / 2.0 + frame_width / 2.0)):
            meshes.append(
                _window_panel_mesh(
                    f"{source_name}.window_frame_{side_index}_{floor_index}_{window_index}_{name}",
                    "window_frame",
                    start,
                    tangent,
                    normal,
                    offset + lateral,
                    frame_width,
                    frame_bottom,
                    frame_height,
                    depth,
                    window.frame_material,
                    window.frame_color,
                )
            )
        for name, z in (("bottom", bottom - frame_width / 2.0), ("top", bottom + height + frame_width / 2.0)):
            meshes.append(
                _window_panel_mesh(
                    f"{source_name}.window_frame_{side_index}_{floor_index}_{window_index}_{name}",
                    "window_frame",
                    start,
                    tangent,
                    normal,
                    offset,
                    frame_span,
                    z - frame_width / 2.0,
                    frame_width,
                    depth,
                    window.frame_material,
                    window.frame_color,
                )
            )

    mullion_width = max(frame_width * 0.65, 0.025)
    for mullion_index in range(max(window.vertical_mullions, 0)):
        lateral = -window.width / 2.0 + window.width * (mullion_index + 1) / (window.vertical_mullions + 1)
        meshes.append(
            _window_panel_mesh(
                f"{source_name}.window_mullion_v_{side_index}_{floor_index}_{window_index}_{mullion_index}",
                "window_mullion",
                start,
                tangent,
                normal,
                offset + lateral,
                mullion_width,
                bottom,
                height,
                depth + 0.01,
                window.frame_material,
                window.frame_color,
            )
        )

    for mullion_index in range(max(window.horizontal_mullions, 0)):
        z = bottom + height * (mullion_index + 1) / (window.horizontal_mullions + 1)
        meshes.append(
            _window_panel_mesh(
                f"{source_name}.window_mullion_h_{side_index}_{floor_index}_{window_index}_{mullion_index}",
                "window_mullion",
                start,
                tangent,
                normal,
                offset,
                window.width,
                z - mullion_width / 2.0,
                mullion_width,
                depth + 0.01,
                window.frame_material,
                window.frame_color,
            )
        )

    if window.sill_depth > 0 and window.sill_thickness > 0:
        center = _point_on_segment(start, tangent, normal, offset, window.sill_depth / 2.0)
        vertices, faces = _oriented_box(
            center,
            tangent,
            normal,
            window.width + frame_width * 2.5,
            window.sill_depth,
            window.sill_thickness,
            max(0.0, bottom - window.sill_thickness),
        )
        meshes.append(
            MeshSpec(
                name=f"{source_name}.window_sill_{side_index}_{floor_index}_{window_index}",
                role="window_sill",
                material=window.sill_material,
                color=window.sill_color,
                texture_path=None,
                vertices=vertices,
                faces=faces,
            )
        )

    if _style_has_shutters(style):
        shutter_width = max(min(window.width * 0.28, 0.34), 0.16)
        shutter_height = height + frame_width
        shutter_bottom = max(0.0, bottom - frame_width * 0.5)
        for name, lateral in (
            ("left", -window.width / 2.0 - shutter_width / 2.0 - frame_width),
            ("right", window.width / 2.0 + shutter_width / 2.0 + frame_width),
        ):
            meshes.append(
                _window_panel_mesh(
                    f"{source_name}.shutter_{side_index}_{floor_index}_{window_index}_{name}",
                    "shutter",
                    start,
                    tangent,
                    normal,
                    offset + lateral,
                    shutter_width,
                    shutter_bottom,
                    shutter_height,
                    depth + 0.025,
                    "Grammar Facade Shutters",
                    (0.14, 0.19, 0.14, 1.0),
                )
            )

    return meshes


def _facade_depth_meshes(
    source_name: str,
    side_index: int,
    start: Point3D,
    end: Point3D,
    normal: Point2D,
    street_side_index: int,
    floor_heights: list[float],
    total_height: float,
    style: FacadeStyleConfig,
    tags: dict[str, str],
) -> list[MeshSpec]:
    length = distance_2d(start, end)
    if length <= 0.1:
        return []
    tangent = _tangent(start, end)
    ground_height = floor_heights[0] if floor_heights else total_height
    meshes: list[MeshSpec] = []
    street_facing = side_index == street_side_index

    if street_facing and _is_retail_style(style, tags):
        sign_height = min(0.62, max(0.32, ground_height * 0.16))
        sign_bottom = min(max(ground_height - sign_height - 0.35, 2.4), max(0.2, ground_height - sign_height))
        meshes.append(
            _window_panel_mesh(
                f"{source_name}.signboard_{side_index}",
                "signboard",
                start,
                tangent,
                normal,
                length / 2.0,
                length * 0.82,
                sign_bottom,
                sign_height,
                0.08,
                "Grammar Retail Signboards",
                (0.88, 0.72, 0.24, 1.0),
            )
        )
        awning_width = length * 0.7
        awning_depth = 0.85
        awning_center = _point_on_segment(start, tangent, normal, length / 2.0, awning_depth / 2.0)
        vertices, faces = _oriented_box(
            awning_center,
            tangent,
            normal,
            awning_width,
            awning_depth,
            0.12,
            max(2.1, sign_bottom - 0.22),
        )
        meshes.append(MeshSpec(f"{source_name}.awning_{side_index}", "awning", "Grammar Fabric Awnings", (0.56, 0.08, 0.07, 1.0), None, vertices, faces))

    if street_facing and (_is_industrial_style(style, tags) or _is_parking_style(style, tags)):
        door_width = min(length * 0.55, 4.8 if _is_industrial_style(style, tags) else 5.6)
        door_height = min(max(ground_height - 0.25, 2.2), 4.2)
        meshes.append(
            _window_panel_mesh(
                f"{source_name}.garage_door_{side_index}",
                "garage_door",
                start,
                tangent,
                normal,
                length / 2.0,
                door_width,
                0.0,
                door_height,
                0.09,
                "Grammar Sectional Garage Doors",
                (0.24, 0.25, 0.24, 1.0),
            )
        )
        if _is_industrial_style(style, tags):
            dock_center = _point_on_segment(start, tangent, normal, length / 2.0, 0.38)
            vertices, faces = _oriented_box(dock_center, tangent, normal, door_width + 1.0, 0.76, 0.45, 0.0)
            meshes.append(MeshSpec(f"{source_name}.loading_dock_{side_index}", "loading_dock", "Grammar Concrete Loading Docks", (0.42, 0.42, 0.39, 1.0), None, vertices, faces))

    if _should_add_stair_core(style, tags, street_facing, length, total_height):
        core_width = min(max(length * 0.18, 1.1), 2.4)
        core_height = max(total_height - 0.4, 1.0)
        meshes.append(
            _window_panel_mesh(
                f"{source_name}.stair_core_{side_index}",
                "stair_core",
                start,
                tangent,
                normal,
                length * 0.5,
                core_width,
                0.2,
                core_height,
                0.06,
                "Grammar Stair Core Glass",
                (0.09, 0.18, 0.22, 0.82),
            )
        )

    meshes.extend(_facade_pattern_meshes(source_name, side_index, start, end, tangent, normal, floor_heights, total_height, style, tags))
    return meshes


def _facade_pattern_meshes(
    source_name: str,
    side_index: int,
    start: Point3D,
    end: Point3D,
    tangent: Point2D,
    normal: Point2D,
    floor_heights: list[float],
    total_height: float,
    style: FacadeStyleConfig,
    tags: dict[str, str],
) -> list[MeshSpec]:
    length = distance_2d(start, end)
    meshes: list[MeshSpec] = []
    tokens = _style_tokens(style, tags)

    if _has_any(tokens, {"plattenbau", "prefab", "industrial", "warehouse", "parking", "office", "curtain"}):
        vertical_count = min(max(int(length // 3.2), 1), 8)
        for index in range(1, vertical_count + 1):
            offset = length * index / (vertical_count + 1)
            meshes.append(
                _window_panel_mesh(
                    f"{source_name}.panel_seam_v_{side_index}_{index}",
                    "panel_seam",
                    start,
                    tangent,
                    normal,
                    offset,
                    0.045,
                    0.0,
                    total_height,
                    0.045,
                    "Grammar Facade Panel Seams",
                    (0.2, 0.205, 0.2, 1.0),
                )
            )
        for floor_index, bottom in enumerate(_floor_bottoms(floor_heights)[1:], start=1):
            meshes.append(
                _window_panel_mesh(
                    f"{source_name}.panel_seam_h_{side_index}_{floor_index}",
                    "panel_seam",
                    start,
                    tangent,
                    normal,
                    length / 2.0,
                    length,
                    bottom - 0.025,
                    0.05,
                    0.045,
                    "Grammar Facade Panel Seams",
                    (0.2, 0.205, 0.2, 1.0),
                )
            )

    if _has_any(tokens, {"passivhaus", "contemporary", "modern", "bauhaus"}):
        for floor_index, bottom in enumerate(_floor_bottoms(floor_heights)[1:], start=1):
            meshes.append(
                _window_panel_mesh(
                    f"{source_name}.insulation_band_{side_index}_{floor_index}",
                    "insulation_band",
                    start,
                    tangent,
                    normal,
                    length / 2.0,
                    length * 0.94,
                    bottom - 0.06,
                    0.12,
                    0.065,
                    "Grammar Insulation Shadow Bands",
                    (0.78, 0.78, 0.72, 1.0),
                )
            )

    if _has_any(tokens, {"gruenderzeit", "jugendstil", "fachwerk", "historic", "altbau", "kontorhaus", "church", "cathedral", "sacral"}):
        for floor_index, bottom in enumerate(_floor_bottoms(floor_heights)[1:], start=1):
            meshes.append(
                _window_panel_mesh(
                    f"{source_name}.facade_ornament_band_{side_index}_{floor_index}",
                    "facade_ornament",
                    start,
                    tangent,
                    normal,
                    length / 2.0,
                    length * 0.92,
                    bottom - 0.08,
                    0.16,
                    0.09,
                    "Grammar Facade Ornament Bands",
                    (0.78, 0.72, 0.62, 1.0),
                )
            )
        pilaster_count = min(max(int(length // 4.0), 1), 5)
        if _has_any(tokens, {"church", "cathedral", "sacral"}):
            pilaster_count = min(max(int(length // 3.2), 2), 8)
        for index in range(1, pilaster_count + 1):
            offset = length * index / (pilaster_count + 1)
            meshes.append(
                _window_panel_mesh(
                    f"{source_name}.facade_ornament_pilaster_{side_index}_{index}",
                    "facade_ornament",
                    start,
                    tangent,
                    normal,
                    offset,
                    0.16,
                    0.0,
                    total_height,
                    0.08,
                    "Grammar Facade Ornament Pilasters",
                    (0.72, 0.66, 0.56, 1.0),
                )
            )
    return meshes


def _window_panel_mesh(
    name: str,
    role: str,
    start: Point3D,
    tangent: Point2D,
    normal: Point2D,
    center_offset: float,
    width: float,
    bottom: float,
    height: float,
    depth: float,
    material: str,
    color: tuple[float, float, float, float],
) -> MeshSpec:
    center = _point_on_segment(start, tangent, normal, center_offset, depth)
    left = _move(center, tangent, -width / 2.0)
    right = _move(center, tangent, width / 2.0)
    vertices = [
        (left[0], left[1], bottom),
        (right[0], right[1], bottom),
        (right[0], right[1], bottom + height),
        (left[0], left[1], bottom + height),
    ]
    return MeshSpec(name, role, material, color, None, vertices, [[0, 1, 2, 3]])


def _door_mesh(
    source_name: str,
    side_index: int,
    start: Point3D,
    end: Point3D,
    normal: Point2D,
    offset: float,
    ground_floor_height: float,
    style: FacadeStyleConfig,
) -> MeshSpec:
    tangent = _tangent(start, end)
    door = style.door
    height = min(door.height, max(1.6, ground_floor_height - 0.25))
    center = _point_on_segment(start, tangent, normal, offset, door.depth)
    left = _move(center, tangent, -door.width / 2.0)
    right = _move(center, tangent, door.width / 2.0)
    vertices = [
        (left[0], left[1], 0.0),
        (right[0], right[1], 0.0),
        (right[0], right[1], height),
        (left[0], left[1], height),
    ]
    return MeshSpec(
        name=f"{source_name}.door_{side_index}",
        role="door",
        material=door.material,
        color=door.color,
        texture_path=door.texture_path,
        vertices=vertices,
        faces=[[0, 1, 2, 3]],
        texture_scale=door.texture_scale,
    )


def _door_detail_meshes(
    source_name: str,
    side_index: int,
    start: Point3D,
    end: Point3D,
    normal: Point2D,
    offset: float,
    ground_floor_height: float,
    style: FacadeStyleConfig,
) -> list[MeshSpec]:
    tangent = _tangent(start, end)
    door = style.door
    height = min(door.height, max(1.6, ground_floor_height - 0.25))
    depth = door.depth + max(door.frame_depth, 0.0)
    frame_width = max(door.frame_width, 0.0)
    meshes: list[MeshSpec] = []

    if frame_width > 0:
        for name, lateral in (("left", -door.width / 2.0 - frame_width / 2.0), ("right", door.width / 2.0 + frame_width / 2.0)):
            meshes.append(
                _window_panel_mesh(
                    f"{source_name}.door_frame_{side_index}_{name}",
                    "door_frame",
                    start,
                    tangent,
                    normal,
                    offset + lateral,
                    frame_width,
                    0.0,
                    height + frame_width,
                    depth,
                    door.frame_material,
                    door.frame_color,
                )
            )
        meshes.append(
            _window_panel_mesh(
                f"{source_name}.door_frame_{side_index}_top",
                "door_frame",
                start,
                tangent,
                normal,
                offset,
                door.width + frame_width * 2.0,
                height,
                frame_width,
                depth,
                door.frame_material,
                door.frame_color,
            )
        )

    if door.handle_enabled and door.handle_radius > 0:
        handle_center = _point_on_segment(start, tangent, normal, offset + door.width * 0.28, depth + door.handle_radius)
        vertices, faces = _oriented_box(
            handle_center,
            tangent,
            normal,
            door.handle_radius * 1.6,
            door.handle_radius * 1.6,
            door.handle_radius * 1.6,
            max(0.7, min(height * 0.55, height - 0.25)),
        )
        meshes.append(
            MeshSpec(
                name=f"{source_name}.door_handle_{side_index}",
                role="door_handle",
                material=door.handle_material,
                color=door.handle_color,
                texture_path=None,
                vertices=vertices,
                faces=faces,
            )
        )

    if door.canopy_enabled and door.canopy_width > 0 and door.canopy_depth > 0 and door.canopy_thickness > 0:
        canopy_center = _point_on_segment(start, tangent, normal, offset, door.canopy_depth / 2.0)
        vertices, faces = _oriented_box(
            canopy_center,
            tangent,
            normal,
            door.canopy_width,
            door.canopy_depth,
            door.canopy_thickness,
            min(ground_floor_height - door.canopy_thickness, height + 0.18),
        )
        meshes.append(
            MeshSpec(
                name=f"{source_name}.door_canopy_{side_index}",
                role="door_canopy",
                material=door.canopy_material,
                color=door.canopy_color,
                texture_path=None,
                vertices=vertices,
                faces=faces,
            )
        )

    return meshes


def _ledge_mesh(
    source_name: str,
    side_index: int,
    floor_index: int,
    start: Point3D,
    end: Point3D,
    normal: Point2D,
    floor_bottom: float,
    style: FacadeStyleConfig,
) -> MeshSpec:
    ledge = style.ledge
    z = max(0.05, floor_bottom + ledge.height)
    outer_start = _offset_point(start, normal, ledge.depth)
    outer_end = _offset_point(end, normal, ledge.depth)
    vertices = [
        (start[0], start[1], z),
        (end[0], end[1], z),
        (outer_end[0], outer_end[1], z),
        (outer_start[0], outer_start[1], z),
    ]
    return MeshSpec(
        name=f"{source_name}.ledge_{side_index}_{floor_index}",
        role="ledge",
        material=ledge.material,
        color=ledge.color,
        texture_path=ledge.texture_path,
        vertices=vertices,
        faces=[[0, 1, 2, 3]],
        texture_scale=ledge.texture_scale,
    )


def _balcony_mesh(
    source_name: str,
    side_index: int,
    floor_index: int,
    window_index: int,
    start: Point3D,
    end: Point3D,
    normal: Point2D,
    offset: float,
    floor_bottom: float,
    style: FacadeStyleConfig,
) -> MeshSpec:
    tangent = _tangent(start, end)
    balcony = style.balcony
    center = _point_on_segment(start, tangent, normal, offset, balcony.depth / 2.0)
    bottom = floor_bottom + 0.08
    vertices, faces = _oriented_box(
        center,
        tangent,
        normal,
        balcony.width,
        balcony.depth,
        balcony.slab_height,
        bottom,
    )
    return MeshSpec(
        name=f"{source_name}.balcony_{side_index}_{floor_index}_{window_index}",
        role="balcony",
        material=balcony.material,
        color=balcony.color,
        texture_path=balcony.texture_path,
        vertices=vertices,
        faces=faces,
        texture_scale=balcony.texture_scale,
    )


def _balcony_detail_meshes(
    source_name: str,
    side_index: int,
    floor_index: int,
    window_index: int,
    start: Point3D,
    end: Point3D,
    normal: Point2D,
    offset: float,
    floor_bottom: float,
    style: FacadeStyleConfig,
) -> list[MeshSpec]:
    tangent = _tangent(start, end)
    balcony = style.balcony
    bottom = floor_bottom + 0.08
    rail_bottom = bottom + balcony.slab_height
    rail_center = _point_on_segment(start, tangent, normal, offset, balcony.depth)
    rail_bar_width = max(balcony.railing_bar_width, 0.02)
    rail_bar_depth = max(balcony.railing_bar_depth, 0.02)
    meshes: list[MeshSpec] = []

    front_vertices, front_faces = _oriented_box(
        rail_center,
        tangent,
        normal,
        balcony.width,
        rail_bar_depth,
        balcony.railing_height,
        rail_bottom,
    )
    meshes.append(
        MeshSpec(
            name=f"{source_name}.balcony_rail_front_{side_index}_{floor_index}_{window_index}",
            role="balcony_rail",
            material=balcony.railing_material,
            color=balcony.railing_color,
            texture_path=None,
            vertices=front_vertices,
            faces=front_faces,
        )
    )

    side_depth = max(balcony.depth - rail_bar_depth, 0.05)
    for side_name, lateral in (("left", -balcony.width / 2.0), ("right", balcony.width / 2.0)):
        side_center = _point_on_segment(
            start,
            tangent,
            normal,
            offset + lateral,
            balcony.depth / 2.0,
        )
        vertices, faces = _oriented_box(
            side_center,
            normal,
            tangent,
            side_depth,
            rail_bar_width,
            balcony.railing_height,
            rail_bottom,
        )
        meshes.append(
            MeshSpec(
                name=f"{source_name}.balcony_rail_{side_name}_{side_index}_{floor_index}_{window_index}",
                role="balcony_rail",
                material=balcony.railing_material,
                color=balcony.railing_color,
                texture_path=None,
                vertices=vertices,
                faces=faces,
            )
        )

    bar_count = max(balcony.railing_bar_count, 0)
    for bar_index in range(bar_count):
        lateral = -balcony.width / 2.0 + balcony.width * (bar_index + 1) / (bar_count + 1)
        bar_center = _point_on_segment(start, tangent, normal, offset + lateral, balcony.depth)
        vertices, faces = _oriented_box(
            bar_center,
            tangent,
            normal,
            rail_bar_width,
            rail_bar_depth,
            balcony.railing_height,
            rail_bottom,
        )
        meshes.append(
            MeshSpec(
                name=f"{source_name}.balcony_bar_{side_index}_{floor_index}_{window_index}_{bar_index}",
                role="balcony_bar",
                material=balcony.railing_material,
                color=balcony.railing_color,
                texture_path=None,
                vertices=vertices,
                faces=faces,
            )
        )

    return meshes


def _roof_mesh(source_name: str, footprint: list[Point3D], height: float, roof: RoofStyleConfig, tags: dict[str, str]) -> MeshSpec:
    roof_type = roof.type.lower()
    if roof_type == "gabled":
        return _gabled_roof_mesh(source_name, footprint, height, roof, tags)
    if roof_type == "hipped":
        return _hipped_roof_mesh(source_name, footprint, height, roof, tags)
    if roof_type in {"pyramid", "pyramidal"}:
        return _pyramid_roof_mesh(source_name, footprint, height, roof)
    roof_z = height - max(roof.surface_inset, 0.0) if roof.edge_enabled else height
    vertices = _roof_base_vertices(footprint, roof_z, roof.overhang)
    return MeshSpec(
        name=f"{source_name}.roof",
        role="roof",
        material=roof.material,
        color=roof.color,
        texture_path=roof.texture_path,
        vertices=vertices,
        faces=[list(range(len(vertices)))],
        texture_scale=roof.texture_scale,
    )


def _gabled_roof_mesh(source_name: str, footprint: list[Point3D], height: float, roof: RoofStyleConfig, tags: dict[str, str]) -> MeshSpec:
    base = _roof_base_vertices(footprint, height, roof.overhang)
    frame = _roof_frame(base, _gabled_ridge_direction(base, roof, tags), height, height + roof.height)
    vertices = base + [_ridge_projection(point, frame) for point in base]
    faces = []
    count = len(base)
    for index in range(count):
        next_index = (index + 1) % count
        faces.append([index, next_index, count + next_index, count + index])
    return MeshSpec(f"{source_name}.roof", "roof", roof.material, roof.color, roof.texture_path, vertices, faces, texture_scale=roof.texture_scale)


def _hipped_roof_mesh(source_name: str, footprint: list[Point3D], height: float, roof: RoofStyleConfig, tags: dict[str, str]) -> MeshSpec:
    base = _roof_base_vertices(footprint, height, roof.overhang)
    frame = _roof_frame(base, _roof_orientation_direction(base, tags), height, height + roof.height)
    long_span = frame.max_long - frame.min_long
    side_span = frame.max_side - frame.min_side
    if long_span <= side_span * 1.25:
        return _pyramid_roof_mesh(source_name, footprint, height, roof)
    inset = min(side_span * 0.42, long_span * 0.24)
    ridge_start = _point_from_roof_axes(frame, frame.min_long + inset, 0.0, frame.ridge_z)
    ridge_end = _point_from_roof_axes(frame, frame.max_long - inset, 0.0, frame.ridge_z)
    vertices = base + [ridge_start, ridge_end]
    ridge_start_index = len(base)
    ridge_end_index = len(base) + 1
    faces = []
    for index, point in enumerate(base):
        next_index = (index + 1) % len(base)
        next_point = base[next_index]
        station = _roof_long_station(point, frame)
        next_station = _roof_long_station(next_point, frame)
        if station < frame.min_long + inset and next_station < frame.min_long + inset:
            faces.append([index, next_index, ridge_start_index])
        elif station > frame.max_long - inset and next_station > frame.max_long - inset:
            faces.append([index, next_index, ridge_end_index])
        else:
            faces.append([index, next_index, ridge_end_index, ridge_start_index])
    return MeshSpec(f"{source_name}.roof", "roof", roof.material, roof.color, roof.texture_path, vertices, faces, texture_scale=roof.texture_scale)


def _pyramid_roof_mesh(source_name: str, footprint: list[Point3D], height: float, roof: RoofStyleConfig) -> MeshSpec:
    base = _roof_base_vertices(footprint, height, roof.overhang)
    center_x = sum(point[0] for point in base) / len(base)
    center_y = sum(point[1] for point in base) / len(base)
    vertices = base + [(center_x, center_y, height + roof.height)]
    peak = len(vertices) - 1
    faces = [[index, (index + 1) % len(base), peak] for index in range(len(base))]
    return MeshSpec(f"{source_name}.roof", "roof", roof.material, roof.color, roof.texture_path, vertices, faces, texture_scale=roof.texture_scale)


def _roof_style_for_building(source_name: str, styles: list[FacadeStyleConfig]) -> FacadeStyleConfig | None:
    roof_styles = [style for style in styles if style.roof is not None]
    if not roof_styles:
        return None
    return roof_styles[_stable_index(source_name, len(roof_styles))]


def _gabled_ridge_direction(base: list[Point3D], roof: RoofStyleConfig, tags: dict[str, str]) -> Point2D:
    oriented = _roof_orientation_direction(base, tags)
    if oriented != (0.0, 0.0):
        return oriented
    alignment = str(getattr(roof, "ridge_alignment", "closest_street")).strip().lower()
    if alignment in {"closest_street", "nearest_street", "street", "highway"}:
        hinted = _ridge_direction_from_tags(tags)
        if hinted != (0.0, 0.0):
            return hinted
    return _longest_axis_direction(base)


def _roof_orientation_direction(base: list[Point3D], tags: dict[str, str]) -> Point2D:
    orientation = (tags.get("roof:orientation") or tags.get("roof:direction") or "").strip().lower()
    if not orientation:
        return (0.0, 0.0)
    longest = _longest_axis_direction(base)
    if orientation in {"along", "parallel", "longitudinal"}:
        return longest
    if orientation in {"across", "perpendicular", "transverse"}:
        return (-longest[1], longest[0])
    cardinal = _direction_from_cardinal_or_angle(orientation)
    if cardinal != (0.0, 0.0):
        return cardinal
    return (0.0, 0.0)


def _direction_from_cardinal_or_angle(value: str) -> Point2D:
    normalized = value.replace(" ", "").replace("_", "-")
    cardinal = {
        "n": (0.0, 1.0),
        "s": (0.0, 1.0),
        "north": (0.0, 1.0),
        "south": (0.0, 1.0),
        "e": (1.0, 0.0),
        "w": (1.0, 0.0),
        "east": (1.0, 0.0),
        "west": (1.0, 0.0),
        "n-s": (0.0, 1.0),
        "north-south": (0.0, 1.0),
        "s-n": (0.0, 1.0),
        "e-w": (1.0, 0.0),
        "east-west": (1.0, 0.0),
        "w-e": (1.0, 0.0),
        "ne-sw": _normalize_2d((1.0, 1.0)),
        "sw-ne": _normalize_2d((1.0, 1.0)),
        "nw-se": _normalize_2d((-1.0, 1.0)),
        "se-nw": _normalize_2d((-1.0, 1.0)),
    }
    if normalized in cardinal:
        return cardinal[normalized]
    try:
        angle = float(normalized.removesuffix("deg").replace("°", ""))
    except ValueError:
        return (0.0, 0.0)
    from math import cos, radians, sin

    radians_value = radians(angle)
    return _normalize_2d((sin(radians_value), cos(radians_value)))


def _ridge_direction_from_tags(tags: dict[str, str]) -> Point2D:
    for key in ("grammar:roof:ridge_direction", "roof:ridge:direction"):
        value = tags.get(key)
        if not value:
            continue
        parts = value.replace(";", ",").split(",")
        if len(parts) < 2:
            continue
        try:
            direction = _normalize_2d((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
        if direction != (0.0, 0.0):
            return direction
    return (0.0, 0.0)


def _longest_axis_direction(points: list[Point3D]) -> Point2D:
    min_x, min_y, max_x, max_y = _bounds(points)
    if (max_x - min_x) >= (max_y - min_y):
        return (1.0, 0.0)
    return (0.0, 1.0)


def _ridge_line_for_direction(base: list[Point3D], direction: Point2D, z: float) -> list[Point3D]:
    direction = _normalize_2d(direction)
    if direction == (0.0, 0.0):
        direction = _longest_axis_direction(base)
    center = _centroid_2d(base)
    projections = [
        (point[0] - center[0]) * direction[0] + (point[1] - center[1]) * direction[1]
        for point in base
    ]
    start_distance = min(projections)
    end_distance = max(projections)
    if end_distance - start_distance <= 1e-6:
        min_x, min_y, max_x, max_y = _bounds(base)
        span = max(max_x - min_x, max_y - min_y) / 2.0
        start_distance = -span
        end_distance = span
    return [
        (center[0] + direction[0] * start_distance, center[1] + direction[1] * start_distance, z),
        (center[0] + direction[0] * end_distance, center[1] + direction[1] * end_distance, z),
    ]


def _roof_frame(base: list[Point3D], direction: Point2D, eave_z: float, ridge_z: float) -> RoofFrame:
    direction = _normalize_2d(direction)
    if direction == (0.0, 0.0):
        direction = _longest_axis_direction(base)
    normal = (-direction[1], direction[0])
    center = _centroid_2d(base)
    long_values = [_roof_axis_value(point, center, direction) for point in base]
    side_values = [_roof_axis_value(point, center, normal) for point in base]
    return RoofFrame(center, direction, normal, min(long_values), max(long_values), min(side_values), max(side_values), eave_z, ridge_z)


def _ridge_projection(point: Point3D, frame: RoofFrame) -> Point3D:
    station = _roof_long_station(point, frame)
    station = min(frame.max_long, max(frame.min_long, station))
    return _point_from_roof_axes(frame, station, 0.0, frame.ridge_z)


def _roof_long_station(point: Point3D, frame: RoofFrame) -> float:
    return _roof_axis_value(point, frame.center, frame.direction)


def _roof_side_station(point: Point3D, frame: RoofFrame) -> float:
    return _roof_axis_value(point, frame.center, frame.normal)


def _roof_axis_value(point: Point3D, center: Point2D, axis: Point2D) -> float:
    return (point[0] - center[0]) * axis[0] + (point[1] - center[1]) * axis[1]


def _point_from_roof_axes(frame: RoofFrame, long_value: float, side_value: float, z: float) -> Point3D:
    return (
        frame.center[0] + frame.direction[0] * long_value + frame.normal[0] * side_value,
        frame.center[1] + frame.direction[1] * long_value + frame.normal[1] * side_value,
        z,
    )


def _roof_surface_z(long_value: float, side_value: float, frame: RoofFrame) -> float:
    half_span = max(abs(frame.min_side), abs(frame.max_side), 1e-6)
    slope = 1.0 - min(abs(side_value) / half_span, 1.0)
    return frame.eave_z + (frame.ridge_z - frame.eave_z) * slope


def _face_area_2d(points: list[Point3D]) -> float:
    area = 0.0
    for start, end in zip(points, points[1:] + points[:1]):
        area += start[0] * end[1] - end[0] * start[1]
    return abs(area) / 2.0


def _roof_detail_meshes(source_name: str, footprint: list[Point3D], height: float, roof: RoofStyleConfig, tags: dict[str, str]) -> list[MeshSpec]:
    roof_type = roof.type.lower()
    base = _roof_base_vertices(footprint, height, roof.overhang)
    if roof_type == "flat":
        frame = _roof_frame(base, _longest_axis_direction(base), height, height)
    else:
        direction = _gabled_ridge_direction(base, roof, tags) if roof_type == "gabled" else _longest_axis_direction(base)
        frame = _roof_frame(base, direction, height, height + roof.height)
    meshes: list[MeshSpec] = []
    meshes.extend(_roof_tile_meshes(source_name, frame, roof, roof_type))
    meshes.extend(_roof_window_meshes(source_name, frame, roof, roof_type))
    meshes.extend(_dormer_meshes(source_name, frame, roof, roof_type))
    meshes.extend(_chimney_meshes(source_name, frame, roof))
    return meshes


def _roof_tile_meshes(source_name: str, frame: RoofFrame, roof: RoofStyleConfig, roof_type: str) -> list[MeshSpec]:
    rows = max(int(getattr(roof, "tile_rows", 0)), 0)
    if rows <= 0 or roof_type == "flat":
        return []
    long_span = max(frame.max_long - frame.min_long, 0.0)
    side_span = max(frame.max_side - frame.min_side, 0.0)
    if long_span <= 0.0 or side_span <= 0.0:
        return []
    width = long_span * 0.92
    row_depth = max(getattr(roof, "tile_spacing", 0.55) * 0.12, getattr(roof, "tile_depth", 0.035))
    meshes = []
    for side_sign in (-1.0, 1.0):
        side_limit = frame.max_side if side_sign > 0 else frame.min_side
        for row_index in range(rows):
            factor = (row_index + 1) / (rows + 1)
            side_value = side_limit * (1.0 - factor * 0.88)
            z = _roof_surface_z(0.0, side_value, frame) + max(getattr(roof, "tile_depth", 0.035), 0.0)
            center = _point_from_roof_axes(frame, 0.0, side_value, z)
            vertices, faces = _oriented_box(
                (center[0], center[1]),
                frame.direction,
                frame.normal,
                width,
                row_depth,
                max(getattr(roof, "tile_depth", 0.035), 0.01),
                z,
            )
            meshes.append(MeshSpec(
                f"{source_name}.roof_tile_{len(meshes)}",
                "roof_tile",
                getattr(roof, "tile_material", "Grammar Roof Tile Bands"),
                getattr(roof, "tile_color", (0.28, 0.07, 0.045, 1.0)),
                None,
                vertices,
                faces,
            ))
    return meshes


def _roof_window_meshes(source_name: str, frame: RoofFrame, roof: RoofStyleConfig, roof_type: str) -> list[MeshSpec]:
    count = max(int(getattr(roof, "roof_window_count", 0)), 0)
    if count <= 0 or roof_type == "flat":
        return []
    meshes = []
    long_positions = _detail_positions(count, frame.min_long, frame.max_long, inset=0.22)
    for index, long_value in enumerate(long_positions):
        side_sign = -1.0 if index % 2 == 0 else 1.0
        side_limit = frame.min_side if side_sign < 0 else frame.max_side
        side_value = side_limit * 0.48
        z = _roof_surface_z(long_value, side_value, frame) + 0.045
        center = _point_from_roof_axes(frame, long_value, side_value, z)
        vertices, faces = _oriented_box(
            (center[0], center[1]),
            frame.direction,
            frame.normal,
            getattr(roof, "roof_window_width", 0.75),
            getattr(roof, "roof_window_height", 1.05),
            0.035,
            z,
        )
        meshes.append(MeshSpec(
            f"{source_name}.roof_window_{index}",
            "roof_window",
            getattr(roof, "roof_window_material", "Grammar Roof Window Glass"),
            getattr(roof, "roof_window_color", (0.08, 0.16, 0.2, 0.86)),
            None,
            vertices,
            faces,
        ))
    return meshes


def _dormer_meshes(source_name: str, frame: RoofFrame, roof: RoofStyleConfig, roof_type: str) -> list[MeshSpec]:
    count = max(int(getattr(roof, "dormer_count", 0)), 0)
    if count <= 0 or roof_type not in {"gabled", "hipped"}:
        return []
    meshes = []
    long_positions = _detail_positions(count, frame.min_long, frame.max_long, inset=0.25)
    for index, long_value in enumerate(long_positions):
        side_sign = -1.0 if index % 2 == 0 else 1.0
        side_limit = frame.min_side if side_sign < 0 else frame.max_side
        side_value = side_limit * 0.55
        dormer_width = getattr(roof, "dormer_width", 1.35)
        dormer_depth = getattr(roof, "dormer_depth", 0.9)
        dormer_height = getattr(roof, "dormer_height", 0.9)
        z = _roof_surface_z(long_value, side_value, frame)
        center = _point_from_roof_axes(frame, long_value, side_value, z + dormer_height / 2.0)
        outward = (frame.normal[0] * side_sign, frame.normal[1] * side_sign)
        vertices, faces = _oriented_box(
            (center[0], center[1]),
            frame.direction,
            outward,
            dormer_width,
            dormer_depth,
            dormer_height,
            z,
        )
        meshes.append(MeshSpec(
            f"{source_name}.dormer_{index}",
            "dormer",
            getattr(roof, "dormer_material", "Grammar Dormer Cladding"),
            getattr(roof, "dormer_color", (0.62, 0.58, 0.5, 1.0)),
            None,
            vertices,
            faces,
        ))
        window_center = _point_from_roof_axes(frame, long_value, side_value + side_sign * dormer_depth * 0.52, z + dormer_height * 0.48)
        window_vertices, window_faces = _oriented_box(
            (window_center[0], window_center[1]),
            frame.direction,
            outward,
            dormer_width * 0.45,
            0.035,
            dormer_height * 0.38,
            z + dormer_height * 0.3,
        )
        meshes.append(MeshSpec(
            f"{source_name}.dormer_window_{index}",
            "roof_window",
            getattr(roof, "roof_window_material", "Grammar Roof Window Glass"),
            getattr(roof, "roof_window_color", (0.08, 0.16, 0.2, 0.86)),
            None,
            window_vertices,
            window_faces,
        ))
    return meshes


def _chimney_meshes(source_name: str, frame: RoofFrame, roof: RoofStyleConfig) -> list[MeshSpec]:
    count = max(int(getattr(roof, "chimney_count", 0)), 0)
    if count <= 0:
        return []
    meshes = []
    long_positions = _detail_positions(count, frame.min_long, frame.max_long, inset=0.28)
    for index, long_value in enumerate(long_positions):
        side_value = (frame.max_side if index % 2 else frame.min_side) * 0.18
        chimney_width = getattr(roof, "chimney_width", 0.45)
        chimney_depth = getattr(roof, "chimney_depth", 0.38)
        chimney_height = getattr(roof, "chimney_height", 1.15)
        z = _roof_surface_z(long_value, side_value, frame)
        center = _point_from_roof_axes(frame, long_value, side_value, z + chimney_height / 2.0)
        vertices, faces = _oriented_box(
            (center[0], center[1]),
            frame.direction,
            frame.normal,
            chimney_width,
            chimney_depth,
            chimney_height,
            z,
        )
        meshes.append(MeshSpec(
            f"{source_name}.chimney_{index}",
            "chimney",
            getattr(roof, "chimney_material", "Grammar Brick Chimney"),
            getattr(roof, "chimney_color", (0.42, 0.16, 0.1, 1.0)),
            None,
            vertices,
            faces,
        ))
    return meshes


def _detail_positions(count: int, minimum: float, maximum: float, *, inset: float) -> list[float]:
    if count <= 0:
        return []
    span = maximum - minimum
    if span <= 0.0:
        return [(minimum + maximum) / 2.0]
    start = minimum + span * inset
    end = maximum - span * inset
    if count == 1:
        return [(start + end) / 2.0]
    return [start + (end - start) * index / (count - 1) for index in range(count)]


def _gutter_meshes(source_name: str, footprint: list[Point3D], height: float, roof: RoofStyleConfig) -> list[MeshSpec]:
    if not footprint:
        return []
    roof_type = roof.type.lower()
    z = height + 0.03 if roof_type != "flat" else height + max(roof.edge_height - roof.surface_inset, 0.0) * 0.55
    base = _roof_base_vertices(footprint, z, roof.overhang)
    meshes: list[MeshSpec] = []
    for side_index, (start, end) in enumerate(_segments(base)):
        length = distance_2d(start, end)
        if length <= 0.2:
            continue
        tangent = _tangent(start, end)
        normal = outward_normal(start, end, polygon_is_ccw(base))
        center = ((start[0] + end[0]) / 2.0 + normal[0] * 0.08, (start[1] + end[1]) / 2.0 + normal[1] * 0.08)
        vertices, faces = _oriented_box(center, tangent, normal, length, 0.12, 0.08, z - 0.04)
        meshes.append(MeshSpec(f"{source_name}.gutter_{side_index}", "gutter", "Grammar Roof Gutters", (0.18, 0.18, 0.17, 1.0), None, vertices, faces))
    return meshes


def _roof_service_meshes(
    source_name: str,
    footprint: list[Point3D],
    height: float,
    roof: RoofStyleConfig,
    style: FacadeStyleConfig,
    tags: dict[str, str],
) -> list[MeshSpec]:
    if roof.type.lower() != "flat":
        return []
    tokens = _style_tokens(style, tags)
    if not _has_any(tokens, {"office", "industrial", "warehouse", "retail", "supermarket", "modern", "passivhaus", "parking"}):
        return []
    base = _roof_base_vertices(footprint, height, max(roof.overhang, 0.0))
    min_x, min_y, max_x, max_y = _bounds(base)
    width = max_x - min_x
    depth = max_y - min_y
    if width <= 1.0 or depth <= 1.0:
        return []
    roof_z = height if not roof.edge_enabled else height + max(roof.edge_height - roof.surface_inset, 0.0)
    axis = (1.0, 0.0) if width >= depth else (0.0, 1.0)
    normal = (0.0, 1.0) if axis == (1.0, 0.0) else (1.0, 0.0)
    center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    meshes: list[MeshSpec] = []

    if _has_any(tokens, {"office", "industrial", "warehouse", "retail", "supermarket", "modern", "passivhaus"}):
        panel_count = 4 if max(width, depth) > 12.0 else 2
        for index in range(panel_count):
            lateral = (index - (panel_count - 1) / 2.0) * 1.45
            panel_center = (center[0] + normal[0] * lateral, center[1] + normal[1] * lateral)
            vertices, faces = _oriented_box(panel_center, axis, normal, min(max(width, depth) * 0.32, 3.8), 0.82, 0.08, roof_z + 0.05)
            meshes.append(MeshSpec(f"{source_name}.pv_panel_{index}", "pv_panel", "Grammar Roof PV Panels", (0.04, 0.07, 0.09, 1.0), None, vertices, faces))

    if _has_any(tokens, {"office", "industrial", "warehouse", "retail", "supermarket"}):
        hvac_count = 3 if max(width, depth) > 16.0 else 1
        for index in range(hvac_count):
            shift = (index - (hvac_count - 1) / 2.0) * 1.7
            unit_center = (center[0] + axis[0] * shift - normal[0] * depth * 0.16, center[1] + axis[1] * shift - normal[1] * width * 0.16)
            vertices, faces = _oriented_box(unit_center, axis, normal, 1.1, 0.82, 0.55, roof_z + 0.05)
            meshes.append(MeshSpec(f"{source_name}.hvac_unit_{index}", "hvac_unit", "Grammar Roof HVAC Units", (0.52, 0.54, 0.52, 1.0), None, vertices, faces))

    if _has_any(tokens, {"office", "industrial", "warehouse", "supermarket", "parking"}):
        plant_width = min(max(width, depth) * 0.42, 5.5)
        plant_center = (center[0] + normal[0] * depth * 0.22, center[1] + normal[1] * width * 0.22)
        vertices, faces = _oriented_box(plant_center, axis, normal, plant_width, 0.18, 1.05, roof_z + 0.05)
        meshes.append(MeshSpec(f"{source_name}.roof_plant_screen", "roof_plant", "Grammar Roof Plant Screens", (0.24, 0.25, 0.24, 1.0), None, vertices, faces))
    return meshes


def _roof_edge_meshes(source_name: str, footprint: list[Point3D], height: float, roof: RoofStyleConfig) -> list[MeshSpec]:
    if roof.type.lower() != "flat" or not roof.edge_enabled or roof.edge_width <= 0 or roof.edge_height <= 0:
        return []

    base = _roof_base_vertices(footprint, height, roof.overhang)
    ccw = polygon_is_ccw(footprint)
    bottom = height - max(roof.surface_inset, 0.0)
    meshes: list[MeshSpec] = []
    for side_index, (start, end) in enumerate(_segments(base)):
        normal = outward_normal(start, end, ccw)
        tangent = _tangent(start, end)
        length = distance_2d(start, end)
        if length == 0:
            continue
        center = ((start[0] + end[0]) / 2.0 + normal[0] * roof.edge_width / 2.0, (start[1] + end[1]) / 2.0 + normal[1] * roof.edge_width / 2.0)
        vertices, faces = _oriented_box(center, tangent, normal, length, roof.edge_width, roof.edge_height, bottom)
        meshes.append(
            MeshSpec(
                name=f"{source_name}.roof_edge_{side_index}",
                role="roof_edge",
                material=roof.edge_material,
                color=roof.edge_color,
                texture_path=None,
                vertices=vertices,
                faces=faces,
            )
        )

    cap_size = max(roof.corner_cap_size, roof.edge_width)
    for corner_index, point in enumerate(base):
        prev_point = base[corner_index - 1]
        next_point = base[(corner_index + 1) % len(base)]
        prev_normal = outward_normal(prev_point, point, ccw)
        next_normal = outward_normal(point, next_point, ccw)
        normal = _normalize_2d((prev_normal[0] + next_normal[0], prev_normal[1] + next_normal[1]))
        if normal == (0.0, 0.0):
            normal = next_normal
        tangent = _normalize_2d((next_point[0] - prev_point[0], next_point[1] - prev_point[1]))
        center = (point[0] + normal[0] * roof.edge_width / 2.0, point[1] + normal[1] * roof.edge_width / 2.0)
        vertices, faces = _oriented_box(center, tangent, normal, cap_size, cap_size, roof.edge_height, bottom)
        meshes.append(
            MeshSpec(
                name=f"{source_name}.roof_corner_cap_{corner_index}",
                role="roof_edge",
                material=roof.edge_material,
                color=roof.edge_color,
                texture_path=None,
                vertices=vertices,
                faces=faces,
            )
        )
    return meshes


def _antenna_meshes(
    source_name: str,
    footprint: list[Point3D],
    height: float,
    roof: RoofStyleConfig,
    style: FacadeStyleConfig,
) -> list[MeshSpec]:
    antenna = style.antenna
    if not antenna.enabled or antenna.count <= 0 or antenna.mast_height <= 0:
        return []

    min_x, min_y, max_x, max_y = _bounds(footprint)
    center = _centroid_2d(footprint)
    roof_z = height
    if roof.type.lower() == "flat" and roof.edge_enabled:
        roof_z = height + max(roof.edge_height - roof.surface_inset, 0.0)
    positions = _antenna_positions(center, (min_x, min_y, max_x, max_y), antenna.count)
    meshes: list[MeshSpec] = []
    for index, position in enumerate(positions):
        meshes.extend(_antenna_instance_meshes(source_name, index, position, roof_z, antenna))
    return meshes


def _antenna_instance_meshes(source_name: str, index: int, position: Point2D, roof_z: float, antenna) -> list[MeshSpec]:  # type: ignore[no-untyped-def]
    tangent = (1.0, 0.0)
    normal = (0.0, 1.0)
    mast_width = max(antenna.mast_radius * 2.0, 0.025)
    meshes: list[MeshSpec] = []

    base_vertices, base_faces = _oriented_box(position, tangent, normal, antenna.base_width, antenna.base_depth, antenna.base_height, roof_z)
    meshes.append(MeshSpec(f"{source_name}.antenna_base_{index}", "antenna", antenna.material, antenna.color, None, base_vertices, base_faces))

    mast_vertices, mast_faces = _oriented_box(
        position,
        tangent,
        normal,
        mast_width,
        mast_width,
        antenna.mast_height,
        roof_z + antenna.base_height,
    )
    meshes.append(MeshSpec(f"{source_name}.antenna_mast_{index}", "antenna", antenna.material, antenna.color, None, mast_vertices, mast_faces))

    top_z = roof_z + antenna.base_height + antenna.mast_height
    antenna_type = antenna.type.lower()
    if antenna_type in {"cellular", "communication", "office_cluster"}:
        panel_count = 4 if antenna_type == "office_cluster" else 3
        for panel_index in range(panel_count):
            direction = panel_index % 4
            panel_tangent = (1.0, 0.0) if direction % 2 == 0 else (0.0, 1.0)
            panel_normal = (0.0, 1.0) if direction == 0 else (1.0, 0.0) if direction == 1 else (0.0, -1.0) if direction == 2 else (-1.0, 0.0)
            panel_center = (position[0] + panel_normal[0] * (antenna.panel_depth + 0.08), position[1] + panel_normal[1] * (antenna.panel_depth + 0.08))
            vertices, faces = _oriented_box(
                panel_center,
                panel_tangent,
                panel_normal,
                antenna.panel_width,
                antenna.panel_depth,
                antenna.panel_height,
                top_z - antenna.panel_height * 0.85,
            )
            meshes.append(MeshSpec(f"{source_name}.antenna_panel_{index}_{panel_index}", "antenna_panel", antenna.accent_material, antenna.accent_color, None, vertices, faces))
    elif antenna_type == "satellite":
        dish_center = (position[0] + antenna.panel_depth + 0.08, position[1])
        vertices, faces = _oriented_box(
            dish_center,
            normal,
            tangent,
            antenna.panel_width,
            antenna.panel_depth,
            antenna.panel_height,
            top_z - antenna.panel_height * 0.65,
        )
        meshes.append(MeshSpec(f"{source_name}.antenna_satellite_dish_{index}", "antenna_panel", antenna.accent_material, antenna.accent_color, None, vertices, faces))
        arm_center = (position[0] + antenna.panel_depth * 0.75, position[1])
        arm_vertices, arm_faces = _oriented_box(
            arm_center,
            tangent,
            normal,
            antenna.panel_depth * 2.0,
            mast_width,
            mast_width,
            top_z - antenna.panel_height * 0.2,
        )
        meshes.append(MeshSpec(f"{source_name}.antenna_satellite_arm_{index}", "antenna_panel", antenna.material, antenna.color, None, arm_vertices, arm_faces))
    elif antenna_type == "broadcast":
        for bar_index, z_factor in enumerate((0.35, 0.62, 0.88)):
            vertices, faces = _oriented_box(
                position,
                tangent if bar_index % 2 == 0 else normal,
                normal if bar_index % 2 == 0 else tangent,
                antenna.panel_width * 3.0,
                mast_width,
                mast_width,
                roof_z + antenna.base_height + antenna.mast_height * z_factor,
            )
            meshes.append(MeshSpec(f"{source_name}.antenna_broadcast_bar_{index}_{bar_index}", "antenna_panel", antenna.accent_material, antenna.accent_color, None, vertices, faces))
    elif antenna_type == "lightning_rod":
        spike_vertices, spike_faces = _oriented_box(
            position,
            tangent,
            normal,
            mast_width * 0.6,
            mast_width * 0.6,
            max(antenna.mast_height * 0.28, 0.25),
            top_z,
        )
        meshes.append(MeshSpec(f"{source_name}.antenna_lightning_tip_{index}", "antenna_panel", antenna.accent_material, antenna.accent_color, None, spike_vertices, spike_faces))
    elif antenna_type == "lamp_post":
        lamp_center = (position[0] + antenna.panel_depth * 0.6, position[1])
        vertices, faces = _oriented_box(
            lamp_center,
            tangent,
            normal,
            max(antenna.panel_width, mast_width * 3.0),
            max(antenna.panel_depth, mast_width * 1.8),
            max(antenna.panel_height, mast_width * 2.0),
            top_z - max(antenna.panel_height, mast_width * 2.0) * 0.5,
        )
        meshes.append(MeshSpec(f"{source_name}.roof_lamp_head_{index}", "roof_lamp", antenna.accent_material, antenna.accent_color, None, vertices, faces))
    else:
        bar_width = antenna.panel_width * (2.2 if antenna_type == "radio" else 1.6)
        for bar_index, z_factor in enumerate((0.55, 0.78)):
            vertices, faces = _oriented_box(
                position,
                tangent if bar_index % 2 == 0 else normal,
                normal if bar_index % 2 == 0 else tangent,
                bar_width,
                mast_width,
                mast_width,
                roof_z + antenna.base_height + antenna.mast_height * z_factor,
            )
            meshes.append(MeshSpec(f"{source_name}.antenna_tv_bar_{index}_{bar_index}", "antenna_panel", antenna.accent_material, antenna.accent_color, None, vertices, faces))

    return meshes


def _style_tokens(style: FacadeStyleConfig, tags: dict[str, str]) -> set[str]:
    raw = " ".join(
        [
            style.name,
            style.wall_material,
            style.window.material,
            style.ledge.material,
            tags.get("building", ""),
            tags.get("building:part", ""),
            tags.get("building:use", ""),
            tags.get("shop", ""),
            tags.get("office", ""),
            tags.get("industrial", ""),
            tags.get("amenity", ""),
            tags.get("religion", ""),
            tags.get("denomination", ""),
            tags.get("landuse", ""),
            tags.get("parking", ""),
            tags.get("start_date", ""),
        ]
    ).lower()
    normalized = raw.replace("_", " ").replace("-", " ").replace(":", " ")
    return {token for token in normalized.split() if token}


def _has_any(tokens: set[str], values: set[str]) -> bool:
    return bool(tokens & values)


def _is_retail_style(style: FacadeStyleConfig, tags: dict[str, str]) -> bool:
    tokens = _style_tokens(style, tags)
    return bool(tags.get("shop")) or _has_any(tokens, {"retail", "shopfront", "shop", "supermarket", "commercial"})


def _is_industrial_style(style: FacadeStyleConfig, tags: dict[str, str]) -> bool:
    tokens = _style_tokens(style, tags)
    return bool(tags.get("industrial")) or _has_any(tokens, {"industrial", "warehouse", "factory", "logistics", "manufacturing"})


def _is_parking_style(style: FacadeStyleConfig, tags: dict[str, str]) -> bool:
    tokens = _style_tokens(style, tags)
    return _has_any(tokens, {"parking", "garage", "multistorey", "car"})


def _style_has_shutters(style: FacadeStyleConfig) -> bool:
    tokens = _style_tokens(style, {})
    return _has_any(tokens, {"fachwerk", "mediterranean", "rowhouse", "reihenhaus", "siedlung", "gruenderzeit", "jugendstil"})


def _should_add_stair_core(
    style: FacadeStyleConfig,
    tags: dict[str, str],
    street_facing: bool,
    length: float,
    total_height: float,
) -> bool:
    if street_facing or length < 4.0 or total_height < 7.0:
        return False
    tokens = _style_tokens(style, tags)
    return _has_any(tokens, {"office", "parking", "plattenbau", "apartment", "apartments", "residential", "modern"})


def _floor_bottoms(floor_heights: list[float]) -> list[float]:
    bottoms = []
    current = 0.0
    for height in floor_heights:
        bottoms.append(current)
        current += height
    return bottoms


def _segments(points: list[Point3D]) -> list[tuple[Point3D, Point3D]]:
    return [(points[index], points[(index + 1) % len(points)]) for index in range(len(points))]


def _parse_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _tangent(start: Point3D, end: Point3D) -> Point2D:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = hypot(dx, dy)
    if length == 0:
        return (0.0, 0.0)
    return (dx / length, dy / length)


def _point_on_segment(start: Point3D, tangent: Point2D, normal: Point2D, offset: float, depth: float) -> Point2D:
    return (
        start[0] + tangent[0] * offset + normal[0] * depth,
        start[1] + tangent[1] * offset + normal[1] * depth,
    )


def _move(point: Point2D, tangent: Point2D, distance: float) -> Point2D:
    return (point[0] + tangent[0] * distance, point[1] + tangent[1] * distance)


def _offset_point(point: Point3D, normal: Point2D, depth: float) -> Point2D:
    return (point[0] + normal[0] * depth, point[1] + normal[1] * depth)


def _oriented_box(
    center: Point2D,
    tangent: Point2D,
    normal: Point2D,
    width: float,
    depth: float,
    height: float,
    bottom: float,
) -> tuple[list[Point3D], list[list[int]]]:
    half_width = width / 2.0
    half_depth = depth / 2.0
    corners = []
    for lateral, outward in [
        (-half_width, -half_depth),
        (half_width, -half_depth),
        (half_width, half_depth),
        (-half_width, half_depth),
    ]:
        corners.append((
            center[0] + tangent[0] * lateral + normal[0] * outward,
            center[1] + tangent[1] * lateral + normal[1] * outward,
        ))
    vertices = [(x, y, bottom) for x, y in corners] + [(x, y, bottom + height) for x, y in corners]
    faces = [
        [0, 1, 2, 3],
        [4, 7, 6, 5],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
    ]
    return vertices, faces


def _ledge_applies(style: FacadeStyleConfig, floor_index: int) -> bool:
    return style.ledge.enabled and style.ledge.every_n_floors > 0 and floor_index % style.ledge.every_n_floors == 0


def _balcony_applies(style: FacadeStyleConfig, floor_index: int) -> bool:
    return style.balcony.enabled and floor_index > 0 and style.balcony.every_n_floors > 0 and floor_index % style.balcony.every_n_floors == 0


def _street_facing_side_index(footprint: list[Point3D], tags: dict[str, str]) -> int:
    point = _street_reference_point_from_tags(tags)
    if point is not None:
        distances = [
            _point_segment_distance_sq_2d(point, (start[0], start[1]), (end[0], end[1]))
            for start, end in _segments(footprint)
        ]
        if distances:
            return min(range(len(distances)), key=lambda index: distances[index])

    value = tags.get("grammar:street_facing_side")
    if value is not None:
        try:
            return int(value) % len(footprint)
        except (TypeError, ValueError, ZeroDivisionError):
            return 0
    return 0


def _street_reference_point_from_tags(tags: dict[str, str]) -> Point2D | None:
    value = tags.get("grammar:street:point") or tags.get("grammar:street_point")
    if not value:
        return None
    parts = value.replace(";", ",").split(",")
    if len(parts) < 2:
        return None
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        return None


def _point_segment_distance_sq_2d(point: Point2D, start: Point2D, end: Point2D) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return (point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq
    t = min(1.0, max(0.0, t))
    closest = (start[0] + dx * t, start[1] + dy * t)
    return (point[0] - closest[0]) ** 2 + (point[1] - closest[1]) ** 2


def _door_applies(
    style: FacadeStyleConfig,
    side_index: int,
    street_side_index: int = 0,
    tags: dict[str, str] | None = None,
) -> bool:
    if tags and (tags.get("grammar:disable_ground_entrance") or "").strip().lower() in {"1", "yes", "true"}:
        return False
    placement = style.door.placement.lower()
    return style.door.enabled and placement != "none" and (
        placement == "each_facade"
        or side_index == street_side_index
    )


def _window_overlaps_door(offset: float, door_offset: float, style: FacadeStyleConfig) -> bool:
    clearance = (style.window.width + style.door.width) / 2.0 + max(style.door.frame_width, 0.0)
    return abs(offset - door_offset) < clearance


def _variant_wall_color(style: FacadeStyleConfig, source_name: str, side_index: int) -> tuple[int, tuple[float, float, float, float]]:
    colors = style.wall_color_variants
    mode = style.wall_color_variant_mode.lower()
    if not colors or mode == "none":
        return -1, style.wall_color
    if mode == "building":
        index = _stable_index(source_name, len(colors))
    elif mode == "facade":
        index = _stable_index(f"{source_name}:{side_index}", len(colors))
    else:
        index = side_index % len(colors)
    return index, colors[index]


def _row_wall_color(style: FacadeStyleConfig, floor_index: int) -> tuple[int, tuple[float, float, float, float]]:
    colors = style.wall_row_colors
    if not colors:
        return -1, style.wall_color
    mode = style.wall_row_color_mode.lower()
    if mode == "ground_accent" and floor_index == 0:
        return 0, colors[0]
    if mode == "ground_accent":
        index = 1 + (floor_index - 1) % max(1, len(colors) - 1)
        return min(index, len(colors) - 1), colors[min(index, len(colors) - 1)]
    index = floor_index % len(colors)
    return index, colors[index]


def _wall_material_name(base: str, kind: str, color_index: int) -> str:
    if color_index < 0:
        return base
    return f"{base} {kind} {color_index + 1}"


def _stable_index(value: str, count: int) -> int:
    if count <= 1:
        return 0
    total = 0
    for char in value:
        total = (total * 33 + ord(char)) % count
    return total


def _normalize_2d(value: Point2D) -> Point2D:
    length = hypot(value[0], value[1])
    if length == 0:
        return (0.0, 0.0)
    return (value[0] / length, value[1] / length)


def _centroid_2d(points: list[Point3D]) -> Point2D:
    return (sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points))


def _antenna_positions(center: Point2D, bounds: tuple[float, float, float, float], count: int) -> list[Point2D]:
    if count <= 1:
        return [center]
    min_x, min_y, max_x, max_y = bounds
    inset_x = max((max_x - min_x) * 0.22, 0.5)
    inset_y = max((max_y - min_y) * 0.22, 0.5)
    candidates = [
        center,
        (max_x - inset_x, max_y - inset_y),
        (min_x + inset_x, max_y - inset_y),
        (max_x - inset_x, min_y + inset_y),
        (min_x + inset_x, min_y + inset_y),
    ]
    return candidates[:count]


def _bounds(points: list[Point3D]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _roof_base_vertices(footprint: list[Point3D], height: float, overhang: float) -> list[Point3D]:
    if overhang <= 0:
        return [(x, y, height) for x, y, _z in footprint]
    center_x = sum(point[0] for point in footprint) / len(footprint)
    center_y = sum(point[1] for point in footprint) / len(footprint)
    vertices = []
    for x, y, _z in footprint:
        dx = x - center_x
        dy = y - center_y
        length = hypot(dx, dy)
        if length == 0:
            vertices.append((x, y, height))
        else:
            vertices.append((x + dx / length * overhang, y + dy / length * overhang, height))
    return vertices


def distance_2d(start: Point3D, end: Point3D) -> float:
    return hypot(end[0] - start[0], end[1] - start[1])
