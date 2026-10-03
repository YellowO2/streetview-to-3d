"""Configuration model for rule-based building generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


Color = tuple[float, float, float, float]
DEFAULT_EXCLUDED_BUILDING_VALUES = ("shelter",)
DEFAULT_BATCH_ROLES = (
    "window",
    "window_frame",
    "window_mullion",
    "window_sill",
    "ledge",
    "balcony",
    "balcony_rail",
    "balcony_bar",
    "door_frame",
    "door_handle",
    "door_canopy",
    "signboard",
    "awning",
    "garage_door",
    "loading_dock",
    "stair_core",
    "shutter",
    "facade_ornament",
    "panel_seam",
    "insulation_band",
    "roof_edge",
    "roof_lamp",
    "roof_tile",
    "roof_window",
    "dormer",
    "chimney",
    "gutter",
    "pv_panel",
    "hvac_unit",
    "roof_plant",
    "antenna",
    "antenna_panel",
)


@dataclass(slots=True)
class WindowStyleConfig:
    width: float = 1.25
    height: float = 1.55
    sill_height: float = 0.85
    spacing: float = 2.7
    min_margin: float = 0.8
    depth: float = 0.04
    material: str = "Grammar Glass"
    color: Color = (0.12, 0.22, 0.32, 1.0)
    texture_path: str | None = None
    texture_scale: float = 1.0
    frame_width: float = 0.08
    frame_depth: float = 0.03
    frame_material: str = "Grammar Window Frames"
    frame_color: Color = (0.86, 0.84, 0.78, 1.0)
    vertical_mullions: int = 1
    horizontal_mullions: int = 0
    sill_depth: float = 0.16
    sill_thickness: float = 0.06
    sill_material: str = "Grammar Window Sills"
    sill_color: Color = (0.78, 0.74, 0.68, 1.0)


@dataclass(slots=True)
class LedgeStyleConfig:
    enabled: bool = True
    depth: float = 0.16
    height: float = 0.08
    every_n_floors: int = 1
    material: str = "Grammar Ledges"
    color: Color = (0.78, 0.74, 0.68, 1.0)
    texture_path: str | None = None
    texture_scale: float = 1.0


@dataclass(slots=True)
class BalconyStyleConfig:
    enabled: bool = True
    width: float = 1.9
    depth: float = 0.75
    slab_height: float = 0.12
    railing_height: float = 0.9
    every_n_floors: int = 2
    material: str = "Grammar Balconies"
    color: Color = (0.58, 0.58, 0.55, 1.0)
    texture_path: str | None = None
    texture_scale: float = 1.0
    railing_material: str = "Grammar Balcony Railings"
    railing_color: Color = (0.16, 0.16, 0.15, 1.0)
    railing_bar_count: int = 5
    railing_bar_width: float = 0.04
    railing_bar_depth: float = 0.04


@dataclass(slots=True)
class DoorStyleConfig:
    enabled: bool = True
    placement: str = "first_facade"
    width: float = 1.25
    height: float = 2.25
    depth: float = 0.08
    material: str = "Grammar Door"
    color: Color = (0.16, 0.1, 0.06, 1.0)
    texture_path: str | None = None
    texture_scale: float = 1.0
    frame_width: float = 0.12
    frame_depth: float = 0.04
    frame_material: str = "Grammar Door Frames"
    frame_color: Color = (0.72, 0.68, 0.6, 1.0)
    handle_enabled: bool = True
    handle_radius: float = 0.05
    handle_material: str = "Grammar Door Handles"
    handle_color: Color = (0.82, 0.66, 0.32, 1.0)
    canopy_enabled: bool = False
    canopy_width: float = 1.8
    canopy_depth: float = 0.75
    canopy_thickness: float = 0.08
    canopy_material: str = "Grammar Door Canopies"
    canopy_color: Color = (0.36, 0.36, 0.34, 1.0)


@dataclass(slots=True)
class AntennaStyleConfig:
    enabled: bool = True
    type: str = "tv"
    count: int = 1
    mast_height: float = 1.4
    mast_radius: float = 0.035
    base_width: float = 0.45
    base_depth: float = 0.45
    base_height: float = 0.18
    panel_width: float = 0.35
    panel_height: float = 0.7
    panel_depth: float = 0.06
    material: str = "Grammar Antenna Metal"
    color: Color = (0.34, 0.34, 0.33, 1.0)
    accent_material: str = "Grammar Antenna Panels"
    accent_color: Color = (0.78, 0.78, 0.72, 1.0)


@dataclass(slots=True)
class RoofStyleConfig:
    type: str = "flat"
    height: float = 1.6
    overhang: float = 0.25
    ridge_alignment: str = "closest_street"
    material: str = "Grammar Roof"
    color: Color = (0.34, 0.08, 0.06, 1.0)
    texture_path: str | None = None
    texture_scale: float = 1.0
    edge_enabled: bool = True
    edge_width: float = 0.28
    edge_height: float = 0.35
    surface_inset: float = 0.08
    edge_material: str = "Grammar Roof Edge"
    edge_color: Color = (0.22, 0.22, 0.2, 1.0)
    corner_cap_size: float = 0.42
    tile_rows: int = 6
    tile_depth: float = 0.035
    tile_spacing: float = 0.55
    tile_material: str = "Grammar Roof Tile Bands"
    tile_color: Color = (0.28, 0.07, 0.045, 1.0)
    dormer_count: int = 0
    dormer_width: float = 1.35
    dormer_depth: float = 0.9
    dormer_height: float = 0.9
    dormer_material: str = "Grammar Dormer Cladding"
    dormer_color: Color = (0.62, 0.58, 0.5, 1.0)
    roof_window_count: int = 0
    roof_window_width: float = 0.75
    roof_window_height: float = 1.05
    roof_window_material: str = "Grammar Roof Window Glass"
    roof_window_color: Color = (0.08, 0.16, 0.2, 0.86)
    chimney_count: int = 0
    chimney_width: float = 0.45
    chimney_depth: float = 0.38
    chimney_height: float = 1.15
    chimney_material: str = "Grammar Brick Chimney"
    chimney_color: Color = (0.42, 0.16, 0.1, 1.0)


@dataclass(slots=True)
class FacadeStyleConfig:
    name: str = "default"
    building_values: list[str] = field(default_factory=list)
    tag_filters: dict[str, list[str]] = field(default_factory=dict)
    default_levels: int | None = None
    default_floor_height: float | None = None
    roof: RoofStyleConfig | None = None
    wall_material: str = "Grammar Facade"
    wall_color: Color = (0.72, 0.68, 0.6, 1.0)
    wall_texture_path: str | None = None
    wall_texture_scale: float = 1.0
    wall_color_variants: list[Color] = field(default_factory=list)
    wall_color_variant_mode: str = "none"
    wall_row_colors: list[Color] = field(default_factory=list)
    wall_row_color_mode: str = "cycle"
    window: WindowStyleConfig = field(default_factory=WindowStyleConfig)
    ledge: LedgeStyleConfig = field(default_factory=LedgeStyleConfig)
    balcony: BalconyStyleConfig = field(default_factory=BalconyStyleConfig)
    door: DoorStyleConfig = field(default_factory=DoorStyleConfig)
    antenna: AntennaStyleConfig = field(default_factory=AntennaStyleConfig)


@dataclass(slots=True)
class BuildingGrammarConfig:
    root_collection: str = "Procedural Buildings"
    source_collection: str | None = "Buildings"
    include_selected_only: bool = False
    replace_existing: bool = True
    enable_building_parts: bool = True
    skip_parent_footprints_with_parts: bool = True
    inherit_parent_tags_for_parts: bool = True
    building_part_match_tolerance: float = 0.25
    use_mesh_instancing: bool = True
    batch_generated_meshes: bool = True
    batch_roles: list[str] = field(default_factory=lambda: list(DEFAULT_BATCH_ROLES))
    default_levels: int = 4
    default_floor_height: float = 3.1
    irregular_floor_heights: dict[int, float] = field(default_factory=dict)
    excluded_building_values: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDED_BUILDING_VALUES))
    styles: list[FacadeStyleConfig] = field(default_factory=lambda: [FacadeStyleConfig()])
    roof: RoofStyleConfig = field(default_factory=RoofStyleConfig)
    roof_street_alignment_search_radius: float = 80.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BuildingGrammarConfig":
        return cls(
            root_collection=str(data.get("root_collection", "Procedural Buildings")),
            source_collection=data.get("source_collection", "Buildings"),
            include_selected_only=bool(data.get("include_selected_only", False)),
            replace_existing=bool(data.get("replace_existing", True)),
            enable_building_parts=bool(data.get("enable_building_parts", True)),
            skip_parent_footprints_with_parts=bool(data.get("skip_parent_footprints_with_parts", True)),
            inherit_parent_tags_for_parts=bool(data.get("inherit_parent_tags_for_parts", True)),
            building_part_match_tolerance=float(data.get("building_part_match_tolerance", 0.25)),
            use_mesh_instancing=bool(data.get("use_mesh_instancing", True)),
            batch_generated_meshes=bool(data.get("batch_generated_meshes", True)),
            batch_roles=_string_list(data.get("batch_roles", DEFAULT_BATCH_ROLES)) or list(DEFAULT_BATCH_ROLES),
            default_levels=int(data.get("default_levels", 4)),
            default_floor_height=float(data.get("default_floor_height", 3.1)),
            irregular_floor_heights={
                int(key): float(value)
                for key, value in data.get("irregular_floor_heights", {}).items()
            },
            excluded_building_values=_string_list(data.get("excluded_building_values", DEFAULT_EXCLUDED_BUILDING_VALUES)),
            styles=[_style_from_dict(item) for item in data.get("styles", [{}])],
            roof=_roof_from_dict(data.get("roof", {})),
            roof_street_alignment_search_radius=float(data.get("roof_street_alignment_search_radius", 80.0)),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "BuildingGrammarConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_collection": self.root_collection,
            "source_collection": self.source_collection,
            "include_selected_only": self.include_selected_only,
            "replace_existing": self.replace_existing,
            "enable_building_parts": self.enable_building_parts,
            "skip_parent_footprints_with_parts": self.skip_parent_footprints_with_parts,
            "inherit_parent_tags_for_parts": self.inherit_parent_tags_for_parts,
            "building_part_match_tolerance": self.building_part_match_tolerance,
            "use_mesh_instancing": self.use_mesh_instancing,
            "batch_generated_meshes": self.batch_generated_meshes,
            "batch_roles": self.batch_roles,
            "default_levels": self.default_levels,
            "default_floor_height": self.default_floor_height,
            "irregular_floor_heights": self.irregular_floor_heights,
            "excluded_building_values": self.excluded_building_values,
            "styles": [_style_to_dict(style) for style in self.styles],
            "roof": _roof_to_dict(self.roof),
            "roof_street_alignment_search_radius": self.roof_street_alignment_search_radius,
        }


def ensure_config(config: BuildingGrammarConfig | dict[str, Any] | str | Path | None) -> BuildingGrammarConfig:
    if config is None:
        return BuildingGrammarConfig()
    if isinstance(config, BuildingGrammarConfig):
        return config
    if isinstance(config, (str, Path)):
        return BuildingGrammarConfig.from_json_file(config)
    return BuildingGrammarConfig.from_dict(config)


def building_value_is_excluded(tags: dict[str, str], config: BuildingGrammarConfig) -> bool:
    building_value = tags.get("building")
    if building_value is None:
        return False
    excluded = {value.strip().lower() for value in config.excluded_building_values}
    return building_value.strip().lower() in excluded and not matching_styles_for_tags(tags, config.styles)


def matching_styles_for_tags(tags: dict[str, str], styles: list[FacadeStyleConfig]) -> list[FacadeStyleConfig]:
    return [style for style in styles if style_matches_tags(style, tags)]


def selectable_styles_for_tags(tags: dict[str, str], styles: list[FacadeStyleConfig]) -> list[FacadeStyleConfig]:
    matched = matching_styles_for_tags(tags, styles)
    if matched:
        return matched
    semantic = semantic_styles_for_tags(tags, styles)
    if semantic:
        return semantic
    unfiltered = [style for style in styles if not style.building_values and not style.tag_filters]
    return unfiltered or styles


def style_matches_tags(style: FacadeStyleConfig, tags: dict[str, str]) -> bool:
    if style.building_values:
        building_values = {
            tags.get("building", ""),
            tags.get("building:part", ""),
            tags.get("building:use", ""),
        }
        normalized_building_values = {value.strip().lower() for value in building_values if value}
        if normalized_building_values & {value.strip().lower() for value in style.building_values}:
            return True
    for key, values in style.tag_filters.items():
        tag_value = tags.get(key)
        if tag_value is None:
            continue
        allowed = {value.strip().lower() for value in values}
        if tag_value.strip().lower() in allowed or "*" in allowed:
            return True
    return False


def semantic_styles_for_tags(tags: dict[str, str], styles: list[FacadeStyleConfig]) -> list[FacadeStyleConfig]:
    keywords = _semantic_style_keywords(tags)
    if not keywords:
        return []
    scored: list[tuple[int, int, FacadeStyleConfig]] = []
    for index, style in enumerate(styles):
        haystack = " ".join([style.name, style.wall_material]).lower()
        score = sum(weight for keyword, weight in keywords.items() if keyword in haystack)
        if score > 0:
            scored.append((score, -index, style))
    scored.sort(reverse=True)
    return [style for _score, _index, style in scored]


def _semantic_style_keywords(tags: dict[str, str]) -> dict[str, int]:
    keywords: dict[str, int] = {}
    building = tags.get("building", "").strip().lower()
    building_use = tags.get("building:use", "").strip().lower()
    shop = tags.get("shop", "").strip().lower()
    office = tags.get("office", "").strip().lower()
    industrial = tags.get("industrial", "").strip().lower()
    landuse = tags.get("landuse", "").strip().lower()
    amenity = tags.get("amenity", "").strip().lower()
    religion = tags.get("religion", "").strip().lower()

    if shop:
        keywords.update({"retail": 4, "shop": 4})
        if shop == "supermarket" or building == "supermarket":
            keywords["supermarket"] = 8
    if office or building in {"office", "commercial"} or building_use == "office":
        keywords.update({"office": 7, "curtain": 2, "atrium": 2, "steel": 1})
    if industrial or landuse == "industrial" or building in {"industrial", "warehouse", "factory", "manufacture"}:
        keywords.update({"industrial": 8, "warehouse": 6, "factory": 4})
    if building in {"church", "cathedral", "chapel", "religious"} or amenity == "place_of_worship":
        keywords.update({"church": 8, "cathedral": 6, "sacral": 5, "stone": 2, "historic": 2})
        if building == "cathedral" or tags.get("name", "").strip().lower().find("cathedral") >= 0:
            keywords["cathedral"] = 10
        if religion:
            keywords[religion] = 2

    year = _start_year(tags.get("start_date") or tags.get("building:start_date"))
    if year is not None:
        if year < 1918:
            keywords.update({"gruenderzeit": 4, "jugendstil": 3, "fachwerk": 2, "historic": 2})
        elif year < 1935:
            keywords.update({"bauhaus": 5, "siedlung": 4})
        elif year < 1975:
            keywords.update({"postwar": 4, "nachkriegsmoderne": 4})
        elif year < 1995:
            keywords.update({"plattenbau": 5, "prefab": 3})
        else:
            keywords.update({"contemporary": 4, "modern": 3, "passivhaus": 2})
    return keywords


def _start_year(value: str | None) -> int | None:
    if not value:
        return None
    digits = "".join(char if char.isdigit() else " " for char in value).split()
    if not digits:
        return None
    try:
        year = int(digits[0][:4])
    except ValueError:
        return None
    return year if 1000 <= year <= 2200 else None


def _style_from_dict(data: dict[str, Any]) -> FacadeStyleConfig:
    return FacadeStyleConfig(
        name=str(data.get("name", "default")),
        building_values=_string_list(data.get("building_values", [])),
        tag_filters={
            str(key): _string_list(value)
            for key, value in data.get("tag_filters", {}).items()
        },
        default_levels=int(data["default_levels"]) if data.get("default_levels") is not None else None,
        default_floor_height=float(data["default_floor_height"]) if data.get("default_floor_height") is not None else None,
        roof=_roof_from_dict(data["roof"]) if data.get("roof") else None,
        wall_material=str(data.get("wall_material", "Grammar Facade")),
        wall_color=_color(data.get("wall_color"), (0.72, 0.68, 0.6, 1.0)),
        wall_texture_path=data.get("wall_texture_path"),
        wall_texture_scale=float(data.get("wall_texture_scale", 1.0)),
        wall_color_variants=_color_list(data.get("wall_color_variants")),
        wall_color_variant_mode=str(data.get("wall_color_variant_mode", "none")),
        wall_row_colors=_color_list(data.get("wall_row_colors")),
        wall_row_color_mode=str(data.get("wall_row_color_mode", "cycle")),
        window=_window_from_dict(data.get("window", {})),
        ledge=_ledge_from_dict(data.get("ledge", {})),
        balcony=_balcony_from_dict(data.get("balcony", {})),
        door=_door_from_dict(data.get("door", {})),
        antenna=_antenna_from_dict(data.get("antenna", {})),
    )


def _window_from_dict(data: dict[str, Any]) -> WindowStyleConfig:
    return WindowStyleConfig(
        width=float(data.get("width", 1.25)),
        height=float(data.get("height", 1.55)),
        sill_height=float(data.get("sill_height", 0.85)),
        spacing=float(data.get("spacing", 2.7)),
        min_margin=float(data.get("min_margin", 0.8)),
        depth=float(data.get("depth", 0.04)),
        material=str(data.get("material", "Grammar Glass")),
        color=_color(data.get("color"), (0.12, 0.22, 0.32, 1.0)),
        texture_path=data.get("texture_path"),
        texture_scale=float(data.get("texture_scale", 1.0)),
        frame_width=float(data.get("frame_width", 0.08)),
        frame_depth=float(data.get("frame_depth", 0.03)),
        frame_material=str(data.get("frame_material", "Grammar Window Frames")),
        frame_color=_color(data.get("frame_color"), (0.86, 0.84, 0.78, 1.0)),
        vertical_mullions=int(data.get("vertical_mullions", 1)),
        horizontal_mullions=int(data.get("horizontal_mullions", 0)),
        sill_depth=float(data.get("sill_depth", 0.16)),
        sill_thickness=float(data.get("sill_thickness", 0.06)),
        sill_material=str(data.get("sill_material", "Grammar Window Sills")),
        sill_color=_color(data.get("sill_color"), (0.78, 0.74, 0.68, 1.0)),
    )


def _ledge_from_dict(data: dict[str, Any]) -> LedgeStyleConfig:
    return LedgeStyleConfig(
        enabled=bool(data.get("enabled", True)),
        depth=float(data.get("depth", 0.16)),
        height=float(data.get("height", 0.08)),
        every_n_floors=int(data.get("every_n_floors", 1)),
        material=str(data.get("material", "Grammar Ledges")),
        color=_color(data.get("color"), (0.78, 0.74, 0.68, 1.0)),
        texture_path=data.get("texture_path"),
        texture_scale=float(data.get("texture_scale", 1.0)),
    )


def _balcony_from_dict(data: dict[str, Any]) -> BalconyStyleConfig:
    return BalconyStyleConfig(
        enabled=bool(data.get("enabled", True)),
        width=float(data.get("width", 1.9)),
        depth=float(data.get("depth", 0.75)),
        slab_height=float(data.get("slab_height", 0.12)),
        railing_height=float(data.get("railing_height", 0.9)),
        every_n_floors=int(data.get("every_n_floors", 2)),
        material=str(data.get("material", "Grammar Balconies")),
        color=_color(data.get("color"), (0.58, 0.58, 0.55, 1.0)),
        texture_path=data.get("texture_path"),
        texture_scale=float(data.get("texture_scale", 1.0)),
        railing_material=str(data.get("railing_material", "Grammar Balcony Railings")),
        railing_color=_color(data.get("railing_color"), (0.16, 0.16, 0.15, 1.0)),
        railing_bar_count=int(data.get("railing_bar_count", 5)),
        railing_bar_width=float(data.get("railing_bar_width", 0.04)),
        railing_bar_depth=float(data.get("railing_bar_depth", 0.04)),
    )


def _door_from_dict(data: dict[str, Any]) -> DoorStyleConfig:
    return DoorStyleConfig(
        enabled=bool(data.get("enabled", True)),
        placement=str(data.get("placement", "first_facade")),
        width=float(data.get("width", 1.25)),
        height=float(data.get("height", 2.25)),
        depth=float(data.get("depth", 0.08)),
        material=str(data.get("material", "Grammar Door")),
        color=_color(data.get("color"), (0.16, 0.1, 0.06, 1.0)),
        texture_path=data.get("texture_path"),
        texture_scale=float(data.get("texture_scale", 1.0)),
        frame_width=float(data.get("frame_width", 0.12)),
        frame_depth=float(data.get("frame_depth", 0.04)),
        frame_material=str(data.get("frame_material", "Grammar Door Frames")),
        frame_color=_color(data.get("frame_color"), (0.72, 0.68, 0.6, 1.0)),
        handle_enabled=bool(data.get("handle_enabled", True)),
        handle_radius=float(data.get("handle_radius", 0.05)),
        handle_material=str(data.get("handle_material", "Grammar Door Handles")),
        handle_color=_color(data.get("handle_color"), (0.82, 0.66, 0.32, 1.0)),
        canopy_enabled=bool(data.get("canopy_enabled", False)),
        canopy_width=float(data.get("canopy_width", 1.8)),
        canopy_depth=float(data.get("canopy_depth", 0.75)),
        canopy_thickness=float(data.get("canopy_thickness", 0.08)),
        canopy_material=str(data.get("canopy_material", "Grammar Door Canopies")),
        canopy_color=_color(data.get("canopy_color"), (0.36, 0.36, 0.34, 1.0)),
    )


def _antenna_from_dict(data: dict[str, Any]) -> AntennaStyleConfig:
    return AntennaStyleConfig(
        enabled=bool(data.get("enabled", True)),
        type=str(data.get("type", "tv")),
        count=int(data.get("count", 1)),
        mast_height=float(data.get("mast_height", 1.4)),
        mast_radius=float(data.get("mast_radius", 0.035)),
        base_width=float(data.get("base_width", 0.45)),
        base_depth=float(data.get("base_depth", 0.45)),
        base_height=float(data.get("base_height", 0.18)),
        panel_width=float(data.get("panel_width", 0.35)),
        panel_height=float(data.get("panel_height", 0.7)),
        panel_depth=float(data.get("panel_depth", 0.06)),
        material=str(data.get("material", "Grammar Antenna Metal")),
        color=_color(data.get("color"), (0.34, 0.34, 0.33, 1.0)),
        accent_material=str(data.get("accent_material", "Grammar Antenna Panels")),
        accent_color=_color(data.get("accent_color"), (0.78, 0.78, 0.72, 1.0)),
    )


def _roof_from_dict(data: dict[str, Any]) -> RoofStyleConfig:
    return RoofStyleConfig(
        type=str(data.get("type", "flat")),
        height=float(data.get("height", 1.6)),
        overhang=float(data.get("overhang", 0.25)),
        ridge_alignment=str(data.get("ridge_alignment", "closest_street")),
        material=str(data.get("material", "Grammar Roof")),
        color=_color(data.get("color"), (0.34, 0.08, 0.06, 1.0)),
        texture_path=data.get("texture_path"),
        texture_scale=float(data.get("texture_scale", 1.0)),
        edge_enabled=bool(data.get("edge_enabled", True)),
        edge_width=float(data.get("edge_width", 0.28)),
        edge_height=float(data.get("edge_height", 0.35)),
        surface_inset=float(data.get("surface_inset", 0.08)),
        edge_material=str(data.get("edge_material", "Grammar Roof Edge")),
        edge_color=_color(data.get("edge_color"), (0.22, 0.22, 0.2, 1.0)),
        corner_cap_size=float(data.get("corner_cap_size", 0.42)),
        tile_rows=int(data.get("tile_rows", 6)),
        tile_depth=float(data.get("tile_depth", 0.035)),
        tile_spacing=float(data.get("tile_spacing", 0.55)),
        tile_material=str(data.get("tile_material", "Grammar Roof Tile Bands")),
        tile_color=_color(data.get("tile_color"), (0.28, 0.07, 0.045, 1.0)),
        dormer_count=int(data.get("dormer_count", 0)),
        dormer_width=float(data.get("dormer_width", 1.35)),
        dormer_depth=float(data.get("dormer_depth", 0.9)),
        dormer_height=float(data.get("dormer_height", 0.9)),
        dormer_material=str(data.get("dormer_material", "Grammar Dormer Cladding")),
        dormer_color=_color(data.get("dormer_color"), (0.62, 0.58, 0.5, 1.0)),
        roof_window_count=int(data.get("roof_window_count", 0)),
        roof_window_width=float(data.get("roof_window_width", 0.75)),
        roof_window_height=float(data.get("roof_window_height", 1.05)),
        roof_window_material=str(data.get("roof_window_material", "Grammar Roof Window Glass")),
        roof_window_color=_color(data.get("roof_window_color"), (0.08, 0.16, 0.2, 0.86)),
        chimney_count=int(data.get("chimney_count", 0)),
        chimney_width=float(data.get("chimney_width", 0.45)),
        chimney_depth=float(data.get("chimney_depth", 0.38)),
        chimney_height=float(data.get("chimney_height", 1.15)),
        chimney_material=str(data.get("chimney_material", "Grammar Brick Chimney")),
        chimney_color=_color(data.get("chimney_color"), (0.42, 0.16, 0.1, 1.0)),
    )


def _style_to_dict(style: FacadeStyleConfig) -> dict[str, Any]:
    return {
        "name": style.name,
        "building_values": style.building_values,
        "tag_filters": style.tag_filters,
        "default_levels": style.default_levels,
        "default_floor_height": style.default_floor_height,
        "roof": _roof_to_dict(style.roof) if style.roof is not None else None,
        "wall_material": style.wall_material,
        "wall_color": style.wall_color,
        "wall_texture_path": style.wall_texture_path,
        "wall_texture_scale": style.wall_texture_scale,
        "wall_color_variants": style.wall_color_variants,
        "wall_color_variant_mode": style.wall_color_variant_mode,
        "wall_row_colors": style.wall_row_colors,
        "wall_row_color_mode": style.wall_row_color_mode,
        "window": asdict(style.window),
        "ledge": asdict(style.ledge),
        "balcony": asdict(style.balcony),
        "door": asdict(style.door),
        "antenna": asdict(style.antenna),
    }


def _roof_to_dict(roof: RoofStyleConfig) -> dict[str, Any]:
    return asdict(roof)


def _color(value: Any, fallback: Color) -> Color:
    if value is None:
        return fallback
    channels = tuple(float(channel) for channel in value)
    if len(channels) == 3:
        return channels + (1.0,)
    if len(channels) == 4:
        return channels
    return fallback


def _color_list(value: Any) -> list[Color]:
    if not value:
        return []
    return [_color(item, (0.72, 0.68, 0.6, 1.0)) for item in value]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]
