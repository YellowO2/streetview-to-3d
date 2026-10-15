"""The world laid round the scene, as the files the scene keeps it in."""
import logging

import numpy as np

from streetview_to_3d.postprocess.World import World
from streetview_to_3d.postprocess.world.tools.file_names import BLOCKS, BUILDINGS, LAND, LIFE, ROADS, TERRAIN, WATER
from streetview_to_3d.postprocess.world.tools.reach import RADIUS_M

logger = logging.getLogger(__name__)


def world_files(land, roads, buildings, water, life):
    """World of land (Land, coloured), roads (Roads), buildings (Buildings), water (Water) and
    life (life.json's contents); says what went into it."""
    world = World()
    world.meshes[LAND] = dict(pts=land.pts, cols=land.cols, faces=land.faces, gap=land.gap, near=land.near)
    if len(roads.mesh["faces"]):
        world.meshes[ROADS] = roads.mesh
    points = roads.points
    if len(points.pts):
        world.points[TERRAIN] = dict(pts=points.pts, cols=points.cols, gap=points.gap)
    if water.surfaces:
        world.json[WATER] = {"surfaces": water.surfaces, "shore": water.shore_grid()}
    world.json[LIFE] = life
    blocks = buildings.blocks
    if len(blocks.pts):
        world.points[BUILDINGS] = dict(pts=blocks.pts, cols=blocks.cols, gap=blocks.gap, normal=blocks.normal,
                                       kind=blocks.kind, near=blocks.near)
    if buildings.solid is not None:
        world.meshes[BLOCKS] = buildings.solid
    _say(land, roads, buildings, water, life)
    return world


def _say(land, roads, buildings, water, life):
    """Log what the world holds, in one line."""
    net, outlines, blocks = roads.network, buildings.outlines, buildings.blocks
    elevation = land.elevation
    fix = np.abs(elevation.fixes - elevation.shift)
    n_roads = len(net.strips) + len(net.fills) if net.shapes else 0
    n_solid = int((~blocks.reached).sum())
    painted = roads.points.painted + land.painted + blocks.painted
    logger.info(f"terrain: land {len(land.pts)} vertices, {len(land.faces)} triangles to {RADIUS_M:.0f} m, "
                f"{len(roads.points.pts)} road points, {len(roads.mesh['faces'])} road triangles, "
                f"{len(blocks.pts)} building points, {painted} of them "
                f"painted from the panos ({len(water.surfaces)} water surfaces ({water.source}), {land.source} colour, "
                f"{len(outlines.items)} buildings ({n_solid} solid) -- {outlines.fitted} fitted onto DA3's walls, "
                f"{outlines.trimmed} trimmed to them, {blocks.cut} of their points "
                f"left to DA3's own, {blocks.seen} coloured by the panos, {outlines.satellite_roofs} roofs by the "
                f"satellite -- {n_roads} roads, {len(roads.height.decks)} bridges, "
                f"{len(roads.mouths[2]) // 14} tunnel mouths, {len(life['cars']['roads'])} stretches of road for cars, "
                f"keeping {life['cars']['side']}), map shifted "
                f"{elevation.shift:+.1f} m to Google's datum, the ground near {elevation.known} panos and "
                f"{elevation.more} other places "
                f"their elevation (the map off it by up to {fix.max() if len(fix) else 0:.1f} m, "
                f"median {np.median(fix) if len(fix) else 0:.1f})")
