"""Lay one ground under a placed scene, coloured from its panos.

Pieces are levelled so their roads meet, every cloud's walkable ground becomes one ground,
painted from the panos; each node takes the ground its pano painted, and its trees a sway.
"""
from streetview_to_3d.postprocess.fill.handed_out import handed_out
from streetview_to_3d.postprocess.fill.level import level
from streetview_to_3d.postprocess.fill.on_road import on_road
from streetview_to_3d.postprocess.fill.one_ground import one_ground
from streetview_to_3d.postprocess.fill.painted import painted
from streetview_to_3d.postprocess.fill.placed_cameras import placed_cameras
from streetview_to_3d.postprocess.fill.scene_ground import scene_ground
from streetview_to_3d.postprocess.fill.walkable_ground import walkable_ground
from streetview_to_3d.postprocess.fill.world_clouds import world_clouds


def fill(scene):
    """(the filled scene, its SceneGround); a scene with nothing placed comes back as it is,
    with no ground."""
    clouds = world_clouds(scene)
    if not clouds:
        return scene, None
    cameras = placed_cameras(scene)
    walkable = walkable_ground(clouds, cameras)
    scene, clouds = level(scene, clouds, walkable)
    cameras = placed_cameras(scene)
    clouds, surface = one_ground(clouds, cameras, walkable)
    surface = painted(surface, clouds, cameras)
    surface = on_road(surface, cameras)
    ground = scene_ground(surface)
    scene = handed_out(scene, clouds, surface, ground, cameras)
    return scene, ground
