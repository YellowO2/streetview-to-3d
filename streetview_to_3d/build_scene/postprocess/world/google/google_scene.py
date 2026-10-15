"""The scene told to draw Google's points instead of the map's world."""
from streetview_to_3d.build_scene.postprocess.world.tools.file_names import GOOGLE


def google_scene(scene):
    """scene with google.ply named, and the map-built land, roads and buildings no longer."""
    scene.google, scene.land, scene.terrain, scene.roads, scene.buildings, scene.blocks = GOOGLE, None, None, None, None, None
    return scene
