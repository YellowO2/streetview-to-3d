"""Turn a prepared corridor's panos into one scene of DA3 point clouds."""
from streetview_to_3d.reconstruct.WalkSettings import WalkSettings
from streetview_to_3d.reconstruct.assemble import assemble
from streetview_to_3d.reconstruct.open_scene import open_scene
from streetview_to_3d.reconstruct.walk import walk


def reconstruct(prep, settings: WalkSettings = WalkSettings()):
    """The Scene of prep's corridor (panos.main.panos): a node per dot, those the walk reached
    with DA3's pose and points in their piece's frame. Placement happens afterwards, in postprocess/."""
    scene = open_scene(prep)
    pieces = walk(prep, settings)
    scene = assemble(scene, pieces, prep)
    return scene
