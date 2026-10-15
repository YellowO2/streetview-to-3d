"""Turn a prepared corridor's panos into one scene of DA3 point clouds."""
from streetview_to_3d.reconstruct.WalkSettings import WalkSettings
from streetview_to_3d.reconstruct.assemble import assemble
from streetview_to_3d.reconstruct.open_scene import open_scene
from streetview_to_3d.reconstruct.walk import walk


def reconstruct(corridor, settings: WalkSettings = WalkSettings()):
    """The Scene of the corridor (panos.main.panos): a node per dot, those the walk reached
    with DA3's pose and points in their piece's frame. Placement happens afterwards, in postprocess/."""
    scene = open_scene(corridor)
    pieces = walk(corridor, settings)
    scene = assemble(scene, pieces, corridor)
    return scene
