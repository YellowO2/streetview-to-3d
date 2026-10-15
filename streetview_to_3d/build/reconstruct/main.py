"""Turn a prepared corridor's panos into one scene of DA3 point clouds."""
from streetview_to_3d.build.models.DepthModel import DepthModel
from streetview_to_3d.build.models.Segmenter import Segmenter
from streetview_to_3d.build.reconstruct.PanoDepth import PanoDepth
from streetview_to_3d.build.reconstruct.WalkSettings import WalkSettings
from streetview_to_3d.build.reconstruct.assemble import assemble
from streetview_to_3d.build.reconstruct.open_scene import open_scene
from streetview_to_3d.build.reconstruct.pieces import pieces as shape_pieces
from streetview_to_3d.build.reconstruct.walk import walk


def reconstruct(corridor, settings: WalkSettings, depth_model: DepthModel, masker: Segmenter):
    """The Scene of the corridor (panos.main.panos): a node per dot, those the walk reached
    with DA3's (depth_model's) pose and points in their piece's frame, what masker labels
    left out, and its pano's class map from masker. Placement happens afterwards, in
    postprocess/."""
    scene = open_scene(corridor)
    depth = PanoDepth(depth_model, masker, settings)
    segments, labels = walk(corridor, depth, settings)
    pieces = shape_pieces(segments)
    scene = assemble(scene, pieces, corridor, labels)
    return scene
