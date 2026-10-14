"""Grid the one ground into the scene's ground map, as the world reads it."""
from streetview_to_3d.postprocess.seams import SceneGround


def scene_ground(ground):
    """The SceneGround of a PaintedGround (seams.py: height, colour and road share per square)."""
    return SceneGround.from_points(ground.points, ground.colours, ground.road)
