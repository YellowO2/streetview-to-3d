"""Grid the one ground into the scene's ground map, as the world reads it."""
from streetview_to_3d.build_scene.postprocess.SceneGround import SceneGround


def scene_ground(ground):
    """The SceneGround of a PaintedGround (SceneGround.py: height, colour and road share per square)."""
    return SceneGround.from_points(ground.points, ground.colours, ground.road)
