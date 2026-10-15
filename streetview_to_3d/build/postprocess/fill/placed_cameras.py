"""The cameras of a scene's placed nodes, where they stand now."""
from streetview_to_3d.build.postprocess.Camera import Camera


def placed_cameras(scene):
    """[Camera] of the scene's placed nodes, in order, each with its pano's photo."""
    return [Camera(n) for n in scene.placed()]
