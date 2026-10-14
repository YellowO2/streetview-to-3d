"""Each placed node's points moved into the world."""
from streetview_to_3d.common.Cloud import Cloud


def world_clouds(scene):
    """[Cloud in the world] of the scene's placed nodes, in order."""
    return [Cloud(n.cloud.world(n.transform), n.cloud.colours) for n in scene.placed()]
