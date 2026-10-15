"""The scene naming the files its world is saved as."""
from streetview_to_3d.build_scene.postprocess.world.tools.file_names import FIELDS


def scene_files(scene, world):
    """scene with each of its world's files (World) named, the rest None."""
    for field, name in FIELDS:
        setattr(scene, field, name if name in world.meshes or name in world.points or name in world.json
                else None)
    return scene
