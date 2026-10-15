"""Build a prepared corridor's panoramas into a finished 3D scene on disk.

The corridor is prepared first, without the GPU (panos.main.panos): the UI's first button.
Its second button runs this: DA3 on the GPU, then everything after it on the CPU.
"""
from streetview_to_3d.Settings import Settings
from streetview_to_3d.models.DepthModel import DepthModel
from streetview_to_3d.models.Segmenter import Segmenter
from streetview_to_3d.postprocess.main import postprocess
from streetview_to_3d.reconstruct.main import reconstruct
from streetview_to_3d.save import save


def streetview_to_3d(corridor, settings: Settings = Settings()):
    """The folder holding the finished scene (scene.json beside its clouds, ground and world)."""
    depth_model = DepthModel(settings.walk.model)
    masker = Segmenter(settings.walk.masker)
    scene = reconstruct(corridor, settings.walk, depth_model, masker)
    scene, ground, world = postprocess(scene, settings.post)
    folder = save(scene, ground, world)
    return folder
