"""Build a prepared corridor's panoramas into a finished 3D scene on disk.

The corridor is prepared first, without the GPU (panos.main.panos): the UI's first button.
Its second button runs this: DA3 on the GPU, then everything after it on the CPU.
"""
from streetview_to_3d.Settings import Settings
from streetview_to_3d.postprocess.main import postprocess
from streetview_to_3d.reconstruct.main import reconstruct
from streetview_to_3d.run_folder import run_folder
from streetview_to_3d.save import save


def streetview_to_3d(prep, cfg: Settings = Settings(), log=print):
    """The folder holding the finished scene (scene.json beside its clouds, ground and world)."""
    folder = run_folder(log)
    scene = reconstruct(prep, cfg.walk)
    scene, ground, world = postprocess(scene, cfg.post, folder, log)
    folder = save(scene, ground, world, folder)
    return folder
