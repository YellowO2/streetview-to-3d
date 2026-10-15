"""The models: DA3 depth (DepthModel.py) and the segmenter (Segmenter.py), handed to the steps
that run them. ui/pano_api.py runs them on one pano over the Space's API; drawn locally:

    python -m scripts.draw_mask PANO_ID OUT.jpg

On a Space, DA3 and the maskers are built at startup so ZeroGPU only moves them per call
(built inside a call they reloaded every time).
"""
import sys
import types

from streetview_to_3d.tools.on_gpu import ON_SPACES

if ON_SPACES:
    # depth_anything_3 imports pycolmap (unused), whose CUDA init segfaults outside a GPU call
    sys.modules.setdefault("pycolmap", types.ModuleType("pycolmap"))
    from streetview_to_3d.build_scene.models.DepthModel import DepthModel
    from streetview_to_3d.build_scene.models.Segmenter import Segmenter
    DepthModel().loaded()
    Segmenter().segment([])  # the maskers, built at startup like DA3
