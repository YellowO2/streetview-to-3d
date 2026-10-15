"""Build a prepared corridor into a finished 3D scene: reconstruct (GPU), then postprocess
and save (CPU); main.build_scene runs them, and is what this machine offers."""
from streetview_to_3d.build_scene.main import build_scene

__all__ = ["build_scene"]
