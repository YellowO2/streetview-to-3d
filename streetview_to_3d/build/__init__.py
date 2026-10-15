"""Build a prepared corridor into a finished 3D scene: reconstruct (GPU), then postprocess
and save (CPU); main.build runs them, and is what this machine offers."""
from streetview_to_3d.build.main import build

__all__ = ["build"]
