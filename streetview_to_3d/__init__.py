"""Street View panoramas along a street, reconstructed into one
placed 3D point cloud.

What callers use: main.streetview_to_3d builds a scene from panos.main.panos's corridor
into a folder; load reads a saved scene back; build_viewer makes the viewer that opens
one; on_gpu is the one way to a GPU, for an app's own GPU work.
"""
# First, so `spaces` is imported before anything initialises CUDA.
from streetview_to_3d.tools.on_gpu import on_gpu

from streetview_to_3d.load import load
from streetview_to_3d.visualise.build_viewer import build_viewer

__all__ = ["on_gpu", "load", "build_viewer"]
