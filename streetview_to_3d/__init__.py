"""Street View panoramas along a street, reconstructed into one
placed 3D point cloud.

What callers use: panos prepares a route's corridor, and main.streetview_to_3d builds it
into a scene folder; load reads a saved scene back; build_viewer makes the viewer that opens
one; on_gpu is the one way to a GPU, for an app's own GPU work; date_recency_key and
cap_bucket_for_date rank Street View dates as the corridor's date graphs do.
"""
# First, so `spaces` is imported before anything initialises CUDA.
from streetview_to_3d.tools.on_gpu import on_gpu

from streetview_to_3d.load import load
from streetview_to_3d.panos.main import panos
from streetview_to_3d.panos.tools.dates import cap_bucket_for_date, date_recency_key
from streetview_to_3d.visualise.build_viewer import build_viewer

__all__ = ["on_gpu", "panos", "load", "build_viewer", "date_recency_key", "cap_bucket_for_date"]
