"""Street Builder, the Gradio app on streetview_to_3d: pick an area and a route on a map
(map_selection/), prepare and build it (prepare_button.py, reconstruct_button.py) and look
at the result. What an app mounts: the page (main.street_builder), the API-only pano
endpoints (pano_api) and the map's bridge, which goes in the page head.

Each button calls into streetview_to_3d, the engine, as any library.
"""
import streetview_to_3d  # noqa: F401  first: it imports `spaces` before anything touches CUDA
from street_builder.main import street_builder
from street_builder.map_selection.map_section import BRIDGE_CSS, BRIDGE_HEAD_SCRIPT
from street_builder.pano_api import pano_api

__all__ = ["street_builder", "pano_api", "BRIDGE_CSS", "BRIDGE_HEAD_SCRIPT"]
