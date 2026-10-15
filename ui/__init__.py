"""Street Builder's page: pick an area and a route on a map (map_selection/), prepare and
build it (prepare_button.py, reconstruct_button.py) and look at the result. What an app
mounts: the page (main.ui, handed the engine's prepare and build), the API-only pano
endpoints (pano_api) and the map's bridge, which goes in the page head.

Each button calls what it is handed: the engine's jobs, wired in the project's main.py.
"""
import streetview_to_3d  # noqa: F401  first: it imports `spaces` before anything touches CUDA
from ui.main import ui
from ui.map_selection.map_section import BRIDGE_CSS, BRIDGE_HEAD_SCRIPT
from ui.pano_api import pano_api

__all__ = ["ui", "pano_api", "BRIDGE_CSS", "BRIDGE_HEAD_SCRIPT"]
