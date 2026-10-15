"""Step 1, which panoramas: along the chosen corridor the panos, their dates and links
(fetch_corridor_nodes.py), the dates that cover it best (date_graphs.py), downloaded
(downloaded.py); main.panos runs them, and is what this machine offers. No GPU; no CLI yet, the UI drives it. The map's
candidates are in ui/map_selection/."""
from streetview_to_3d.panos.main import panos

__all__ = ["panos"]
