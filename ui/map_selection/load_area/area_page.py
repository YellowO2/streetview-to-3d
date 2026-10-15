"""Show a freshly loaded area, its nearest pano the start of the selection."""
import gradio as gr

from ui.map_selection.tools.map_view import map_page
from ui.map_selection.tools.real_links import with_real_links
from ui.map_selection.tools.typed_radius import typed_radius


def area_page(place, found, radius_input):
    """The map page for the nodes and links found round place; the nearest node (found
    sorted by distance) starts the selection, with its own real links added."""
    nodes, edges = found
    if not nodes:
        raise gr.Error("No Street View coverage found near that location.")
    lat, lon = place
    start_key = nodes[0]["key"]
    state = {
        "lat": lat, "lon": lon, "nodes": nodes, "edges": edges,
        "selected": [start_key], "selected_edges": [], "view": None,
        "radius_m": typed_radius(radius_input), "preview_center": None, "area": None,
    }
    return map_page(with_real_links(state, start_key))
