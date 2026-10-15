"""Show everything found inside the circle, all selected."""
import gradio as gr
import numpy as np

from ui.map_selection.tools.map_view import map_page
from streetview_to_3d.tools.geo import local_m_to_latlon


def selection_page(circle, found):
    """The map page for the nodes and links found inside circle, every one selected; the
    circle becomes an area whose edges drag (map_clicks.py)."""
    nodes, edges = found
    if not nodes:
        raise gr.Error("No Street View coverage found in that area.")
    state = {
        "lat": circle.lat, "lon": circle.lon, "nodes": nodes, "edges": edges,
        "selected": [n["key"] for n in nodes], "selected_edges": list(edges), "view": None,
        "radius_m": None, "preview_center": None,
        "area": _circle(circle.lat, circle.lon, circle.radius_m),
    }
    return map_page(state)


def _circle(lat, lon, radius_m, corners=8):
    """An area as a polygon: a few (lat, lon) corners on the circle radius_m round (lat, lon),
    so dragging one moves a good part of its edge."""
    a = np.linspace(0, 2 * np.pi, corners, endpoint=False)
    return [list(local_m_to_latlon(radius_m * np.cos(t), radius_m * np.sin(t), lat, lon)) for t in a]
