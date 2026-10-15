"""Select every Street View graph within the typed radius, on Expand."""
from functools import partial, update_wrapper

import gradio as gr
import numpy as np

from ui.map_selection.MapWidgets import MapWidgets
from ui.map_selection.tools.map_view import map_html, summary_markdown
from streetview_to_3d.tools.geo import extract_lat_lon, local_m_to_latlon


def _handle_expand_area(area_input, radius_input, state, progress=gr.Progress(track_tqdm=False), *, expand):
    """Select the whole real Street View graph within a radius (expand_area.py),
    ready for "Prepare"."""
    try:
        lat, lon = extract_lat_lon(area_input)
    except ValueError as e:
        raise gr.Error(str(e))
    try:
        radius_m = float(radius_input)
    except (TypeError, ValueError):
        raise gr.Error("Enter a valid radius in meters.")
    if radius_m <= 0:
        raise gr.Error("Radius must be positive.")

    progress(None, desc="Finding nearby panoramas…")
    nodes, edges = expand(lat, lon, radius_m)
    if not nodes:
        raise gr.Error("No Street View coverage found in that area.")

    state = {
        "lat": lat, "lon": lon, "nodes": nodes, "edges": edges,
        "selected": [n["key"] for n in nodes], "selected_edges": list(edges), "view": None,
        "radius_m": None, "preview_center": None,
        # its circle as a shape whose edges drag (map_clicks.py)
        "area": _circle(lat, lon, radius_m),
    }
    return map_html(state), summary_markdown(state), state


def _circle(lat, lon, radius_m, corners=8):
    """An area as a polygon: a few (lat, lon) corners on the circle radius_m round (lat, lon),
    so dragging one moves a good part of its edge."""
    a = np.linspace(0, 2 * np.pi, corners, endpoint=False)
    return [list(local_m_to_latlon(radius_m * np.cos(t), radius_m * np.sin(t), lat, lon)) for t in a]


def expand_button(widgets: MapWidgets, expand):
    """Wire the Expand button: expand (expand_area) walks the graphs inside the radius."""
    widgets.expand.click(
        fn=update_wrapper(partial(_handle_expand_area, expand=expand), _handle_expand_area),
        inputs=[widgets.area_input, widgets.radius_input, widgets.state],
        outputs=[widgets.map_view, widgets.selection_view, widgets.state],
        api_name="handle_expand_area",
        show_progress="minimal",
        show_progress_on=[widgets.selection_view],
    )
