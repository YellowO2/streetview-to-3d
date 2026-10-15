"""Load the area round a location as it is pasted into the map's box."""
from functools import partial, update_wrapper

import gradio as gr

from ui.map_selection.MapWidgets import MapWidgets
from ui.map_selection.tools.map_view import map_html, summary_markdown
from ui.map_selection.tools.real_links import with_real_links
from ui.map_selection.tools.typed_radius import typed_radius
from streetview_to_3d.tools.geo import extract_lat_lon


def _handle_load_area(area_input, radius_input, state, *, find_nodes):
    """A location pasted (or typed): its area loads, the nearest pano the
    start -- no button. Text not yet a location (half typed) is left alone."""
    try:
        lat, lon = extract_lat_lon(area_input)
    except ValueError:
        return gr.skip(), gr.skip(), state

    nodes, edges = find_nodes(lat, lon)
    if not nodes:
        raise gr.Error("No Street View coverage found near that location.")

    # nodes is already distance-sorted, so the nearest one to the input
    # coordinate is the graph's fixed start node.
    start_key = nodes[0]["key"]
    state = {
        "lat": lat, "lon": lon, "nodes": nodes, "edges": edges,
        "selected": [start_key], "selected_edges": [], "view": None,
        "radius_m": typed_radius(radius_input), "preview_center": None, "area": None,
    }
    state = with_real_links(state, start_key)
    return map_html(state), summary_markdown(state), state


def location_box(widgets: MapWidgets, find_nodes):
    """Wire the location box: find_nodes (nearby_nodes) gives the panos and links round it."""
    widgets.area_input.change(
        fn=update_wrapper(partial(_handle_load_area, find_nodes=find_nodes), _handle_load_area),
        inputs=[widgets.area_input, widgets.radius_input, widgets.state],
        outputs=[widgets.map_view, widgets.selection_view, widgets.state],
        api_name="handle_load_area",
    )
