"""Load the area round a location as it is pasted into the map's box."""
from functools import partial, update_wrapper

import gradio as gr

from ui.map_selection.MapWidgets import MapWidgets
from streetview_to_3d.tools.geo import extract_lat_lon


def _where(area_input):
    """The typed place (lat, lon); text not yet a location (half typed) stops here, quietly."""
    try:
        return extract_lat_lon(area_input)
    except ValueError:
        raise gr.Error("Not a location yet.", visible=False, print_exception=False)


def location_box(widgets: MapWidgets, job, nearby_nodes):
    """Wire the location box: as a location is typed, job (load_area) loads its area, with
    nearby_nodes finding the panos and links round it."""
    place = gr.State(None)
    widgets.area_input.change(
        fn=_where, inputs=[widgets.area_input], outputs=[place], api_visibility="private",
    ).success(
        fn=update_wrapper(partial(job, nearby_nodes=nearby_nodes), job),
        inputs=[place, widgets.radius_input],
        outputs=[widgets.map_view, widgets.selection_view, widgets.state],
        api_name="handle_load_area",
    )
