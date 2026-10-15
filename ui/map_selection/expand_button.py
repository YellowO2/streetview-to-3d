"""Select every Street View graph within the typed radius, on Expand."""
from functools import partial, update_wrapper

import gradio as gr

from ui.map_selection.Circle import Circle
from ui.map_selection.MapWidgets import MapWidgets
from streetview_to_3d.tools.geo import extract_lat_lon


def _circle_typed(area_input, radius_input):
    """The Circle typed into the boxes; a bad place or radius stops here, saying why."""
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
    return Circle(lat, lon, radius_m)


def expand_button(widgets: MapWidgets, job, expand):
    """Wire the Expand button: job (expand_selection) selects everything expand (expand_area)
    walks inside the typed circle."""
    circle = gr.State(None)
    widgets.expand.click(
        fn=_circle_typed, inputs=[widgets.area_input, widgets.radius_input], outputs=[circle],
        api_visibility="private",
    ).success(
        fn=update_wrapper(partial(job, expand=expand), job),
        inputs=[circle],
        outputs=[widgets.map_view, widgets.selection_view, widgets.state],
        api_name="handle_expand_area",
        show_progress="minimal",
        show_progress_on=[widgets.selection_view],
    )
