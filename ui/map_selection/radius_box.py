"""Draw the radius circle on the map as it is typed."""
from ui.map_selection.MapWidgets import MapWidgets
from ui.map_selection.tools.map_view import map_html
from ui.map_selection.tools.typed_radius import typed_radius
from streetview_to_3d.tools.geo import extract_lat_lon


def _handle_preview_radius(area_input, radius_input, state):
    """Draw the radius circle as it is typed, without expanding; touches only
    radius_m/preview_center, so it is safe on every keystroke."""
    try:
        lat, lon = extract_lat_lon(area_input)
    except ValueError:
        lat, lon = state.get("lat"), state.get("lon")
    if lat is None:
        return map_html(state), state

    state = {**state, "radius_m": typed_radius(radius_input), "preview_center": (lat, lon)}
    return map_html(state), state


def radius_box(widgets: MapWidgets):
    """Wire the radius box to the map's circle."""
    widgets.radius_input.change(
        fn=_handle_preview_radius,
        inputs=[widgets.area_input, widgets.radius_input, widgets.state],
        outputs=[widgets.map_view, widgets.state],
        api_name="handle_preview_radius",
    )
