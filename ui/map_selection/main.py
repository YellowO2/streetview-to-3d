"""Pick an area and a route on the map.

A pasted location loads its area, Expand selects everything in a radius, clicks extend the
selection along real Street View links.
"""
from ui.map_selection.clear_button import clear_button
from ui.map_selection.expand_area import expand_area
from ui.map_selection.expand_button import expand_button
from ui.map_selection.location_box import location_box
from ui.map_selection.map_clicks import map_clicks
from ui.map_selection.map_widgets import map_widgets
from ui.map_selection.nearby_nodes import nearby_nodes
from ui.map_selection.radius_box import radius_box


def map_selection():
    """Build and wire the map section, inside a gr.Blocks. Returns its gr.State: the
    loaded nodes and the selection."""
    widgets = map_widgets()
    location_box(widgets, nearby_nodes)
    expand_button(widgets, expand_area)
    radius_box(widgets)
    map_clicks(widgets, expand_area)
    clear_button(widgets)
    map_state = widgets.state
    return map_state
