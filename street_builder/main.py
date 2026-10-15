"""Lay out the page: pick a route on the map, then prepare and build it."""
from street_builder.map_selection.map_section import map_section
from street_builder.page_controls import page_controls
from street_builder.prepare_button import prepare_button
from street_builder.reconstruct_button import reconstruct_button


def street_builder():
    """Build and wire the page, inside a gr.Blocks: the map, then button 1 prepares the
    selected route and button 2 builds and shows its scene."""
    map_state = map_section()
    controls = page_controls()
    corridor = prepare_button(map_state, controls)
    reconstruct_button(corridor, controls)
