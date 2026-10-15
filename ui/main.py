"""Lay out the page: pick a route on the map, then prepare and build it."""
from ui.map_selection.map_section import map_section
from ui.page_controls import page_controls
from ui.prepare_button import prepare_button
from ui.reconstruct_button import reconstruct_button


def ui(prepare, build):
    """Build and wire the page, inside a gr.Blocks: the map, then button 1 prepares the
    selected route (prepare) and button 2 builds and shows its scene (build)."""
    map_state = map_section()
    controls = page_controls()
    corridor = prepare_button(map_state, controls, prepare)
    reconstruct_button(corridor, controls, build)
