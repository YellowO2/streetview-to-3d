"""Lay out the page: pick a route on the map, then prepare and build it."""
from ui.map_selection.main import map_selection
from ui.on_prepare.main import on_prepare
from ui.on_reconstruct.main import on_reconstruct
from ui.page_controls import page_controls
from ui.pano_api import pano_api
from ui.prepare_button import prepare_button
from ui.reconstruct_button import reconstruct_button


def ui(prepare, build):
    """Build and wire the page, inside a gr.Blocks: the map, then button 1 has on_prepare
    prepare the selected route (prepare) and button 2 has on_reconstruct build and show its
    scene (build); and the API-only endpoints for one pano."""
    map_state = map_selection()
    controls = page_controls()
    corridor = prepare_button(map_state, controls, on_prepare, prepare)
    reconstruct_button(corridor, controls, on_reconstruct, build)
    pano_api()
