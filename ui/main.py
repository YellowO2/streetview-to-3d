"""Lay out the page: pick a route on the map, then prepare and build it."""
from ui.build_scene.main import build_scene
from ui.map_selection.main import map_selection
from ui.page_controls import page_controls
from ui.pano_api import pano_api
from ui.prepare_button import prepare_button
from ui.prepare_corridor.main import prepare_corridor
from ui.reconstruct_button import reconstruct_button


def ui(prepare, build):
    """Build and wire the page, inside a gr.Blocks: the map, then button 1 has prepare_corridor
    prepare the selected route (prepare) and button 2 has build_scene build and show its
    scene (build); and the API-only endpoints for one pano."""
    map_state = map_selection()
    controls = page_controls()
    corridor = prepare_button(map_state, controls, prepare_corridor, prepare)
    reconstruct_button(corridor, controls, build_scene, build)
    pano_api()
