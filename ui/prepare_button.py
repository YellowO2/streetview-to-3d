"""Button 1: check the traced route, then have its panoramas gathered."""
from functools import partial, update_wrapper

import gradio as gr

from ui.Controls import Controls
from ui.Route import Route
from ui.tools.map_state import corridor_edges, nodes_by_key


def _route(state):
    """The Route traced on the map, and the page saying it is being fetched; a selection
    that is not a route stops here, saying why."""
    selected = state.get("selected", [])
    if len(selected) < 2 or not state.get("selected_edges", []):
        raise gr.Error("Select at least 2 connected nodes tracing the route (start to a goal).")
    by_key = nodes_by_key(state)
    start_node = by_key.get(selected[0])
    if not start_node:
        raise gr.Error("Start node not found.")
    goals = [(by_key[k]["lat"], by_key[k]["lon"]) for k in selected[1:] if k in by_key]
    route = Route((start_node["lat"], start_node["lon"]), goals, corridor_edges(state), (state["lat"], state["lon"]))
    return route, None, "<p>Fetching panoramas… This may take a few minutes.</p>"


def _failed():
    return None, "<p>Preparation failed. Try again.</p>"


def prepare_button(map_state, controls: Controls, job, prepare):
    """Wire button 1: the route on the map (map_state) goes to job (on_prepare), which
    has prepare (the engine's gather_panoramas) gather it; returns the gr.State holding the prepared
    Corridor, for button 2."""
    route = gr.State(None)
    corridor = gr.State(None)
    shown = [corridor, controls.status]
    controls.prepare.click(
        fn=_route, inputs=[map_state], outputs=[route, *shown], api_visibility="private",
        show_progress="hidden",
    ).success(
        fn=update_wrapper(partial(job, prepare=prepare), job), inputs=[route], outputs=shown,
        api_name="handle_pathfind_prepare", show_progress="hidden", show_progress_on=[controls.status],
    ).failure(
        fn=_failed, outputs=shown, api_visibility="private", show_progress="hidden",
    )
    return corridor
