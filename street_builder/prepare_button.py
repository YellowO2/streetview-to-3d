"""Run button 1: gather and download the selected route's panoramas."""
import gradio as gr

from street_builder.Controls import Controls
from street_builder.tools.map_state import corridor_edges, nodes_by_key
from streetview_to_3d.panos.main import panos


def _handle_pathfind_prepare(state):
    """Button 1: gather and download the selection's candidates (no GPU; see panos.main.panos)."""
    selected = state.get("selected", [])
    selected_edges = state.get("selected_edges", [])
    if len(selected) < 2 or not selected_edges:
        raise gr.Error("Select at least 2 connected nodes tracing the route (start to a goal).")

    by_key = nodes_by_key(state)
    start_node = by_key.get(selected[0])
    if not start_node:
        raise gr.Error("Start node not found.")
    start = (start_node["lat"], start_node["lon"])
    goals = [(by_key[k]["lat"], by_key[k]["lon"]) for k in selected[1:] if k in by_key]

    yield None, "<p>Fetching panoramas… This may take a few minutes.</p>"
    try:
        corridor = panos(start, goals, corridor_edges(state), (state["lat"], state["lon"]))
    except Exception as e:
        yield None, "<p>Preparation failed. Try again.</p>"
        raise gr.Error(f"Prepare failed: {e}")

    n = sum(len(c) for g in corridor.date_graphs for c in g["dot_candidates"].values())
    yield corridor, f"<p>{n} panoramas ready. Now click button 2.</p>"


def prepare_button(map_state, controls: Controls):
    """Wire button 1 to the map's state (map_state: gr.State); returns the gr.State holding
    the prepared Corridor, for button 2."""
    corridor = gr.State(None)
    controls.prepare.click(
        fn=_handle_pathfind_prepare,
        inputs=[map_state],
        outputs=[corridor, controls.status],
        api_name="handle_pathfind_prepare",
        show_progress="hidden",
        show_progress_on=[controls.status],
    )
    return corridor
