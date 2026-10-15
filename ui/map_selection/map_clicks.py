"""Grow the selection from clicks and drags on the map.

The map is a sandboxed iframe, so a click is relayed: the iframe posts a message, a listener
in the page head (map_widgets.BRIDGE_HEAD_SCRIPT) writes it into a hidden textbox, and its
.change() runs two chains: one for clicks on a node, one for drags of the area's edges.
"""
import json
from functools import partial, update_wrapper

import gradio as gr

from ui.map_selection.Drag import Drag
from ui.map_selection.MapWidgets import MapWidgets
from ui.map_selection.tools.map_view import AREA_MESSAGE_TYPE, map_page
from ui.map_selection.tools.real_links import with_real_links


def _message(payload_str, state):
    """The relayed message, once an area is loaded; anything else stops here, quietly."""
    try:
        payload = json.loads(payload_str) if payload_str and state.get("lat") is not None else None
    except (TypeError, ValueError):
        payload = None
    if not payload:
        raise gr.Error("Nothing to do.", visible=False, print_exception=False)
    return payload


def _clicked(payload_str, state):
    """The clicked node's key and the map's view, if this message is a click."""
    payload = _message(payload_str, state)
    if payload.get("type") == AREA_MESSAGE_TYPE or not payload.get("key"):
        raise gr.Error("Not a click.", visible=False, print_exception=False)
    return payload["key"], payload.get("view")


def _dragged(payload_str, state):
    """The Drag, if this message is the area's edges dragged into a shape."""
    payload = _message(payload_str, state)
    area = payload.get("area") or []
    if payload.get("type") != AREA_MESSAGE_TYPE or len(area) < 3:
        raise gr.Error("Not a drag.", visible=False, print_exception=False)
    return Drag(area, payload.get("view"), (state["lat"], state["lon"]))


def _follow_click(click, state):
    """A click adds only real links to the selection, never guessed from proximity; every
    real link to an already selected node is recorded, so branches and loops just work."""
    key, view = click
    # refresh the clicked node's real links before validating the click
    state = with_real_links(state, key)
    selected = list(state["selected"])
    selected_set = set(selected)
    selected_edges = list(state.get("selected_edges", []))
    confirmed = {frozenset(e) for e in selected_edges}
    edge_set = {frozenset(e) for e in state["edges"]}

    if key not in selected_set:
        if selected_set and not any(frozenset((s, key)) in edge_set for s in selected_set):
            # not linked to the selection: a stale click after a rebuild
            return map_page(state)
        selected.append(key)
        selected_set.add(key)

    for s in selected_set:
        if s == key:
            continue
        fe = frozenset((s, key))
        if fe in edge_set and fe not in confirmed:
            selected_edges.append((s, key))
            confirmed.add(fe)

    # reopen the rebuilt map at the pan/zoom it was clicked at
    new_view = (view["lat"], view["lon"], view["zoom"]) if view else state.get("view")
    return map_page({**state, "selected": selected, "selected_edges": selected_edges, "view": new_view})


def map_clicks(widgets: MapWidgets, job, expand_area):
    """Wire the map's relayed clicks and drags: a click extends the selection; a drag runs
    job (redraw_area), with expand_area walking the dragged area."""
    shown = [widgets.map_view, widgets.selection_view, widgets.state]
    click = gr.State(None)
    widgets.bridge.change(
        fn=_clicked, inputs=[widgets.bridge, widgets.state], outputs=[click], api_visibility="private",
    ).success(
        fn=_follow_click, inputs=[click, widgets.state], outputs=shown, api_name="handle_bridge_message",
    )
    drag = gr.State(None)
    widgets.bridge.change(
        fn=_dragged, inputs=[widgets.bridge, widgets.state], outputs=[drag], api_visibility="private",
    ).success(
        fn=update_wrapper(partial(job, expand_area=expand_area), job), inputs=[drag, widgets.state], outputs=shown,
        api_visibility="private",
    )
