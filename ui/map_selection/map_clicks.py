"""Grow the selection from clicks and drags on the map.

The map is a sandboxed iframe, so a click is relayed: the iframe posts a message, a listener
in the page head (map_widgets.BRIDGE_HEAD_SCRIPT) writes it into a hidden textbox, and its
.change() runs _handle_bridge_message.
"""
import json
from functools import partial, update_wrapper

from ui.map_selection.MapWidgets import MapWidgets
from ui.map_selection.tools.map_view import AREA_MESSAGE_TYPE, map_html, summary_markdown
from ui.map_selection.tools.real_links import with_real_links


def _handle_area_drag(payload, state, expand):
    """The expanded area's edges dragged: everything inside the new shape,
    walked again as expand_area walks -- from what it has already looked
    up, so pulling an edge in fetches nothing and pushing it out only what
    is new. The whole of it selected, as after expanding."""
    area = payload.get("area") or []
    if len(area) < 3 or state.get("lat") is None:
        return state
    nodes, edges = expand(state["lat"], state["lon"], area=area)
    view = payload.get("view")
    return {**state, "nodes": nodes, "edges": edges, "selected": [n["key"] for n in nodes],
            "selected_edges": list(edges), "area": area, "radius_m": None,
            "view": (view["lat"], view["lon"], view["zoom"]) if view else state.get("view")}


def _handle_bridge_message(payload_str, state, *, expand):
    """A map click or area drag, relayed from the iframe (see BRIDGE_HEAD_SCRIPT)."""
    if not payload_str or state.get("lat") is None:
        return map_html(state), summary_markdown(state), state, ""

    try:
        payload = json.loads(payload_str)
    except (TypeError, ValueError):
        return map_html(state), summary_markdown(state), state, ""

    if payload.get("type") == AREA_MESSAGE_TYPE:
        state = _handle_area_drag(payload, state, expand)
        return map_html(state), summary_markdown(state), state, ""

    key = payload.get("key")
    if not key:
        return map_html(state), summary_markdown(state), state, ""

    # refresh the clicked node's real links before validating the click
    state = with_real_links(state, key)

    # A click adds only real links to the selection, never guessed from proximity; every
    # real link to an already selected node is recorded, so branches and loops just work.
    selected = list(state["selected"])
    selected_set = set(selected)
    selected_edges = list(state.get("selected_edges", []))
    confirmed = {frozenset(e) for e in selected_edges}
    edge_set = {frozenset(e) for e in state["edges"]}

    is_new = key not in selected_set
    if is_new:
        has_real_link = any(frozenset((s, key)) in edge_set for s in selected_set)
        if selected_set and not has_real_link:
            # not linked to the selection: a stale click after a rebuild
            return map_html(state), summary_markdown(state), state, ""
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
    view = payload.get("view")
    new_view = (view["lat"], view["lon"], view["zoom"]) if view else state.get("view")

    state = {**state, "selected": selected, "selected_edges": selected_edges, "view": new_view}
    return map_html(state), summary_markdown(state), state, ""


def map_clicks(widgets: MapWidgets, expand):
    """Wire the map's relayed clicks and drags: expand (expand_area) walks a dragged area."""
    widgets.bridge.change(
        fn=update_wrapper(partial(_handle_bridge_message, expand=expand), _handle_bridge_message),
        inputs=[widgets.bridge, widgets.state],
        outputs=[widgets.map_view, widgets.selection_view, widgets.state, widgets.bridge],
        api_name="handle_bridge_message",
    )
