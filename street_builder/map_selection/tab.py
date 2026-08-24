"""Gradio wiring for the map-picking section: load an area, optionally
auto-expand it, then click markers to extend a graph of real Google
Street View nodes. Exposes build_map_section() for street_builder/tab.py
to mount, plus nodes_by_key() (shared state-shape helper) for its pathfind
handlers to use.

The map lives in a sandboxed iframe (see map_ui.py) so a marker click can't
call back into Python directly. The bridge: the iframe does
window.parent.postMessage(...); a listener injected into the page's <head>
(BRIDGE_HEAD_SCRIPT, wired via gr.Blocks(head=...) in app.py) catches that
message and writes it into a hidden Gradio Textbox by simulating a DOM
input event, which triggers this module's Python handler via .change().

This DOM-event bridge is the one part of this feature I can't verify
end-to-end myself (browser testing is intentionally not something I do
autonomously here) — it needs to be tried in an actual browser.
"""
import json

import gradio as gr

from services.geo import extract_lat_lon
from services.streetview_fetch import fetch_pano_by_id, run_async
from street_builder.map_selection import candidates as candidates_mod
from street_builder.map_selection import map_ui

BRIDGE_ELEM_ID = "street_builder_bridge"

# CSS-hidden rather than Gradio's own visible=False: some frontends don't
# render conditionally-hidden components into the DOM at all, which would
# make the textarea unfindable regardless of whether the bridge script runs.
# Keeping it in the DOM (just visually collapsed) removes that uncertainty.
BRIDGE_CSS = f"#{BRIDGE_ELEM_ID} {{ position: fixed !important; width: 1px !important; height: 1px !important; opacity: 0 !important; pointer-events: none !important; overflow: hidden !important; }}"

BRIDGE_HEAD_SCRIPT = f"""
<script>
console.log('[street_builder] bridge listener registered');
window.addEventListener('message', function(ev) {{
  if (!ev.data || ev.data.type !== {json.dumps(map_ui._MESSAGE_TYPE)}) return;
  console.log('[street_builder] message received from map iframe:', ev.data);
  var el = document.querySelector('#{BRIDGE_ELEM_ID} textarea, #{BRIDGE_ELEM_ID} input');
  if (!el) {{
    console.error('[street_builder] bridge element #{BRIDGE_ELEM_ID} not found in DOM');
    return;
  }}
  el.value = JSON.stringify(ev.data);
  el.dispatchEvent(new Event('input', {{bubbles: true}}));
  el.dispatchEvent(new Event('change', {{bubbles: true}}));
  console.log('[street_builder] dispatched input+change on bridge element, value:', el.value);
}});
</script>
"""


def _empty_state():
    return {"lat": None, "lon": None, "nodes": [], "edges": [], "selected": [], "selected_edges": [], "view": None,
            "radius_m": None, "preview_center": None}


def nodes_by_key(state):
    return {n["key"]: n for n in state["nodes"]}


def _summary_markdown(state):
    if not state["selected"]:
        return "_Load an area to set the start node, then click markers to extend the graph._"
    by_key = nodes_by_key(state)
    n_nodes = len(state["selected"])
    n_edges = len(state.get("selected_edges", []))
    lines = [f"**{n_nodes} node(s), {n_edges} edge(s) selected**", ""]
    for i, key in enumerate(state["selected"]):
        n = by_key.get(key)
        if not n:
            continue
        label = "Start" if i == 0 else f"#{i + 1}"
        lines.append(f"{label}. `{n['id']}`")
    return "\n".join(lines)


def _map_html(state, zoom=19):
    radius_m = state.get("radius_m")
    if state["lat"] is None:
        preview = state.get("preview_center")
        if preview:
            return map_ui.build_picker_map(preview[0], preview[1], [], [], [], [], zoom=zoom, radius_m=radius_m)
        return map_ui.build_picker_map(0, 0, [], [], [], [], zoom=2)
    return map_ui.build_picker_map(
        state["lat"], state["lon"], state["nodes"], state["edges"],
        state["selected"], state.get("selected_edges", []),
        zoom=zoom, view=state.get("view"), radius_m=radius_m,
    )


def _augment_real_links(state, key):
    """Fetch this node's own real links directly (Street View's per-pano
    metadata, same as fetch_pano_by_id uses elsewhere) and merge any new
    nodes/edges into state.

    Why this is needed even though nodes/edges already came from
    candidates.nearby_nodes(): that fetch sources positions from Street
    View's TILE coverage listing, which is a different endpoint than
    per-pano links and can genuinely omit a pano that a real link points
    to -- confirmed directly (a specific node's linked neighbor was simply
    absent from the tile listing even when queried centered right on that
    node, not just a radius/max_nodes cutoff issue). nearby_nodes also only
    ever runs once, centered on the original "Load area" point, so a click
    far from that point can be missing edges just from being out of range.
    This fixes both: always goes straight to the accurate per-node source
    for whichever node was actually clicked, regardless of how far it is
    from the original load point or what the bulk tile listing happened to
    include."""
    if not key.startswith("google:"):
        return state  # only Google exposes real link data; Apple has none
    pano_id = key.split(":", 1)[1]
    try:
        meta = run_async(fetch_pano_by_id(pano_id))
    except Exception as e:
        print(f"Link fetch failed for {pano_id}: {e}")
        return state
    if not meta:
        return state

    nodes = list(state["nodes"])
    edges = list(state["edges"])
    by_key = {n["key"]: n for n in nodes}
    edge_set = {frozenset(e) for e in edges}

    for n in meta["neighbors"]:
        other_key = candidates_mod.node_key("google", n["id"])
        if other_key not in by_key:
            new_node = {
                "key": other_key, "source": "google", "id": n["id"],
                "lat": n["lat"], "lon": n["lon"], "heading": None,
            }
            nodes.append(new_node)
            by_key[other_key] = new_node
        fe = frozenset((key, other_key))
        if fe not in edge_set:
            edges.append((key, other_key))
            edge_set.add(fe)

    return {**state, "nodes": nodes, "edges": edges}


def handle_load_area(area_input, state):
    try:
        lat, lon = extract_lat_lon(area_input)
    except ValueError as e:
        raise gr.Error(str(e))

    nodes, edges = candidates_mod.nearby_nodes(lat, lon)
    if not nodes:
        raise gr.Error("No Street View coverage found near that location.")

    # nodes is already distance-sorted, so the nearest one to the input
    # coordinate is the graph's fixed start node.
    start_key = nodes[0]["key"]
    state = {
        "lat": lat, "lon": lon, "nodes": nodes, "edges": edges,
        "selected": [start_key], "selected_edges": [], "view": None,
    }
    state = _augment_real_links(state, start_key)
    return _map_html(state), _summary_markdown(state), state


def handle_expand_area(area_input, radius_input, state, progress=gr.Progress(track_tqdm=True)):
    """Experimental: auto-discover the real Street View graph within a
    radius of the given area (see candidates.expand_area) instead of
    clicking node by node -- sets the WHOLE discovered graph as already
    selected, ready to press "Prepare auto-path" directly. Reuses the
    exact same geocoding handle_load_area uses for the center point."""
    try:
        lat, lon = extract_lat_lon(area_input)
    except ValueError as e:
        raise gr.Error(str(e))
    try:
        radius_m = float(radius_input)
    except (TypeError, ValueError):
        raise gr.Error("Enter a valid radius in meters.")
    if radius_m <= 0:
        raise gr.Error("Radius must be positive.")

    progress(0, desc="Auto-expanding area...")
    nodes, edges = candidates_mod.expand_area(lat, lon, radius_m)
    if not nodes:
        raise gr.Error("No Street View coverage found in that area.")

    state = {
        "lat": lat, "lon": lon, "nodes": nodes, "edges": edges,
        "selected": [n["key"] for n in nodes], "selected_edges": list(edges), "view": None,
        "radius_m": radius_m, "preview_center": None,
    }
    progress(1.0, desc="Done!")
    return _map_html(state), _summary_markdown(state), state


def handle_preview_radius(area_input, radius_input, state):
    """Draws the blue radius circle live as the radius (or location) is
    typed, without running the actual (network-heavy) expand_area walk --
    so the radius can be sanity-checked visually before committing to it.
    Only ever touches radius_m/preview_center, never nodes/edges/selected,
    so it's always safe to fire on every keystroke without disturbing an
    already-loaded graph."""
    try:
        lat, lon = extract_lat_lon(area_input)
    except ValueError:
        lat, lon = state.get("lat"), state.get("lon")
    if lat is None:
        return _map_html(state), state

    try:
        radius_m = float(radius_input)
        if radius_m <= 0:
            radius_m = None
    except (TypeError, ValueError):
        radius_m = None

    state = {**state, "radius_m": radius_m, "preview_center": (lat, lon)}
    return _map_html(state), state


def handle_bridge_message(payload_str, state):
    # Printed server-side (visible in the terminal running `python app.py`,
    # not the browser console) -- confirms whether Gradio's .change() ever
    # actually fires, independent of anything happening in the browser.
    print(f"[street_builder] handle_bridge_message called, payload={payload_str!r}")

    if not payload_str or state.get("lat") is None:
        return _map_html(state), _summary_markdown(state), state, ""

    try:
        payload = json.loads(payload_str)
    except (TypeError, ValueError):
        return _map_html(state), _summary_markdown(state), state, ""

    key = payload.get("key")
    if not key:
        return _map_html(state), _summary_markdown(state), state, ""

    # Always refresh this node's real links before validating the click --
    # see _augment_real_links for why nodes/edges from the initial bulk
    # fetch alone aren't reliable enough to gate frontier expansion on.
    state = _augment_real_links(state, key)

    # Graph selection, fixed start: a click only ever adds a node/edge that's
    # a REAL edge (Street View's own pano.links, sourced in candidates.py) to
    # something already selected -- never guessed from click proximity. This
    # is what lets a branch (two clicks off the same node) or a loop-closing
    # click (a "next" node that happens to already be selected via a
    # different branch) just fall out naturally, instead of needing special
    # handling: any real edge between the clicked node and an already-
    # selected node gets recorded, whether or not the node itself is new.
    selected = list(state["selected"])
    selected_set = set(selected)
    selected_edges = list(state.get("selected_edges", []))
    confirmed = {frozenset(e) for e in selected_edges}
    edge_set = {frozenset(e) for e in state["edges"]}

    is_new = key not in selected_set
    if is_new:
        has_real_link = any(frozenset((s, key)) in edge_set for s in selected_set)
        if selected_set and not has_real_link:
            # Not a real neighbor of anything selected -- ignore. The map only
            # ever shows real frontier nodes as clickable, so this shouldn't
            # normally happen; guards against a stale/late click after a
            # rebuild changed what's selected.
            return _map_html(state), _summary_markdown(state), state, ""
        selected.append(key)
        selected_set.add(key)

    for s in selected_set:
        if s == key:
            continue
        fe = frozenset((s, key))
        if fe in edge_set and fe not in confirmed:
            selected_edges.append((s, key))
            confirmed.add(fe)

    # Carry through whatever pan/zoom the map was at when clicked, so the
    # rebuilt iframe reopens there instead of snapping back to the load center.
    view = payload.get("view")
    new_view = (view["lat"], view["lon"], view["zoom"]) if view else state.get("view")

    state = {**state, "selected": selected, "selected_edges": selected_edges, "view": new_view}
    return _map_html(state), _summary_markdown(state), state, ""


def handle_clear(state):
    # "Clear" resets the graph back to just the fixed start node, not to
    # nothing -- the start node comes from the load-area input, not a click.
    start = [state["nodes"][0]["key"]] if state.get("nodes") else []
    state = {**state, "selected": start, "selected_edges": []}
    return _map_html(state), _summary_markdown(state), state


def build_map_section():
    """Builds the load/expand/click-picker UI and wires its own handlers.
    Returns (state, map_view, selection_view) -- street_builder/tab.py's
    build_tab() reads `state` as input for its own (pathfind) handlers,
    and mounts its own controls below map_view/selection_view."""
    state = gr.State(_empty_state())

    with gr.Row(equal_height=True):
        area_input = gr.Textbox(
            placeholder="Google Maps URL or lat,lon (e.g. 1.3237, 103.7555)",
            show_label=False,
            container=False,
            scale=5,
        )
        load_btn = gr.Button("Load area", variant="primary", scale=1, min_width=100)

    with gr.Row(equal_height=True):
        expand_radius_input = gr.Textbox(
            placeholder="Auto-expand radius in meters (e.g. 500) -- uses the same location above",
            show_label=False,
            container=False,
            scale=5,
        )
        expand_btn = gr.Button("Auto-expand area (experimental)", scale=1, min_width=100)

    map_view = gr.HTML(_map_html(_empty_state()), elem_classes="no-pad")
    # visible=True + CSS hiding (BRIDGE_CSS), not visible=False -- see the
    # comment above BRIDGE_CSS for why.
    bridge = gr.Textbox(elem_id=BRIDGE_ELEM_ID, show_label=False, container=False)

    with gr.Row(equal_height=True):
        selection_view = gr.Markdown(_summary_markdown(_empty_state()), elem_id="street_builder_selection")
        with gr.Column(scale=0, min_width=140):
            clear_btn = gr.Button("Clear selection")

    load_btn.click(
        fn=handle_load_area,
        inputs=[area_input, state],
        outputs=[map_view, selection_view, state],
    )

    expand_btn.click(
        fn=handle_expand_area,
        inputs=[area_input, expand_radius_input, state],
        outputs=[map_view, selection_view, state],
        show_progress="minimal",
        show_progress_on=[selection_view],
    )

    expand_radius_input.change(
        fn=handle_preview_radius,
        inputs=[area_input, expand_radius_input, state],
        outputs=[map_view, state],
    )
    area_input.change(
        fn=handle_preview_radius,
        inputs=[area_input, expand_radius_input, state],
        outputs=[map_view, state],
    )

    bridge.change(
        fn=handle_bridge_message,
        inputs=[bridge, state],
        outputs=[map_view, selection_view, state, bridge],
    )

    clear_btn.click(
        fn=handle_clear,
        inputs=[state],
        outputs=[map_view, selection_view, state],
    )

    return state, map_view, selection_view
