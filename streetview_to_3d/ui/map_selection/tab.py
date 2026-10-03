"""The map-picking section: a pasted location loads its area, Expand selects everything in a
radius, clicks extend the selection along real Street View links.

The map is a sandboxed iframe, so a click is relayed: the iframe posts a message, a listener
in the page head (BRIDGE_HEAD_SCRIPT) writes it into a hidden textbox, and its .change() runs
handle_bridge_message.
"""
import json

import gradio as gr

from streetview_to_3d.common.geo import extract_lat_lon
from streetview_to_3d.panos.fetch_nodes import corridor_points
from streetview_to_3d.reconstruct.runner import estimate_gpu_seconds, estimate_other_seconds
from streetview_to_3d.common.streetview_fetch import fetch_pano_by_id, google_node, run_async
from streetview_to_3d.panos import candidates as candidates_mod
from streetview_to_3d.ui.map_selection import map_ui

BRIDGE_ELEM_ID = "map_bridge"

# hidden by CSS, not visible=False, which can leave it out of the DOM
BRIDGE_CSS = f"#{BRIDGE_ELEM_ID} {{ position: fixed !important; width: 1px !important; height: 1px !important; opacity: 0 !important; pointer-events: none !important; overflow: hidden !important; }}"

BRIDGE_HEAD_SCRIPT = f"""
<script>
window.addEventListener('message', function(ev) {{
  if (!ev.data || {json.dumps(list(map_ui.MESSAGE_TYPES))}.indexOf(ev.data.type) === -1) return;
  var el = document.querySelector('#{BRIDGE_ELEM_ID} textarea, #{BRIDGE_ELEM_ID} input');
  if (!el) {{
    console.error('[map] bridge element #{BRIDGE_ELEM_ID} not found in DOM');
    return;
  }}
  el.value = JSON.stringify(ev.data);
  el.dispatchEvent(new Event('input', {{bubbles: true}}));
  el.dispatchEvent(new Event('change', {{bubbles: true}}));
}});
</script>
"""


def _empty_state():
    return {"lat": None, "lon": None, "nodes": [], "edges": [], "selected": [], "selected_edges": [], "view": None,
            "radius_m": None, "preview_center": None, "area": None}


def nodes_by_key(state):
    return {n["key"]: n for n in state["nodes"]}


def corridor_edges(state):
    """The selected edges as ((lat, lon, pano_id), (lat, lon, pano_id))
    pairs -- what fetch_nodes.corridor_points builds the spots from."""
    by_key = nodes_by_key(state)
    node = lambda k: (by_key[k]["lat"], by_key[k]["lon"], by_key[k]["id"])
    return [(node(a), node(b)) for a, b in state.get("selected_edges", []) if a in by_key and b in by_key]


def _summary_markdown(state):
    if not state["selected"]:
        return "Enter a location and select a street via 'Expand Area' Button or manually clicking. Then press button 1, wait for it to run, then button 2."
    # the GPU is spent per spot -- panos within a few metres merge into one
    # (fetch_nodes.corridor_points) -- not per pano
    n_spots = len(corridor_points(corridor_edges(state))[0])
    gpu, quick = (estimate_gpu_seconds(n_spots, e) / 60 for e in ("Normal", "Quick"))
    other = max(1, round(estimate_other_seconds(n_spots) / 60))
    return (f"**{n_spots} spots selected**  \n"
            f"This takes ~{gpu:.1f} min of GPU. HF offers 5 min/day for a free account. "
            f"You may switch to Quick ({quick:.1f} min) to save usage.  \n"
            f"It will also take around ~{other} mins for postprocessing (aligning, fetching more map, etc), "
            f"which will not cost GPU.")


def _map_html(state, zoom=19):
    radius_m = state.get("radius_m")
    if state["lat"] is None:
        preview = state.get("preview_center")
        if preview:
            return map_ui.build_picker_map(preview[0], preview[1], [], [], [], [], zoom=zoom, radius_m=radius_m)
        return map_ui.build_picker_map(0, 0, [], [], [], [], zoom=2)
    points, adjacency, _ = corridor_points(corridor_edges(state))
    return map_ui.build_picker_map(
        state["lat"], state["lon"], state["nodes"], state["edges"],
        state["selected"], state.get("selected_edges", []),
        zoom=zoom, view=state.get("view"), radius_m=radius_m, spots=(points, adjacency),
        area=state.get("area"),
    )


def _augment_real_links(state, key):
    """Merge this node's own per-pano links into state: the tile listing nearby_nodes uses
    can omit a linked pano, and only covers the area first loaded."""
    if not key.startswith("google:"):
        return state  # only Google panos have real link data
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
        new_node = google_node(n["id"], n["lat"], n["lon"])
        other_key = new_node["key"]
        if other_key not in by_key:
            nodes.append(new_node)
            by_key[other_key] = new_node
        fe = frozenset((key, other_key))
        if fe not in edge_set:
            edges.append((key, other_key))
            edge_set.add(fe)

    return {**state, "nodes": nodes, "edges": edges}


def handle_load_area(area_input, radius_input, state):
    """A location pasted (or typed): its area loads, the nearest pano the
    start -- no button. Text not yet a location (half typed) is left alone."""
    try:
        lat, lon = extract_lat_lon(area_input)
    except ValueError:
        return gr.skip(), gr.skip(), state

    nodes, edges = candidates_mod.nearby_nodes(lat, lon)
    if not nodes:
        raise gr.Error("No Street View coverage found near that location.")

    # nodes is already distance-sorted, so the nearest one to the input
    # coordinate is the graph's fixed start node.
    start_key = nodes[0]["key"]
    state = {
        "lat": lat, "lon": lon, "nodes": nodes, "edges": edges,
        "selected": [start_key], "selected_edges": [], "view": None,
        "radius_m": _radius(radius_input), "preview_center": None, "area": None,
    }
    state = _augment_real_links(state, start_key)
    return _map_html(state), _summary_markdown(state), state


def handle_expand_area(area_input, radius_input, state, progress=gr.Progress(track_tqdm=False)):
    """Select the whole real Street View graph within a radius (candidates.expand_area),
    ready for "Prepare"."""
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

    progress(None, desc="Finding nearby panoramas…")
    nodes, edges = candidates_mod.expand_area(lat, lon, radius_m)
    if not nodes:
        raise gr.Error("No Street View coverage found in that area.")

    state = {
        "lat": lat, "lon": lon, "nodes": nodes, "edges": edges,
        "selected": [n["key"] for n in nodes], "selected_edges": list(edges), "view": None,
        "radius_m": None, "preview_center": None,
        # its circle as a shape whose edges drag (handle_area_drag)
        "area": candidates_mod.circle(lat, lon, radius_m),
    }
    return _map_html(state), _summary_markdown(state), state


def handle_area_drag(payload, state):
    """The expanded area's edges dragged: everything inside the new shape,
    walked again as expand_area walks -- from what it has already looked
    up, so pulling an edge in fetches nothing and pushing it out only what
    is new. The whole of it selected, as after expanding."""
    area = payload.get("area") or []
    if len(area) < 3 or state.get("lat") is None:
        return state
    nodes, edges = candidates_mod.expand_area(state["lat"], state["lon"], area=area)
    view = payload.get("view")
    return {**state, "nodes": nodes, "edges": edges, "selected": [n["key"] for n in nodes],
            "selected_edges": list(edges), "area": area, "radius_m": None,
            "view": (view["lat"], view["lon"], view["zoom"]) if view else state.get("view")}


def handle_preview_radius(area_input, radius_input, state):
    """Draw the radius circle as it is typed, without expanding; touches only
    radius_m/preview_center, so it is safe on every keystroke."""
    try:
        lat, lon = extract_lat_lon(area_input)
    except ValueError:
        lat, lon = state.get("lat"), state.get("lon")
    if lat is None:
        return _map_html(state), state

    state = {**state, "radius_m": _radius(radius_input), "preview_center": (lat, lon)}
    return _map_html(state), state


def _radius(radius_input):
    """The radius typed, in metres; None if it is not one yet."""
    try:
        radius_m = float(radius_input)
    except (TypeError, ValueError):
        return None
    return radius_m if radius_m > 0 else None


def handle_bridge_message(payload_str, state):
    """A map click or area drag, relayed from the iframe (see BRIDGE_HEAD_SCRIPT)."""
    if not payload_str or state.get("lat") is None:
        return _map_html(state), _summary_markdown(state), state, ""

    try:
        payload = json.loads(payload_str)
    except (TypeError, ValueError):
        return _map_html(state), _summary_markdown(state), state, ""

    if payload.get("type") == map_ui.AREA_MESSAGE_TYPE:
        state = handle_area_drag(payload, state)
        return _map_html(state), _summary_markdown(state), state, ""

    key = payload.get("key")
    if not key:
        return _map_html(state), _summary_markdown(state), state, ""

    # refresh the clicked node's real links before validating the click
    state = _augment_real_links(state, key)

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

    # reopen the rebuilt map at the pan/zoom it was clicked at
    view = payload.get("view")
    new_view = (view["lat"], view["lon"], view["zoom"]) if view else state.get("view")

    state = {**state, "selected": selected, "selected_edges": selected_edges, "view": new_view}
    return _map_html(state), _summary_markdown(state), state, ""


def handle_clear(state):
    # back to just the start node (which comes from the location, not a click)
    start = [state["nodes"][0]["key"]] if state.get("nodes") else []
    state = {**state, "selected": start, "selected_edges": []}
    return _map_html(state), _summary_markdown(state), state


def build_map_section():
    """Build and wire the map section. Returns (state, map_view, selection_view) for
    ui/tab.py's build_main_tab."""
    state = gr.State(_empty_state())

    # one row: the location loads as it is pasted, so the only button is Expand
    with gr.Row(equal_height=True):
        area_input = gr.Textbox(
            placeholder="Google Maps URL or lat,lon (e.g. 1.3237, 103.7555)",
            show_label=False,
            container=False,
            scale=4,
        )
        expand_radius_input = gr.Textbox(
            placeholder="Radius in metres (e.g. 50)",
            show_label=False,
            container=False,
            scale=2,
        )
        expand_btn = gr.Button("Expand area", scale=1, min_width=100)

    map_view = gr.HTML(_map_html(_empty_state()), elem_classes="no-pad")
    # hidden by BRIDGE_CSS
    bridge = gr.Textbox(elem_id=BRIDGE_ELEM_ID, show_label=False, container=False)

    with gr.Row(equal_height=True):
        selection_view = gr.Markdown(_summary_markdown(_empty_state()), elem_id="map_selection")
        with gr.Column(scale=0, min_width=140):
            clear_btn = gr.Button("Clear selection")

    area_input.change(
        fn=handle_load_area,
        inputs=[area_input, expand_radius_input, state],
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
