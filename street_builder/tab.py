"""Gradio wiring for the street-builder tab: pick an ordered chain of real
Google Street View nodes off a map, then generate a DA3 point cloud from
them. (Apple Look Around is gathered separately, later, as automatic support
imagery per selected node — not something you pick here.)

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
import os
import uuid

import gradio as gr

import viewers
from paths import SPLATS_DIR
from services.geo import extract_lat_lon, haversine_m
from street_builder import candidates as candidates_mod
from street_builder import map_ui
from street_builder import reconstruct

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
    return {"lat": None, "lon": None, "nodes": [], "edges": [], "selected": [], "view": None}


def _nodes_by_key(state):
    return {n["key"]: n for n in state["nodes"]}


def _summary_markdown(state):
    if not state["selected"]:
        return "_Load an area to set the start node, then click markers to extend the chain._"
    by_key = _nodes_by_key(state)
    lines = []
    prev = None
    for i, key in enumerate(state["selected"]):
        n = by_key.get(key)
        if not n:
            continue
        label = "Start" if i == 0 else f"#{i + 1}"
        gap = f" · {haversine_m(prev['lat'], prev['lon'], n['lat'], n['lon']):.0f}m from previous" if prev else ""
        lines.append(f"{label}. `{n['id']}`{gap}")
        prev = n
    return "\n".join(lines)


def _map_html(state, zoom=19):
    if state["lat"] is None:
        return map_ui.build_picker_map(0, 0, [], [], [], zoom=2)
    return map_ui.build_picker_map(
        state["lat"], state["lon"], state["nodes"], state["edges"], state["selected"],
        zoom=zoom, view=state.get("view"),
    )


def handle_load_area(area_input, state):
    try:
        lat, lon = extract_lat_lon(area_input)
    except ValueError as e:
        raise gr.Error(str(e))

    nodes, edges = candidates_mod.nearby_nodes(lat, lon)
    if not nodes:
        raise gr.Error("No Street View coverage found near that location.")

    # nodes is already distance-sorted, so the nearest one to the input
    # coordinate is the chain's fixed start node.
    start_key = nodes[0]["key"]
    state = {"lat": lat, "lon": lon, "nodes": nodes, "edges": edges, "selected": [start_key], "view": None}
    return _map_html(state), _summary_markdown(state), state


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

    # Singly linked chain, fixed start: a click only ever extends the tail
    # (append) or undoes the most recent link (click the current tail again
    # to pop it). Clicking any other already-selected node is a no-op --
    # removing a middle node would break the chain.
    selected = list(state["selected"])
    if selected and key == selected[-1]:
        if len(selected) > 1:  # never pop the fixed start node
            selected.pop()
    elif key not in selected:
        selected.append(key)

    # Carry through whatever pan/zoom the map was at when clicked, so the
    # rebuilt iframe reopens there instead of snapping back to the load center.
    view = payload.get("view")
    new_view = (view["lat"], view["lon"], view["zoom"]) if view else state.get("view")

    state = {**state, "selected": selected, "view": new_view}
    return _map_html(state), _summary_markdown(state), state, ""


def handle_clear(state):
    # "Clear" resets the chain back to just the fixed start node, not to
    # nothing -- the start node comes from the load-area input, not a click.
    start = [state["nodes"][0]["key"]] if state.get("nodes") else []
    state = {**state, "selected": start}
    return _map_html(state), _summary_markdown(state), state


def handle_generate_chain(state, progress=gr.Progress(track_tqdm=True)):
    if len(state.get("selected", [])) < 2:
        raise gr.Error("Select at least 2 nodes (needs multi-view context for DA3).")

    yield viewers.SPLAT_PLACEHOLDER

    by_key = _nodes_by_key(state)
    ordered_nodes = [by_key[k] for k in state["selected"] if k in by_key]

    output_dir = os.path.join(SPLATS_DIR, uuid.uuid4().hex)
    progress(0, desc=f"Reconstructing {len(ordered_nodes)} nodes...")
    try:
        ply_path = reconstruct.reconstruct_chain(ordered_nodes, output_dir)
    except Exception as e:
        raise gr.Error(f"Reconstruction failed: {e}")

    progress(1.0, desc="Done!")
    yield viewers.pointcloud_viewer_with_download(viewers.file_url(ply_path))


def handle_filter_sweep(state, progress=gr.Progress(track_tqdm=True)):
    """Debug button: runs DA3 inference once and produces one point cloud per
    FILTER_SWEEP_LEVELS threshold, to check how much the consensus filter is
    actually doing. Separate from handle_generate_chain -- doesn't touch or
    replace the normal Generate output."""
    if len(state.get("selected", [])) < 2:
        raise gr.Error("Select at least 2 nodes (needs multi-view context for DA3).")

    yield viewers.SPLAT_PLACEHOLDER

    by_key = _nodes_by_key(state)
    ordered_nodes = [by_key[k] for k in state["selected"] if k in by_key]

    output_dir = os.path.join(SPLATS_DIR, uuid.uuid4().hex)
    progress(0, desc=f"Running filter sweep over {len(ordered_nodes)} nodes...")
    try:
        results = reconstruct.reconstruct_chain_filter_sweep(ordered_nodes, output_dir)
    except Exception as e:
        raise gr.Error(f"Filter sweep failed: {e}")

    progress(1.0, desc="Done!")
    yield viewers.filter_sweep_links(results)


def handle_best4(state, progress=gr.Progress(track_tqdm=True)):
    """Experimental button: instead of trusting the chain's own Google nodes,
    scores a wider candidate pool (chain nodes + nearby Apple panos) solo
    through DA3 and reconstructs using only the top BEST4_FINAL_COUNT
    scorers. Separate from handle_generate_chain -- doesn't touch or replace
    the normal Generate output."""
    if len(state.get("selected", [])) < 2:
        raise gr.Error("Select at least 2 nodes (needs multi-view context for DA3).")

    yield viewers.SPLAT_PLACEHOLDER

    by_key = _nodes_by_key(state)
    ordered_nodes = [by_key[k] for k in state["selected"] if k in by_key]

    output_dir = os.path.join(SPLATS_DIR, uuid.uuid4().hex)
    progress(0, desc=f"Scoring candidates near {len(ordered_nodes)} nodes...")
    try:
        ply_path = reconstruct.reconstruct_chain_best4(ordered_nodes, output_dir)
    except Exception as e:
        raise gr.Error(f"Best-4 reconstruction failed: {e}")

    progress(1.0, desc="Done!")
    yield viewers.pointcloud_viewer_with_download(viewers.file_url(ply_path))


def handle_windowed(state, progress=gr.Progress(track_tqdm=True)):
    """Experimental button: chunk+connect for chains longer than one DA3 call
    can handle -- see reconstruct.reconstruct_chain_windowed. Separate from
    handle_generate_chain and handle_best4 -- doesn't touch or replace either."""
    if len(state.get("selected", [])) < 2:
        raise gr.Error("Select at least 2 nodes (needs multi-view context for DA3).")

    yield viewers.SPLAT_PLACEHOLDER

    by_key = _nodes_by_key(state)
    ordered_nodes = [by_key[k] for k in state["selected"] if k in by_key]

    output_dir = os.path.join(SPLATS_DIR, uuid.uuid4().hex)
    progress(0, desc=f"Chunking + reconstructing {len(ordered_nodes)} nodes...")
    try:
        ply_path = reconstruct.reconstruct_chain_windowed(ordered_nodes, output_dir)
    except Exception as e:
        raise gr.Error(f"Windowed reconstruction failed: {e}")

    progress(1.0, desc="Done!")
    yield viewers.pointcloud_viewer_with_download(viewers.file_url(ply_path))


def handle_greedy(state, progress=gr.Progress(track_tqdm=True)):
    """Experimental button: greedy same-capture-pass sliding-window
    reconstruction -- see reconstruct.reconstruct_chain_greedy. Grades every
    2-node window with a real pairwise DA3 call (not solo score), preferring
    whichever capture pass (Apple build_id / Google historical date) has the
    best coverage, and can return multiple disconnected segments instead of
    one merged cloud if no single pass covers the whole selection. Separate
    from every other button -- doesn't touch or replace any of them."""
    if len(state.get("selected", [])) < 2:
        raise gr.Error("Select at least 2 nodes (needs multi-view context for DA3).")

    yield viewers.SPLAT_PLACEHOLDER

    by_key = _nodes_by_key(state)
    ordered_nodes = [by_key[k] for k in state["selected"] if k in by_key]

    output_dir = os.path.join(SPLATS_DIR, uuid.uuid4().hex)
    progress(0, desc=f"Walking {len(ordered_nodes)} nodes by capture pass...")
    try:
        results = reconstruct.reconstruct_chain_greedy(ordered_nodes, output_dir)
    except Exception as e:
        raise gr.Error(f"Greedy reconstruction failed: {e}")

    progress(1.0, desc="Done!")
    yield viewers.filter_sweep_links(results)


def build_tab():
    state = gr.State(_empty_state())

    with gr.Row(equal_height=True):
        area_input = gr.Textbox(
            placeholder="Google Maps URL or lat,lon (e.g. 1.3237, 103.7555)",
            show_label=False,
            container=False,
            scale=5,
        )
        load_btn = gr.Button("Load area", variant="primary", scale=1, min_width=100)

    map_view = gr.HTML(_map_html(_empty_state()), elem_classes="no-pad")
    # visible=True + CSS hiding (BRIDGE_CSS), not visible=False -- see the
    # comment above BRIDGE_CSS for why.
    bridge = gr.Textbox(elem_id=BRIDGE_ELEM_ID, show_label=False, container=False)

    with gr.Row(equal_height=True):
        selection_view = gr.Markdown(_summary_markdown(_empty_state()), elem_id="street_builder_selection")
        with gr.Column(scale=0, min_width=140):
            clear_btn = gr.Button("Clear selection")
            generate_btn = gr.Button("Generate", variant="primary")
            # Debug: compares consensus-filter strictness levels from one
            # DA3 inference call. Not part of the normal Generate flow.
            filter_sweep_btn = gr.Button("Test filter levels (debug)")
            # Experimental: solo-scores a wider candidate pool and
            # reconstructs with only the top 4. Not part of the normal
            # Generate flow.
            best4_btn = gr.Button("Reconstruct (best-4, experimental)")
            # Experimental: chunk+connect for chains longer than one DA3
            # call can handle. Not part of the normal Generate flow.
            windowed_btn = gr.Button("Reconstruct (windowed, experimental)")
            # Experimental: greedy same-capture-pass sliding window, graded
            # by real pairwise DA3 calls instead of solo score. Can return
            # multiple disconnected segments. Not part of the normal
            # Generate flow.
            greedy_btn = gr.Button("Reconstruct (greedy same-pass, experimental)")

    # Drop-ready from page load (not a static placeholder) -- lets you
    # preview an already-downloaded .ply without needing a GPU run first.
    reconstruct_view = gr.HTML(viewers.build_pointcloud_viewer())

    load_btn.click(
        fn=handle_load_area,
        inputs=[area_input, state],
        outputs=[map_view, selection_view, state],
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

    generate_btn.click(
        fn=handle_generate_chain,
        inputs=[state],
        outputs=[reconstruct_view],
        show_progress="minimal",
        show_progress_on=[reconstruct_view],
    )

    filter_sweep_btn.click(
        fn=handle_filter_sweep,
        inputs=[state],
        outputs=[reconstruct_view],
        show_progress="minimal",
        show_progress_on=[reconstruct_view],
    )

    best4_btn.click(
        fn=handle_best4,
        inputs=[state],
        outputs=[reconstruct_view],
        show_progress="minimal",
        show_progress_on=[reconstruct_view],
    )

    windowed_btn.click(
        fn=handle_windowed,
        inputs=[state],
        outputs=[reconstruct_view],
        show_progress="minimal",
        show_progress_on=[reconstruct_view],
    )

    greedy_btn.click(
        fn=handle_greedy,
        inputs=[state],
        outputs=[reconstruct_view],
        show_progress="minimal",
        show_progress_on=[reconstruct_view],
    )
