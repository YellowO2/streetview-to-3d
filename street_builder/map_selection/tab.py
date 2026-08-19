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
from services.geo import extract_lat_lon
from services.streetview_fetch import fetch_pano_by_id, run_async
from street_builder.map_selection import candidates as candidates_mod
from street_builder.map_selection import map_ui
from street_builder.reconstruction import generate, greedy
from street_builder import main as street_main

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
    return {"lat": None, "lon": None, "nodes": [], "edges": [], "selected": [], "selected_edges": [], "view": None}


def _nodes_by_key(state):
    return {n["key"]: n for n in state["nodes"]}


def _summary_markdown(state):
    if not state["selected"]:
        return "_Load an area to set the start node, then click markers to extend the graph._"
    by_key = _nodes_by_key(state)
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
    if state["lat"] is None:
        return map_ui.build_picker_map(0, 0, [], [], [], [], zoom=2)
    return map_ui.build_picker_map(
        state["lat"], state["lon"], state["nodes"], state["edges"],
        state["selected"], state.get("selected_edges", []),
        zoom=zoom, view=state.get("view"),
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


def handle_generate_chain(state, progress=gr.Progress(track_tqdm=True)):
    if len(state.get("selected", [])) < 2:
        raise gr.Error("Select at least 2 nodes (needs multi-view context for DA3).")

    yield viewers.SPLAT_PLACEHOLDER

    by_key = _nodes_by_key(state)
    ordered_nodes = [by_key[k] for k in state["selected"] if k in by_key]

    output_dir = os.path.join(SPLATS_DIR, uuid.uuid4().hex)
    progress(0, desc=f"Reconstructing {len(ordered_nodes)} nodes...")
    try:
        ply_path = generate.reconstruct_chain(ordered_nodes, output_dir)
    except Exception as e:
        raise gr.Error(f"Reconstruction failed: {e}")

    progress(1.0, desc="Done!")
    yield viewers.pointcloud_viewer_with_download(viewers.file_url(ply_path))


def handle_greedy(state, progress=gr.Progress(track_tqdm=True)):
    """Experimental button: greedy same-capture-pass sliding-window
    reconstruction -- see greedy.reconstruct_chain_greedy. Grades every
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
        results = greedy.reconstruct_chain_greedy(ordered_nodes, output_dir)
    except Exception as e:
        raise gr.Error(f"Greedy reconstruction failed: {e}")

    progress(1.0, desc="Done!")
    yield viewers.labeled_download_links(results)


def handle_pathfind_prepare(state, progress=gr.Progress(track_tqdm=True)):
    """Experimental button, step 1 of 3: gathers every Google + Apple pano
    near the clicked graph's real shape -- branches and loops included,
    since the selection graph (state["selected"] + state["selected_edges"])
    is only ever built from real Street View edges (see
    handle_bridge_message), not guessed from click order -- and downloads
    the top-date candidate batch. No GPU here; see handle_pathfind_run for
    why this is its own separate step instead of one combined button.
    See street_main.prepare_pathfind."""
    selected = state.get("selected", [])
    selected_edges = state.get("selected_edges", [])
    if len(selected) < 2 or not selected_edges:
        raise gr.Error("Select at least 2 connected nodes tracing the route (start to a goal).")

    by_key = _nodes_by_key(state)
    start_node = by_key.get(selected[0])
    if not start_node:
        raise gr.Error("Start node not found.")
    start = (start_node["lat"], start_node["lon"])
    goals = [(by_key[k]["lat"], by_key[k]["lon"]) for k in selected[1:] if k in by_key]
    corridor_edges = [
        ((by_key[a]["lat"], by_key[a]["lon"]), (by_key[b]["lat"], by_key[b]["lon"]))
        for a, b in selected_edges if a in by_key and b in by_key
    ]

    progress(0, desc="Gathering + downloading candidates...")
    try:
        prep = street_main.prepare_pathfind(start, goals, corridor_edges)
    except Exception as e:
        raise gr.Error(f"Prepare failed: {e}")

    progress(1.0, desc="Done!")
    n = len(prep["node_entries"])
    return prep, f"<p>Prepared {n} candidate(s) across {len(prep['top_dates'])} date(s). Ready — press \"Run Auto-path\".</p>"


def handle_pathfind_run(prep, progress=gr.Progress(track_tqdm=True)):
    """Experimental button, step 2 of 3: runs the real multi-goal best-first
    search over whatever handle_pathfind_prepare already downloaded -- the
    fixed start node stays the search's start; every other selected node is
    a goal, and the search doesn't stop at the first one reached, it keeps
    growing toward whatever's still outstanding.

    Split from the prepare step specifically so this GPU-triggering click
    is its own fresh, minimal-latency interaction -- the ZeroGPU proxy
    token's validity is wall-clock, and a long download sitting ahead of
    the @spaces.GPU call (as one combined button used to do) is exactly
    what can let it go stale before the schedule request is ever sent.

    Only saves each segment's own preview here -- joining them (step 3,
    handle_pathfind_join) is a separate button on purpose: it needs no GPU
    at all, so keeping it out of this call means re-testing/tuning the join
    step doesn't require re-running the expensive DA3 search each time.
    See street_main.run_prepared_pathfind_segments/save_pathfind_segments."""
    if not prep:
        raise gr.Error("Nothing prepared yet -- press \"Prepare\" first.")

    yield viewers.SPLAT_PLACEHOLDER, None, None

    try:
        segments = street_main.run_prepared_pathfind_segments(prep)
        output_dir = os.path.join(SPLATS_DIR, uuid.uuid4().hex)
        results = street_main.save_pathfind_segments(segments, output_dir)
        bundle_path = street_main.save_segments_bundle(prep, segments, output_dir)
    except Exception as e:
        raise gr.Error(f"Auto-path failed: {e}")

    note = "" if len(segments) > 1 else "<p>Single segment -- nothing to join.</p>"
    yield viewers.labeled_download_links(results) + note, segments, bundle_path


def handle_pathfind_load_segments(file_path):
    """Loads a previously downloaded segments bundle (see the "Download
    segments" file handle_pathfind_run produces), so Join can run
    immediately without re-running Prepare or the expensive GPU search --
    a different session, or after tweaking join_segments.py. See
    street_main.load_segments_bundle."""
    if not file_path:
        raise gr.Error("Choose a segments file first.")
    try:
        prep, segments = street_main.load_segments_bundle(file_path)
    except Exception as e:
        raise gr.Error(f"Load failed: {e}")
    return prep, segments, f"<p>Loaded {len(segments)} segment(s) from file. Ready — press \"Join segments\".</p>"


def handle_pathfind_join(prep, segments, progress=gr.Progress(track_tqdm=True)):
    """Experimental button, step 3 of 3: fits each segment from the last Run
    against its own real GPS positions and merges them into one point cloud
    (see join_segments.join_segments). No GPU -- safe to press again after
    tweaking the join logic without re-running Run. See
    street_main.save_joined_pathfind."""
    if not prep or not segments:
        raise gr.Error("Nothing to join yet -- press \"Run Auto-path\" first.")
    if len(segments) < 2:
        raise gr.Error("Only one segment -- nothing to join.")

    yield viewers.SPLAT_PLACEHOLDER

    try:
        output_dir = os.path.join(SPLATS_DIR, uuid.uuid4().hex)
        label, ply = street_main.save_joined_pathfind(prep, segments, output_dir)
    except Exception as e:
        raise gr.Error(f"Join failed: {e}")

    yield viewers.labeled_download_links([(label, ply)])


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
            # Experimental: greedy same-capture-pass sliding window, graded
            # by real pairwise DA3 calls instead of solo score. Can return
            # multiple disconnected segments. Not part of the normal
            # Generate flow.
            greedy_btn = gr.Button("Reconstruct (greedy same-pass, experimental)")
            # Experimental: auto-path across the whole clicked graph
            # (branches/loops included). Split into three steps -- prepare
            # (gather + download, no GPU), run (the actual GPU search), join
            # (fit + merge multiple segments, no GPU) -- so the
            # GPU-triggering click is its own fresh interaction instead of
            # following a long download inside one combined request (see
            # handle_pathfind_run's docstring), and re-testing/tuning the
            # join step doesn't require re-running the expensive GPU search
            # each time (see handle_pathfind_join's docstring). Not part of
            # the normal Generate flow.
            pathfind_prepare_btn = gr.Button("1. Prepare auto-path (experimental)")
            pathfind_run_btn = gr.Button("2. Run auto-path")
            pathfind_join_btn = gr.Button("3. Join segments")

    pathfind_status = gr.HTML()
    pathfind_prep_state = gr.State(None)
    pathfind_segments_state = gr.State(None)

    with gr.Row(equal_height=True):
        # Produced by Run -- everything Join needs (prep + segments),
        # pickled to one file. Download it to skip Prepare/Run entirely
        # next time (a later session, or after tweaking join_segments.py):
        # just re-upload it below and press "Load segments".
        pathfind_segments_file = gr.File(label="Segments file (from Run, for Join later)", interactive=False)
        with gr.Column():
            pathfind_segments_upload = gr.File(label="...or load a previously downloaded segments file", file_types=[".pkl"], type="filepath")
            pathfind_load_btn = gr.Button("Load segments")

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

    greedy_btn.click(
        fn=handle_greedy,
        inputs=[state],
        outputs=[reconstruct_view],
        show_progress="minimal",
        show_progress_on=[reconstruct_view],
    )

    pathfind_prepare_btn.click(
        fn=handle_pathfind_prepare,
        inputs=[state],
        outputs=[pathfind_prep_state, pathfind_status],
        show_progress="minimal",
        show_progress_on=[pathfind_status],
    )

    pathfind_run_btn.click(
        fn=handle_pathfind_run,
        inputs=[pathfind_prep_state],
        outputs=[reconstruct_view, pathfind_segments_state, pathfind_segments_file],
        show_progress="minimal",
        show_progress_on=[reconstruct_view],
    )

    pathfind_load_btn.click(
        fn=handle_pathfind_load_segments,
        inputs=[pathfind_segments_upload],
        outputs=[pathfind_prep_state, pathfind_segments_state, pathfind_status],
        show_progress="minimal",
        show_progress_on=[pathfind_status],
    )

    pathfind_join_btn.click(
        fn=handle_pathfind_join,
        inputs=[pathfind_prep_state, pathfind_segments_state],
        outputs=[reconstruct_view],
        show_progress="minimal",
        show_progress_on=[reconstruct_view],
    )
