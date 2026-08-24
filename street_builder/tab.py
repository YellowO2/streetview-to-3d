"""Gradio wiring for the street-builder pathfind/reconstruction flow: given
an already-picked graph of nodes (see street_builder/map_selection/tab.py's
build_map_section), prepare candidates, run the corridor search, and join
segments into a final point cloud.

Composes the whole tab: mounts map_selection's own map-picking section,
then this module's own prepare/run/join controls underneath it, wired
against the same shared `state` map_selection's handlers already update.
"""
import os
import uuid

import gradio as gr

import viewers
from paths import SPLATS_DIR
from street_builder import main as street_main
from street_builder.map_selection.tab import build_map_section, nodes_by_key


def handle_pathfind_prepare(state, progress=gr.Progress(track_tqdm=True)):
    """Experimental button, step 1 of 3: gathers every Google + Apple pano
    near the clicked graph's real shape -- branches and loops included,
    since the selection graph (state["selected"] + state["selected_edges"])
    is only ever built from real Street View edges (see
    map_selection/tab.py's handle_bridge_message), not guessed from click
    order -- and downloads the top-date candidate batch. No GPU here; see
    handle_pathfind_run for why this is its own separate step instead of
    one combined button. See street_main.prepare_pathfind."""
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
    handle_pathfind_join) is a separate button on purpose: it's its own
    separate GPU call, so keeping it out of this call means re-testing/
    tuning join/bridging doesn't require re-running the expensive
    corridor search each time.
    See street_main.run_prepared_pathfind_segments/save_pathfind_segments."""
    if not prep:
        raise gr.Error("Nothing prepared yet -- press \"Prepare\" first.")

    try:
        segments = street_main.run_prepared_pathfind_segments(prep)
        output_dir = os.path.join(SPLATS_DIR, uuid.uuid4().hex)
        results = street_main.save_pathfind_segments(segments, output_dir)
        bundle_path = street_main.save_segments_bundle(segments, output_dir)
    except Exception as e:
        raise gr.Error(f"Auto-path failed: {e}")

    note = "" if len(segments) > 1 else "<p>Single segment -- nothing to join.</p>"
    return viewers.labeled_download_links(results) + note, segments, bundle_path


def handle_pathfind_run_and_join(prep, progress=gr.Progress(track_tqdm=True)):
    """Experimental button, combined 2+3: corridor search + join/bridging
    in ONE GPU session (see street_main.run_prepared_pathfind), instead
    of the separate Run then Join buttons -- avoids paying for two
    separate DA3 model loads when you just want the final result end-
    to-end and don't need to re-test join/bridging separately afterward.
    Still saves a segments bundle (same as handle_pathfind_run), so Join
    can be re-run alone later against this same result if needed."""
    if not prep:
        raise gr.Error("Nothing prepared yet -- press \"Prepare\" first.")

    try:
        output_dir = os.path.join(SPLATS_DIR, uuid.uuid4().hex)
        results, segments, bundle_path = street_main.run_prepared_pathfind(prep, output_dir)
    except Exception as e:
        raise gr.Error(f"Run + Join failed: {e}")

    return viewers.labeled_download_links(results), segments, bundle_path


def handle_pathfind_load_segments(file_path):
    """Loads a previously downloaded segments bundle (see the "Download
    segments" file handle_pathfind_run produces), so Join can run
    immediately without re-running Prepare or the expensive GPU search --
    a different session, or after tweaking join_segments.py. See
    street_main.load_segments_bundle."""
    if not file_path:
        raise gr.Error("Choose a segments file first.")
    try:
        segments = street_main.load_segments_bundle(file_path)
    except Exception as e:
        raise gr.Error(f"Load failed: {e}")
    # Join no longer needs anything from prep (frame_poses in segments
    # already carries each node's lat/lon) -- this placeholder just keeps
    # the "something's ready" gate other handlers check (e.g.
    # handle_pathfind_join's `if not prep or not segments`) true.
    return True, segments, f"<p>Loaded {len(segments)} segment(s) from file. Ready — press \"Join segments\".</p>"


def handle_pathfind_join(prep, segments, progress=gr.Progress(track_tqdm=True)):
    """Experimental button, step 3 of 3: bridges segments together with
    real DA3 tests (see join_segments.join_segments) -- no GPS placement;
    a segment pair known/expected to be adjacent with zero real
    candidates in range is treated as an upstream bug and raises, rather
    than silently falling back to GPS. Its own separate GPU call from
    Run's -- safe to press again after tweaking the join/bridging logic
    without re-running Run. See street_main.save_joined_pathfind."""
    if not prep or not segments:
        raise gr.Error("Nothing to join yet -- press \"Run Auto-path\" first.")
    if len(segments) < 2:
        raise gr.Error("Only one segment -- nothing to join.")

    try:
        output_dir = os.path.join(SPLATS_DIR, uuid.uuid4().hex)
        results = street_main.save_joined_pathfind(segments, output_dir)
    except Exception as e:
        raise gr.Error(f"Join failed: {e}")

    return viewers.labeled_download_links(results)


def build_tab():
    state, map_view, selection_view = build_map_section()

    with gr.Row(equal_height=True):
        with gr.Column(scale=0, min_width=140):
            # Auto-path across the whole clicked graph (branches/loops
            # included). Split into three steps -- prepare (gather +
            # download, no GPU), run (the actual GPU search), join (fit +
            # merge multiple segments, no GPU) -- so the GPU-triggering
            # click is its own fresh interaction instead of following a
            # long download inside one combined request (see
            # handle_pathfind_run's docstring), and re-testing/tuning the
            # join step doesn't require re-running the expensive GPU search
            # each time (see handle_pathfind_join's docstring).
            pathfind_prepare_btn = gr.Button("1. Prepare auto-path (experimental)")
            pathfind_run_btn = gr.Button("2. Run auto-path")
            pathfind_join_btn = gr.Button("3. Join segments")
            pathfind_run_join_btn = gr.Button("2+3. Run + Join (one GPU call)")

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

    pathfind_run_join_btn.click(
        fn=handle_pathfind_run_and_join,
        inputs=[pathfind_prep_state],
        outputs=[reconstruct_view, pathfind_segments_state, pathfind_segments_file],
        show_progress="minimal",
        show_progress_on=[reconstruct_view],
    )
