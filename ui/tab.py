"""The three buttons: prepare candidates, reconstruct, place into one scene.

Mounts map_selection's own map-picking section above its own controls,
wired against the same shared `state` that section's handlers update.
"""
import html as html_lib
import os
import shutil
import uuid

import gradio as gr

from ui import viewers
from paths import SPLATS_DIR
from postprocess import pipeline
from reconstruct import build as street_main
from ui.map_selection.tab import build_map_section, nodes_by_key


def _run_dir(prep):
    """A fresh output directory, opened as a scene holding every place this
    run will try to reconstruct. See scene.py for what a scene holds."""
    path = os.path.join(SPLATS_DIR, uuid.uuid4().hex)
    street_main.open_scene(prep, path)
    return path


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
        prep = street_main.prepare_pathfind(start, goals, corridor_edges,
                                            (state["lat"], state["lon"]))
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
        output_dir = _run_dir(prep)
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
        output_dir = _run_dir(prep)
        results, segments, bundle_path = street_main.run_prepared_pathfind(prep, output_dir)
    except Exception as e:
        raise gr.Error(f"Run + Join failed: {e}")

    return viewers.labeled_download_links(results), segments, bundle_path, output_dir


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
        output_dir = _run_dir(prep)
        results = street_main.save_joined_pathfind(segments, output_dir, prep["catalog"])
    except Exception as e:
        raise gr.Error(f"Join failed: {e}")

    return viewers.labeled_download_links(results)


def handle_postprocess(run_dir, progress=gr.Progress(track_tqdm=True)):
    """Step 3: place the run's pieces into one scene. No GPU.

    Everything up to here reconstructs geometry; this is what decides
    where that geometry actually sits -- see postprocess.pipeline.

    Returns the viewer plus the whole scene as one zip.
    """
    if not run_dir:
        raise gr.Error("Nothing reconstructed yet -- press \"Run + Join\" first.")

    lines = []
    def log(msg=""):
        lines.append(str(msg))
        print(msg)

    progress(0, desc="Placing pieces...")
    try:
        ply = pipeline.process(run_dir, log=log)
    except Exception as e:
        raise gr.Error(f"Post-processing failed: {e}")

    # The Space's own disk is wiped whenever it restarts, so the scene is
    # handed back as a file rather than left behind as a link to it.
    bundle = shutil.make_archive(run_dir.rstrip("/"), "zip", run_dir)

    report = "<pre>" + html_lib.escape("\n".join(lines)) + "</pre>"
    return (viewers.build_pointcloud_viewer(viewers.file_url(ply)) + report,
            bundle)


def build_main_tab():
    state, map_view, selection_view = build_map_section()

    with gr.Row(equal_height=True):
        with gr.Column(scale=0, min_width=140):
            # Auto-path across the whole clicked graph (branches/loops
            # included). Prepare is separate and has no GPU, so the
            # GPU-triggering click is its own fresh interaction rather than
            # following a long download inside one request -- the ZeroGPU
            # proxy token expires on wall-clock time.
            #
            # Run and Join are also available as two separate GPU calls,
            # which is what to use when tuning join/bridging against an
            # already-computed search. They are hidden because the combined
            # call is what an ordinary reconstruction wants: one DA3 model
            # load instead of two, and it reuses the panoramas already on
            # disk, which a separate Join call has to re-fetch.
            pathfind_prepare_btn = gr.Button("1. Prepare auto-path (experimental)")
            pathfind_run_btn = gr.Button("2. Run auto-path", visible=False)
            pathfind_join_btn = gr.Button("3. Join segments", visible=False)
            pathfind_run_join_btn = gr.Button("2. Run + Join (one GPU call)")
            pathfind_post_btn = gr.Button("3. Place into one scene (no GPU)")

    pathfind_status = gr.HTML()
    pathfind_prep_state = gr.State(None)
    pathfind_segments_state = gr.State(None)
    pathfind_dir_state = gr.State(None)

    with gr.Row(equal_height=True):
        # Produced by Run -- everything Join needs (prep + segments),
        # pickled to one file. Download it to skip Prepare/Run entirely
        # next time (a later session, or after tweaking join_segments.py):
        # just re-upload it below and press "Load segments".
        pathfind_segments_file = gr.File(label="Segments file (from Run, for Join later)", interactive=False)
        scene_file = gr.File(label="The placed scene (scene.json + one .ply per node)", interactive=False)
        # Only useful with the hidden Join button, so hidden with it.
        with gr.Column(visible=False):
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
        outputs=[reconstruct_view, pathfind_segments_state, pathfind_segments_file,
                 pathfind_dir_state],
        show_progress="minimal",
        show_progress_on=[reconstruct_view],
    )

    pathfind_post_btn.click(
        fn=handle_postprocess,
        inputs=[pathfind_dir_state],
        outputs=[reconstruct_view, scene_file],
        show_progress="minimal",
        show_progress_on=[reconstruct_view],
    )
