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
    return prep, f"<p>Prepared {n} candidate(s) across {len(prep['top_dates'])} date(s). Ready — press \"Reconstruct\".</p>"

def handle_reconstruct(prep, progress=gr.Progress(track_tqdm=True)):
    """Step 2: walk the corridor and bridge what it finds, in ONE GPU call.

    Search and bridging share a session so DA3 is loaded once and the
    panoramas already on disk are reused -- a second call has no guarantee
    of landing on the same worker.
    """
    if not prep:
        raise gr.Error("Nothing prepared yet -- press \"Prepare\" first.")

    try:
        output_dir = _run_dir(prep)
        results = street_main.run_prepared_pathfind(prep, output_dir)
    except Exception as e:
        raise gr.Error(f"Reconstruct failed: {e}")

    return viewers.summary(results, 'Reconstructed. Press "3. Place into one '
                           "scene\" to fit it to the map -- that step hands back "
                           "the whole scene as a download."), output_dir

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
            # Prepare is separate and has no GPU, so the GPU-triggering
            # click is its own fresh interaction rather than following a
            # long download inside one request -- the ZeroGPU proxy token
            # expires on wall-clock time.
            pathfind_prepare_btn = gr.Button("1. Prepare (fetch panoramas)")
            pathfind_run_btn = gr.Button("2. Reconstruct (GPU)")
            pathfind_post_btn = gr.Button("3. Place into one scene (no GPU)")

    pathfind_status = gr.HTML()
    pathfind_prep_state = gr.State(None)
    pathfind_dir_state = gr.State(None)

    # The Space's disk does not survive a restart, so a finished scene is
    # handed back as a file rather than left behind as a link to it.
    scene_file = gr.File(label="The placed scene (scene.json + one .ply per node)",
                         interactive=False)

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
        fn=handle_reconstruct,
        inputs=[pathfind_prep_state],
        outputs=[reconstruct_view, pathfind_dir_state],
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
