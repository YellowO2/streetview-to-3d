"""The two buttons: prepare candidates, then reconstruct and place.

Mounts map_selection's own map-picking section above its own controls,
wired against the same shared `state` that section's handlers update.
"""
import os
import time
import zipfile

import gradio as gr

from streetview_to_3d import scene as scene_mod
from streetview_to_3d.ui import viewers
from streetview_to_3d.paths import new_run_dir
from streetview_to_3d.postprocess import pipeline
from streetview_to_3d.reconstruct import build as street_main
from streetview_to_3d.services.pipeline_runner import estimate_gpu_seconds
from streetview_to_3d.ui.map_selection.tab import build_map_section, nodes_by_key

def _run_dir(prep):
    """A fresh output directory, opened as a scene holding every place this
    run will try to reconstruct. See scene.py for what a scene holds."""
    path = new_run_dir()
    # Logged so a run can still be found after the page is refreshed:
    # Gradio serves any file under it at this URL, but can't list the folder.
    print(f"run dir: {path}  (scene: {viewers.file_url(os.path.join(path, scene_mod.FILENAME))})",
          flush=True)
    street_main.open_scene(prep, path)
    return path

def handle_pathfind_prepare(state):
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

    yield None, "<p>Fetching panoramas… This may take a few minutes.</p>"
    try:
        prep = street_main.prepare_pathfind(start, goals, corridor_edges,
                                            (state["lat"], state["lon"]))
    except Exception as e:
        yield None, "<p>Preparation failed. Try again.</p>"
        raise gr.Error(f"Prepare failed: {e}")

    n = len(prep["node_entries"])
    yield prep, f"<p>{n} panoramas ready.</p>" + _gpu_note(len(prep["points"]))


def _gpu_note(n_dots):
    """How much ZeroGPU time the Reconstruct click will ask for, so a user
    can check it against their own daily quota before spending it."""
    minutes = estimate_gpu_seconds(n_dots) / 60
    return f"<p>Estimated GPU budget: {minutes:.1f} min. Queue and download time vary.</p>"


def _zip(run_dir):
    """The whole scene -- scene.json and every node's .ply -- as one zip.

    One file, because a browser can only download files, not a folder, and
    the viewer opens exactly this set once unzipped. Stored, not
    compressed: point data barely shrinks, so compressing would only cost
    time. The Space's disk is wiped on restart, so the scene is handed back
    rather than left as a link into it."""
    names = sorted(n for n in os.listdir(run_dir)
                   if n == scene_mod.FILENAME or n.endswith(".ply"))
    archive = os.path.join(run_dir, f"scene_{os.path.basename(run_dir)[:8]}.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        for n in names:
            z.write(os.path.join(run_dir, n), n)
    return archive


def handle_reconstruct(prep, keep_pct, gpu_seconds):
    """Reconstruct (GPU) then place (CPU), in one click.

    Placement never needs its own GPU call, so it runs immediately after
    reconstruction returns rather than waiting for a second click --
    nothing about it requires a fresh ZeroGPU token the way the GPU call
    itself does (see handle_pathfind_prepare for why THAT stays separate).

    keep_pct: how much of each view's own weakest pixels to keep, from the
    slider -- a UI value, not a redeploy, so it can change without
    rebuilding the Space (see services.da3_ops.CONF_LOWER_PERCENTILE).

    gpu_seconds: the GPU window to ask for; 0 sizes it from the dot count
    (see services.pipeline_runner.estimate_gpu_seconds).
    """
    if not prep:
        raise gr.Error("Nothing prepared yet -- press \"Prepare\" first.")

    # nothing to look at until the scene exists, and a previous run's
    # download would be mistaken for this one's
    yield (gr.HTML(visible=False), gr.DownloadButton(visible=False),
           "<p>Reconstructing… This may take a few minutes.</p>")
    try:
        output_dir = _run_dir(prep)
        street_main.run_prepared_pathfind(
            prep, output_dir, conf_lower_percentile=100 - keep_pct,
            gpu_seconds=gpu_seconds or None)
        yield gr.skip(), gr.skip(), "<p>Aligning the scene…</p>"
        t_place = time.monotonic()
        pipeline.process(output_dir, log=print)
        print(f"timing: placement {time.monotonic() - t_place:.1f}s", flush=True)
    except Exception as e:
        yield gr.HTML(visible=True), gr.skip(), "<p>Reconstruction failed. Try again.</p>"
        raise gr.Error(f"Reconstruct failed: {e}")

    scene_url = viewers.file_url(os.path.join(output_dir, scene_mod.FILENAME))
    yield (gr.HTML(viewers.build_pointcloud_viewer(scene_url=scene_url), visible=True),
           gr.DownloadButton(value=_zip(output_dir), visible=True),
           "<p>Scene ready.</p>")

def build_main_tab():
    state, map_view, selection_view = build_map_section()

    # Three sequential steps, so one row read left to right rather than a
    # narrow sidebar column -- the buttons' own full sentences need real
    # width, and nothing else shares this row with them.
    with gr.Row(equal_height=True):
        # Prepare is separate and has no GPU, so the GPU-triggering click
        # is its own fresh interaction rather than following a long
        # download inside one request -- the ZeroGPU proxy token expires
        # on wall-clock time.
        pathfind_prepare_btn = gr.Button("1. Prepare")
        pathfind_run_btn = gr.Button("2. Reconstruct")

    # A real parameter (services.da3_ops.CONF_LOWER_PERCENTILE), not a UI
    # decision -- kept as a component only so it is callable over the API
    # with a different value; hidden so it isn't something every user has
    # to understand. See handle_reconstruct.
    keep_pct_slider = gr.Slider(50, 100, value=80, step=5, visible=False)
    # Same idea: the ZeroGPU window in seconds, 0 = sized from the dot
    # count. Hidden for now; the estimate is shown after Prepare instead.
    gpu_seconds_input = gr.Number(value=0, precision=0, minimum=0, visible=False)

    pathfind_status = gr.HTML()
    pathfind_prep_state = gr.State(None)

    # The download is what a run is for, so it sits above the viewer and
    # only appears once there is a scene to download (see _zip).
    download_btn = gr.DownloadButton("Download scene (.zip)", visible=False,
                                     variant="primary")
    # Drop-ready from page load (not a static placeholder) -- lets you
    # preview an already-downloaded scene without needing a GPU run first.
    # Hidden while a run is going; see handle_reconstruct.
    reconstruct_view = gr.HTML(viewers.build_pointcloud_viewer())

    pathfind_prepare_btn.click(
        fn=handle_pathfind_prepare,
        inputs=[state],
        outputs=[pathfind_prep_state, pathfind_status],
        show_progress="hidden",
        show_progress_on=[pathfind_status],
    )

    pathfind_run_btn.click(
        fn=handle_reconstruct,
        inputs=[pathfind_prep_state, keep_pct_slider, gpu_seconds_input],
        outputs=[reconstruct_view, download_btn, pathfind_status],
        show_progress="hidden",
        show_progress_on=[reconstruct_view],
    )
