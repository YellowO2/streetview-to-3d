"""The two buttons: prepare candidates, then reconstruct and place.

Mounts map_selection's own map-picking section above its own controls,
wired against the same shared `state` that section's handlers update.
"""
import os
import time
import uuid

import gradio as gr

import scene as scene_mod
from ui import viewers
from paths import SPLATS_DIR
from postprocess import pipeline
from reconstruct import build as street_main
from services.pipeline_runner import estimate_gpu_seconds
from ui.map_selection.tab import build_map_section, nodes_by_key

def _run_dir(prep):
    """A fresh output directory, opened as a scene holding every place this
    run will try to reconstruct. See scene.py for what a scene holds."""
    path = os.path.join(SPLATS_DIR, uuid.uuid4().hex)
    # Logged so a run can still be found after the page is refreshed:
    # Gradio serves any file under it at this URL, but can't list the folder.
    print(f"run dir: {path}  (scene: {viewers.file_url(os.path.join(path, scene_mod.FILENAME))})",
          flush=True)
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
    return prep, (f"<p>Prepared {n} candidate(s) across {len(prep['top_dates'])} date(s). "
                  f"Ready — press \"Reconstruct\".</p>" + _gpu_note(len(prep["points"])))

def _gpu_note(n_dots):
    """How much ZeroGPU time the Reconstruct click will ask for, so a user
    can check it against their own daily quota before spending it."""
    minutes = estimate_gpu_seconds(n_dots) / 60
    return (f"<p>{n_dots} place(s) to reconstruct: this needs about <b>{minutes:.1f} min</b> "
            f"of GPU time. ZeroGPU gives each account a daily quota (5 min free, 40 min PRO) "
            f"-- check you have enough left, or the run will be refused.</p>")

def _files(run_dir):
    """scene.json plus every node's own .ply, as plain paths.

    No zip: the Space's disk is wiped on restart, so these are handed back
    directly rather than left as a link into it, and the viewer already
    opens exactly this file set (drag them in, or "Open files")."""
    names = sorted(n for n in os.listdir(run_dir)
                   if n == scene_mod.FILENAME or n.endswith(".ply"))
    return [os.path.join(run_dir, n) for n in names]


def handle_reconstruct(prep, keep_pct, gpu_seconds, progress=gr.Progress(track_tqdm=True)):
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

    try:
        output_dir = _run_dir(prep)
        street_main.run_prepared_pathfind(
            prep, output_dir, conf_lower_percentile=100 - keep_pct,
            gpu_seconds=gpu_seconds or None)
        t_place = time.monotonic()
        pipeline.process(output_dir, log=print)
        print(f"timing: placement {time.monotonic() - t_place:.1f}s", flush=True)
    except Exception as e:
        raise gr.Error(f"Reconstruct failed: {e}")

    scene_url = viewers.file_url(os.path.join(output_dir, scene_mod.FILENAME))
    return (viewers.build_pointcloud_viewer(scene_url=scene_url),
            _files(output_dir))

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
        pathfind_prepare_btn = gr.Button("1. Prepare (fetch panoramas)")
        pathfind_run_btn = gr.Button("2. Reconstruct and place")

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

    # The Space's disk does not survive a restart, so a finished scene is
    # handed back as files rather than left behind as a link into it -- no
    # zip: the viewer already opens exactly this file set directly.
    scene_files = gr.Files(label="The scene (scene.json + one .ply per node)",
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
        inputs=[pathfind_prep_state, keep_pct_slider, gpu_seconds_input],
        outputs=[reconstruct_view, scene_files],
        show_progress="minimal",
        show_progress_on=[reconstruct_view],
    )
