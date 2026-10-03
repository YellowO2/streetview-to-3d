"""The main tab: the map section, then two buttons -- prepare, then reconstruct and place
(and, where set up, publish to the gallery)."""
import os
import zipfile

import gradio as gr

from streetview_to_3d.common import scene as scene_mod
from streetview_to_3d.ui import viewers
from streetview_to_3d.common.paths import new_run_dir
from streetview_to_3d.gallery import publish as gallery
from streetview_to_3d.postprocess import pipeline, seams
from streetview_to_3d.postprocess.world import life, water
from streetview_to_3d.reconstruct import build as street_main
from streetview_to_3d.models import mask_api
from streetview_to_3d.reconstruct.runner import (
    DEFAULT_EFFORT, EFFORT_SECONDS_PER_SPOT, WalkSettings, estimate_gpu_seconds,
)
from streetview_to_3d.ui.map_selection.tab import build_map_section, corridor_edges, nodes_by_key


def _run_dir(prep):
    """A fresh output directory, opened as a scene of every place this run will try."""
    path = new_run_dir()
    # logged so a run can be found after a page refresh (Gradio serves files, not folders)
    print(f"run dir: {path}  (scene: {viewers.file_url(os.path.join(path, scene_mod.FILENAME))})",
          flush=True)
    street_main.open_scene(prep, path)
    return path

def handle_pathfind_prepare(state):
    """Button 1: gather and download the selection's candidates (no GPU; see
    street_main.prepare_pathfind)."""
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

    yield None, "<p>Fetching panoramas… This may take a few minutes.</p>"
    try:
        prep = street_main.prepare_pathfind(start, goals, corridor_edges(state), (state["lat"], state["lon"]))
    except Exception as e:
        yield None, "<p>Preparation failed. Try again.</p>"
        raise gr.Error(f"Prepare failed: {e}")

    n = sum(len(c) for g in prep["date_graphs"] for c in g["dot_candidates"].values())
    yield prep, f"<p>{n} panoramas ready. Now click button 2.</p>"


def _zip(run_dir):
    """The whole scene (scene.json, .ply files, water.json, life.json, ground.npz, labels/) as one
    zip the viewer opens. Stored, not compressed: point data barely shrinks."""
    names = sorted(n for n in os.listdir(run_dir)
                   if n in (scene_mod.FILENAME, water.FILENAME, seams.FILENAME, life.FILENAME)
                   or n.endswith(".ply"))
    archive = os.path.join(run_dir, f"scene_{os.path.basename(run_dir)[:8]}.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        for n in names:
            z.write(os.path.join(run_dir, n), n)
        labels = os.path.join(run_dir, "labels")
        for n in sorted(os.listdir(labels)) if os.path.isdir(labels) else []:
            z.write(os.path.join(labels, n), f"labels/{n}")
    return archive


def handle_reconstruct(prep, keep_pct, gpu_seconds, view_hfov=0, da3_model="", masker="",
                       mask_classes="", fill=True, conf_floor=0, effort=DEFAULT_EFFORT,
                       share=False):
    """Button 2: reconstruct (GPU), then place and fill (CPU), in one click.

    The rest are hidden per-run overrides for the API (0/blank: defaults): keep_pct of
    each view's pixels, gpu_seconds (else sized from effort), view_hfov, da3_model, masker,
    comma-separated mask_classes, fill, conf_floor. See reconstruct.runner.WalkSettings.
    share: also publish the scene to the public gallery (gallery.publish).
    """
    if not prep:
        raise gr.Error("Nothing prepared yet -- press \"Prepare\" first.")

    # nothing to look at until the scene exists, and a previous run's
    # download would be mistaken for this one's
    yield (gr.HTML(visible=False), gr.DownloadButton(visible=False),
           "<p>Reconstructing… This may take a few minutes.</p>")
    try:
        output_dir = _run_dir(prep)
        street_main.run_prepared_pathfind(prep, output_dir, WalkSettings(
            conf_lower_percentile=100 - keep_pct,
            gpu_seconds=gpu_seconds or estimate_gpu_seconds(len(prep["points"]), effort or DEFAULT_EFFORT),
            hfov=view_hfov or None,
            model=(da3_model or "").strip() or None, masker=(masker or "").strip() or None,
            mask_classes=[c.strip() for c in (mask_classes or "").split(",") if c.strip()] or None,
            conf_floor=conf_floor or None))
        yield gr.skip(), gr.skip(), "<p>Aligning the scene and filling the gaps…</p>"
        pipeline.process(output_dir, log=lambda m: print(m, flush=True), fill=bool(fill))
    except Exception as e:
        yield gr.HTML(visible=True), gr.skip(), "<p>Reconstruction failed. Try again.</p>"
        raise gr.Error(f"Reconstruct failed: {e}")

    scene_url = viewers.file_url(os.path.join(output_dir, scene_mod.FILENAME))
    yield (gr.HTML(viewers.build_viewer(scene_url=scene_url), visible=True),
           gr.DownloadButton(value=_zip(output_dir), visible=True),
           "<p>Scene ready.</p>")
    if share and gallery.enabled():
        try:
            url = gallery.link(gallery.publish(output_dir, log=lambda m: print(m, flush=True)))
            yield gr.skip(), gr.skip(), f'<p>Scene ready. <a href="{url}" target="_blank">Share link</a></p>'
        except Exception as e:
            print(f"gallery: not published: {e}", flush=True)


def build_main_tab():
    state, map_view, selection_view = build_map_section()

    with gr.Row(equal_height=True):
        # separate, so the GPU click is a fresh interaction: the ZeroGPU token expires on wall-clock
        pathfind_prepare_btn = gr.Button("1. Prepare")
        pathfind_run_btn = gr.Button("2. Reconstruct")
        # how hard to try: the GPU window per spot
        effort_input = gr.Dropdown(list(EFFORT_SECONDS_PER_SPOT), value=DEFAULT_EFFORT, label="Effort",
                                   show_label=False, container=False, scale=0, min_width=130)

    # hidden per-run overrides, settable over the API (see handle_reconstruct)
    keep_pct_slider = gr.Slider(50, 100, value=75, step=5, visible=False)
    gpu_seconds_input = gr.Number(value=0, precision=0, minimum=0, visible=False)
    view_hfov_input = gr.Number(value=0, precision=0, minimum=0, visible=False)
    da3_model_input = gr.Textbox(value="", visible=False)
    masker_input = gr.Textbox(value="", visible=False)
    mask_classes_input = gr.Textbox(value="", visible=False)
    fill_input = gr.Checkbox(value=True, visible=False)
    conf_floor_input = gr.Number(value=0, minimum=0, visible=False)

    # shown where the gallery is set up (gallery.publish.TOKEN_ENV)
    share_input = gr.Checkbox(value=True, visible=gallery.enabled(),
                              label="Add the scene to the public gallery (anyone can view it)")

    pathfind_status = gr.HTML()
    pathfind_prep_state = gr.State(None)

    # shown once there is a scene to download
    download_btn = gr.DownloadButton("Download scene (.zip)", visible=False,
                                     variant="primary")
    # a working viewer from page load: a downloaded scene can be dropped in
    reconstruct_view = gr.HTML(viewers.build_viewer())

    pathfind_prepare_btn.click(
        fn=handle_pathfind_prepare,
        inputs=[state],
        outputs=[pathfind_prep_state, pathfind_status],
        show_progress="hidden",
        show_progress_on=[pathfind_status],
    )

    # The masker alone on one pano, API only (see models.mask_api).
    gr.api(mask_api.mask_pano, api_name="mask_pano")
    gr.api(mask_api.depth_pano, api_name="depth_pano")

    pathfind_run_btn.click(
        fn=handle_reconstruct,
        inputs=[pathfind_prep_state, keep_pct_slider, gpu_seconds_input, view_hfov_input, da3_model_input, masker_input, mask_classes_input, fill_input, conf_floor_input, effort_input, share_input],
        outputs=[reconstruct_view, download_btn, pathfind_status],
        show_progress="hidden",
        show_progress_on=[reconstruct_view],
    )
