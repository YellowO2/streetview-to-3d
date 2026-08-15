"""
Gradio interface for Street View to 3DGS.

Run locally:  python gradio_app.py
HF Spaces:    set as app.py, add `spaces` to requirements, enable ZeroGPU.
"""

import os
import shutil
import time
import uuid

import gradio as gr
import pillow_heif

pillow_heif.register_heif_opener()

# Must be the first project import. ZeroGPU requires `spaces` (imported by
# services.pipeline_runner) to load before anything CUDA-related does.
# services.lookaround_fetch imports streetlevel's `reproject` module, which
# initializes CUDA at import time to pick its default device -- if that runs
# first, `import spaces` fails with "CUDA has been initialized before
# importing the `spaces` package." Importing pipeline_runner here, before
# the services import below, guarantees the required order regardless of
# what order the names in that later `from services import ...` get resolved in.
from services import pipeline_runner

import viewers
from paths import IMAGES_DIR, SPLATS_DIR
from prompts.presets import PRESET_NAMES, build_prompt, get_preset
from services import geo, lookaround_fetch, slope_correction, streetview_fetch
from street_builder.tab import build_tab as build_street_builder_tab, BRIDGE_HEAD_SCRIPT, BRIDGE_CSS

# Auto-pick up to N nearest neighbors as DA3 depth-support panos.
MAX_SUPPORT_PANOS = 2


# ── handlers ───────────────────────────────────────────────────────────────────


def handle_load(url_input):
    try:
        lat, lon = geo.extract_lat_lon(url_input)
    except ValueError as e:
        raise gr.Error(str(e))

    try:
        meta = streetview_fetch.run_async(streetview_fetch.fetch_pano(lat, lon))
    except Exception as e:
        raise gr.Error(str(e))
    if not meta:
        raise gr.Error("Panorama not found at that location.")

    try:
        img_path = streetview_fetch.run_async(streetview_fetch.download_pano(meta["lat"], meta["lon"]))
    except Exception as e:
        raise gr.Error(f"Download failed: {e}")

    state = {
        "source": "streetview",
        "image_path": img_path,
        "original_image_path": img_path,
        **meta,
    }
    pano_choices = [(f"Google · {d['label']}", f"google:{d['id']}") for d in meta["dates"]]
    try:
        for p in lookaround_fetch.apple_candidates(meta["lat"], meta["lon"]):
            dist_m = geo.haversine_m(meta["lat"], meta["lon"], p.lat, p.lon)
            pano_choices.append((f"Apple · {dist_m:.0f}m · {streetview_fetch.format_date(p.date)}", f"apple:{p.id}"))
    except Exception:
        pass  # Look Around coverage lookup is best-effort; Google picker still works without it.

    return (
        viewers.build_map(meta["lat"], meta["lon"]),
        viewers.build_pano_viewer(viewers.file_url(img_path)),
        state,
        gr.update(choices=pano_choices, value=f"google:{meta['id']}", visible=len(pano_choices) > 1),
    )


def handle_select_pano(pano_state, selected_value):
    if not pano_state or pano_state.get("source") not in ("streetview", "lookaround"):
        raise gr.Error("Load a Street View location first.")
    current_value = ("google" if pano_state["source"] == "streetview" else "apple") + ":" + str(pano_state.get("id"))
    if not selected_value or selected_value == current_value:
        return gr.update(), gr.update(), pano_state

    source, pano_id = selected_value.split(":", 1)

    if source == "google":
        try:
            meta = streetview_fetch.run_async(streetview_fetch.fetch_pano_by_id(pano_id))
        except Exception as e:
            raise gr.Error(f"Failed to load that date: {e}")
        if not meta:
            raise gr.Error("Panorama not found for that date.")
        try:
            img_path = streetview_fetch.run_async(streetview_fetch.download_pano_by_id(meta["id"]))
        except Exception as e:
            raise gr.Error(f"Download failed: {e}")
        state = {"source": "streetview", "image_path": img_path, "original_image_path": img_path, **meta}
    else:
        try:
            pano = lookaround_fetch.apple_nearby_panos(pano_state["lat"], pano_state["lon"]).get(int(pano_id))
        except Exception as e:
            raise gr.Error(f"Failed to look up that Apple panorama: {e}")
        if not pano:
            raise gr.Error("Apple panorama not found near that location.")
        try:
            img_path = lookaround_fetch.download_lookaround(pano)
        except Exception as e:
            raise gr.Error(f"Download failed: {e}")
        meta = lookaround_fetch.apple_pano_to_meta(pano)
        state = {"source": "lookaround", "image_path": img_path, "original_image_path": img_path, **meta}

    return (
        viewers.build_map(meta["lat"], meta["lon"]),
        viewers.build_pano_viewer(viewers.file_url(img_path)),
        state,
    )


def handle_edit(pano_state, prompt, preset_name, progress=gr.Progress()):
    if not pano_state or not pano_state.get("image_path"):
        raise gr.Error("Load or upload a panorama first.")
    if not prompt or not prompt.strip():
        raise gr.Error("Enter an edit prompt.")

    preset = get_preset(preset_name)
    mode = preset["mode"] if preset else "general"
    geom_preserving = preset["geom_preserving"] if preset else False

    src = pano_state["image_path"]
    out_path = os.path.join(IMAGES_DIR, f"edit_{uuid.uuid4().hex}.png")

    progress(0, desc="Running edit (~40s)...")
    try:
        pipeline_runner.run_editor_gpu(src, prompt.strip(), mode, out_path)
    except Exception as e:
        raise gr.Error(f"Edit failed: {e}")

    new_state = {**pano_state, "image_path": out_path}
    if not geom_preserving:
        new_state["original_image_path"] = out_path

    return (
        viewers.build_pano_viewer(viewers.file_url(out_path)),
        new_state,
    )


def handle_upload(file_path):
    if not file_path:
        raise gr.Error("No file selected.")
    ext = os.path.splitext(file_path)[1] or ".jpg"
    dest = os.path.join(IMAGES_DIR, f"upload_{uuid.uuid4().hex}{ext}")
    shutil.copy(file_path, dest)
    state = {"source": "upload", "image_path": dest, "original_image_path": dest}
    return (viewers.build_pano_viewer(viewers.file_url(dest)), state)


def handle_generate(pano_state, scale_mode, output_mode, use_support_panos, correct_slope, slope_multiplier, progress=gr.Progress(track_tqdm=True)):
    if not pano_state or not pano_state.get("image_path"):
        raise gr.Error("Load or upload a panorama first.")

    yield viewers.SPLAT_PLACEHOLDER

    target_path = pano_state["image_path"]
    target_depth_path = pano_state.get("original_image_path", target_path)

    if (
        output_mode == "DA3 Point Cloud"
        and correct_slope
        and pano_state.get("heading") is not None
        and pano_state.get("pitch") is not None
        and pano_state.get("roll") is not None
    ):
        try:
            target_depth_path = slope_correction.correct_slope(
                target_depth_path,
                pano_state["heading"],
                pano_state["pitch"],
                pano_state["roll"],
                multiplier=slope_multiplier,
            )
        except Exception as e:
            raise gr.Error(f"Slope correction failed: {e}")

    support_paths = []

    neighbors = (
        pano_state.get("neighbors", [])[:MAX_SUPPORT_PANOS]
        if use_support_panos and pano_state.get("source") == "streetview"
        else []
    )
    for i, n in enumerate(neighbors):
        try:
            progress(0, desc=f"Downloading support pano {i+1}/{len(neighbors)}...")
            # By id (same download_pano_by_id street_builder's reconstruction
            # script uses), not by lat/lon -- avoids a second coordinate
            # lookup when the exact pano id is already known.
            p = streetview_fetch.run_async(streetview_fetch.download_pano_by_id(n["id"]))
            if p:
                support_paths.append(p)
        except Exception:
            pass

    output_dir = os.path.join(SPLATS_DIR, uuid.uuid4().hex)
    t_start = time.time()

    if output_mode == "DA3 Point Cloud":
        try:
            ply_path = pipeline_runner.run_pointcloud_gpu(target_depth_path, output_dir, support_paths=support_paths)
        except Exception as e:
            raise gr.Error(f"Pipeline failed: {e}")

        if not ply_path or not os.path.exists(ply_path):
            raise gr.Error("Pipeline finished but no point cloud produced.")

        elapsed = time.time() - t_start
        progress(1.0, desc=f"Done! {1 + len(support_paths)} panos, {elapsed:.0f}s")
        yield viewers.pointcloud_download_view(viewers.file_url(ply_path))
        return

    try:
        ply_path = pipeline_runner.run_pipeline_gpu(
            target_path,
            output_dir,
            scale_mode,
            "sharp",
            support_paths=support_paths,
            target_depth_path=target_depth_path if target_depth_path != target_path else None,
        )
    except Exception as e:
        raise gr.Error(f"Pipeline failed: {e}")

    if not ply_path or not os.path.exists(ply_path):
        raise gr.Error("Pipeline finished but no PLY produced.")

    elapsed = time.time() - t_start
    progress(1.0, desc=f"Done! {1 + len(support_paths)} panos, {elapsed:.0f}s")
    yield viewers.splat_viewer_with_download(viewers.file_url(ply_path))



# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Street View to 3DGS") as demo:
    with gr.Tabs():
        with gr.Tab("Single Panorama"):
            gr.Markdown(
                "# Street View to 3DGS\n"
                "Convert a Google Street View location into a 3D Gaussian Splat scene. "
                "[[GitHub](https://github.com/YellowO2/streetview-to-3dgs)]\n\n"
                "**1.** Paste a Google Maps URL → **2.** Optionally edit the panorama → **3.** Generate 3DGS"
            )

            pano_state = gr.State(None)

            # ── Step 1 ─────────────────────────────────────────────────────────────────
            gr.Markdown("## Step 1. Load panorama")
            with gr.Row(equal_height=True):
                url_input = gr.Textbox(
                    placeholder="Google Maps URL or lat,lon (e.g. 1.3237, 103.7555)",
                    show_label=False,
                    container=False,
                    scale=5,
                )
                load_btn = gr.Button("Load", variant="primary", scale=1, min_width=80)
                upload_pano = gr.UploadButton(
                    "Upload panorama (beta)",
                    file_types=[".jpg", ".jpeg", ".png"],
                    scale=1,
                    min_width=80,
                )

            pano_dropdown = gr.Dropdown(
                label="Source pano",
                info="Other captures of this spot — Google Street View dates and nearby Apple Look Around panos.",
                choices=[],
                visible=False,
            )

            with gr.Row(equal_height=True):
                map_view = gr.HTML(viewers.MAP_PLACEHOLDER, elem_classes="no-pad")
                pano_view = gr.HTML(viewers.PANO_PLACEHOLDER, elem_classes="no-pad")

            pano_download = gr.DownloadButton(
                label="⬇  Download current panorama",
                visible=False,
                size="sm",
            )

            gr.Markdown("Edit panorama (optional) — ~0.7 min")
            with gr.Row(equal_height=True):
                edit_prompt = gr.Textbox(
                    placeholder="Pick a preset or type your own (e.g. add snow)",
                    show_label=False,
                    container=False,
                    scale=4,
                    lines=2,
                )
                edit_preset = gr.Dropdown(
                    choices=PRESET_NAMES,
                    value="(none)",
                    show_label=False,
                    container=False,
                    scale=1,
                )
                edit_btn = gr.Button("Edit", scale=1, min_width=80)

            def _apply_preset(preset_name):
                prompt = build_prompt(preset_name)
                return gr.update(value=prompt) if prompt else gr.update(value="")

            edit_preset.change(
                fn=_apply_preset,
                inputs=[edit_preset],
                outputs=[edit_prompt],
            )

            # ── Step 2 ─────────────────────────────────────────────────────────────────
            gr.Markdown("## Step 2. Generate 3DGS ~1.8 min")
            with gr.Row(equal_height=True):
                scale_mode = gr.Dropdown(
                    choices=["da3_y_ground", "da3_2dgrid_global"],
                    value="da3_y_ground",
                    label="Scale mode",
                    info="How depth is aligned to the scene.",
                    scale=2,
                )
                output_mode = gr.Dropdown(
                    choices=["3D Gaussian Splat", "DA3 Point Cloud"],
                    value="3D Gaussian Splat",
                    label="Output type",
                    info="Generate a 3DGS or a point cloud.",
                    scale=2,
                )
                use_support_panos = gr.Checkbox(
                    value=True,
                    label="Use supporting panoramas",
                    info="Nearby panos as DA3 depth/pose context.",
                    scale=1,
                )
                correct_slope = gr.Checkbox(
                    value=False,
                    label="Correct for slope (experimental)",
                    info="De-tilt the target pano using its pitch/roll before DA3.",
                    scale=1,
                    visible=False,
                )
                slope_multiplier = gr.Number(
                    value=1.0,
                    label="Slope correction ×",
                    info="Scales the pitch/roll correction. Try >1 if 1x isn't enough.",
                    scale=1,
                    visible=False,
                )
                generate_btn = gr.Button("Generate", variant="primary", scale=1, min_width=160)

            splat_view = gr.HTML(viewers.SPLAT_PLACEHOLDER)

            def _refresh_pano_download(state):
                path = (state or {}).get("image_path") if state else None
                if path and os.path.exists(path):
                    return gr.update(visible=True, value=path)
                return gr.update(visible=False, value=None)

            pano_state.change(
                fn=_refresh_pano_download,
                inputs=[pano_state],
                outputs=[pano_download],
            )

            load_btn.click(
                fn=handle_load,
                inputs=[url_input],
                outputs=[map_view, pano_view, pano_state, pano_dropdown],
            )

            pano_dropdown.change(
                fn=handle_select_pano,
                inputs=[pano_state, pano_dropdown],
                outputs=[map_view, pano_view, pano_state],
            )

            upload_pano.upload(
                fn=handle_upload,
                inputs=[upload_pano],
                outputs=[pano_view, pano_state],
            ).then(
                fn=lambda: gr.update(choices=[], value=None, visible=False),
                outputs=[pano_dropdown],
            )

            edit_btn.click(
                fn=handle_edit,
                inputs=[pano_state, edit_prompt, edit_preset],
                outputs=[pano_view, pano_state],
                show_progress="minimal",
                show_progress_on=[pano_view],
            )

            output_mode.change(
                fn=lambda mode: gr.update(visible=mode == "3D Gaussian Splat"),
                inputs=[output_mode],
                outputs=[scale_mode],
            )
            output_mode.change(
                fn=lambda mode: (
                    gr.update(visible=mode == "DA3 Point Cloud"),
                    gr.update(visible=mode == "DA3 Point Cloud"),
                ),
                inputs=[output_mode],
                outputs=[correct_slope, slope_multiplier],
            )

            generate_btn.click(
                fn=handle_generate,
                inputs=[pano_state, scale_mode, output_mode, use_support_panos, correct_slope, slope_multiplier],
                outputs=[splat_view],
                show_progress="minimal",
                show_progress_on=[splat_view],
            )
        with gr.Tab("Street Builder"):
            build_street_builder_tab()


if __name__ == "__main__":
    demo.launch(
        allowed_paths=[IMAGES_DIR, SPLATS_DIR],
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Default(),
        css=".no-pad { padding-left: 0 !important; padding-right: 0 !important; } " + BRIDGE_CSS,
        head=BRIDGE_HEAD_SCRIPT,
        # Explicitly off: the startup log showed "with SSR (Node proxy ->
        # Python :7861)" -- an extra Node.js hop HF Spaces enables by
        # default -- right before the Space got stuck permanently on
        # "restarting" despite the Python server itself logging a
        # successful start. Forcing plain client-side rendering removes
        # that layer as a suspect.
        ssr_mode=False,
    )
