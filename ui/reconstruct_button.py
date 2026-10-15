"""Run button 2: build the corridor's scene, then show and share it."""
import logging
import os
import zipfile
from functools import partial, update_wrapper

import gradio as gr

from ui.Controls import Controls
from ui.tools.gallery.publish import publish, sharing
from ui.tools.page import file_url
from ui.tools.viewer import viewer
from streetview_to_3d.Settings import Settings
from streetview_to_3d.types.Scene import FILENAME as SCENE_FILENAME
from streetview_to_3d.postprocess.Settings import Settings as PostprocessSettings
from streetview_to_3d.reconstruct.WalkSettings import WalkSettings
from streetview_to_3d.reconstruct.tools.gpu_seconds import DEFAULT_EFFORT, estimate_gpu_seconds

logger = logging.getLogger(__name__)


def _zip(run_dir):
    """The whole scene folder as save wrote it (scene.json, .ply files, water.json, life.json,
    ground.npz, labels/) as one zip the viewer opens. Stored, not compressed: point data
    barely shrinks."""
    names = sorted(os.listdir(run_dir))
    archive = os.path.join(run_dir, f"scene_{os.path.basename(run_dir)[:8]}.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        for n in names:
            if os.path.isfile(os.path.join(run_dir, n)):
                z.write(os.path.join(run_dir, n), n)
        for folder in names:
            if os.path.isdir(os.path.join(run_dir, folder)):
                for n in sorted(os.listdir(os.path.join(run_dir, folder))):
                    z.write(os.path.join(run_dir, folder, n), f"{folder}/{n}")
    return archive


def _handle_reconstruct(corridor, keep_pct, gpu_seconds, view_hfov=0, da3_model="", masker="",
                       mask_classes="", fill=True, conf_floor=0, effort=DEFAULT_EFFORT,
                       share=False, *, build):
    """Button 2: build (the engine's streetview_to_3d): reconstruct (GPU), then place and
    fill (CPU), in one click.

    The rest are hidden per-run overrides for the API (0/blank: defaults): keep_pct of
    each view's pixels, gpu_seconds (else sized from effort), view_hfov, da3_model, masker,
    comma-separated mask_classes, fill, conf_floor. See reconstruct.WalkSettings.
    share: also publish the scene to the public gallery (ui/tools/gallery/publish.py).
    """
    if not corridor:
        raise gr.Error("Nothing prepared yet -- press \"Prepare\" first.")

    # nothing to look at until the scene exists, and a previous run's
    # download would be mistaken for this one's
    yield (gr.HTML(visible=False), gr.DownloadButton(visible=False),
           "<p>Reconstructing… This may take a few minutes.</p>")
    try:
        walk = WalkSettings(
            conf_lower_percentile=100 - keep_pct,
            gpu_seconds=gpu_seconds or estimate_gpu_seconds(len(corridor.points), effort or DEFAULT_EFFORT),
            hfov=view_hfov or None,
            model=(da3_model or "").strip() or None, masker=(masker or "").strip() or None,
            mask_classes=[c.strip() for c in (mask_classes or "").split(",") if c.strip()] or None,
            conf_floor=conf_floor or None)
        output_dir = build(corridor, Settings(walk=walk, post=PostprocessSettings(fill=bool(fill))))
    except Exception as e:
        yield gr.HTML(visible=True), gr.skip(), "<p>Reconstruction failed. Try again.</p>"
        raise gr.Error(f"Reconstruct failed: {e}")

    scene_url = file_url(os.path.join(output_dir, SCENE_FILENAME))
    yield (gr.HTML(viewer(scene_url=scene_url), visible=True),
           gr.DownloadButton(value=_zip(output_dir), visible=True),
           "<p>Scene ready.</p>")
    if share and sharing():
        try:
            url = publish(output_dir)
            yield gr.skip(), gr.skip(), f'<p>Scene ready. <a href="{url}" target="_blank">Share link</a></p>'
        except Exception as e:
            logger.warning(f"gallery: not published: {e}")


def reconstruct_button(corridor, controls: Controls, build):
    """Wire button 2 to the prepared corridor (a gr.State, from prepare_button), the
    controls' settings and build, which turns a corridor into a scene folder; it shows the
    scene and its download in the controls."""
    controls.reconstruct.click(
        fn=update_wrapper(partial(_handle_reconstruct, build=build), _handle_reconstruct),
        inputs=[corridor, *controls.overrides, controls.effort, controls.share],
        outputs=[controls.viewer, controls.download, controls.status],
        api_name="handle_reconstruct",
        show_progress="hidden",
        show_progress_on=[controls.viewer],
    )
