"""Button 2: read the choices, then have the scene built and shared."""
import logging
from functools import partial, update_wrapper

import gradio as gr

from ui.Controls import Controls
from ui.RunChoices import RunChoices
from ui.tools.gallery.publish import publish, sharing

logger = logging.getLogger(__name__)


def _choices(corridor, keep_pct, gpu_seconds, view_hfov, da3_model, masker, mask_classes, fill, conf_floor,
             effort, share):
    """The RunChoices read from the controls, with the viewer and download hidden while the
    scene is built (a previous run's would be mistaken for this one's); nothing prepared
    stops here, saying so."""
    if not corridor:
        raise gr.Error("Nothing prepared yet -- press \"Prepare\" first.")
    choices = RunChoices(keep_pct, gpu_seconds, view_hfov, da3_model, masker, mask_classes, fill, conf_floor,
                         effort, share)
    return (choices, gr.HTML(visible=False), gr.DownloadButton(visible=False),
            "<p>Reconstructing… This may take a few minutes.</p>")


def _failed():
    return gr.HTML(visible=True), gr.skip(), "<p>Reconstruction failed. Try again.</p>"


def _shared(folder, choices):
    """Publish the scene to the gallery when chosen and set up; the status gains its link."""
    if not (choices.share and sharing()):
        return gr.skip()
    try:
        url = publish(folder)
    except Exception as e:
        logger.warning(f"gallery: not published: {e}")
        return gr.skip()
    return f'<p>Scene ready. <a href="{url}" target="_blank">Share link</a></p>'


def reconstruct_button(corridor, controls: Controls, job, build):
    """Wire button 2: the prepared corridor (a gr.State, from prepare_button) and the
    controls' choices go to job (on_reconstruct), which has build (the engine's
    build_scene) make the scene; it is shown and offered for download, then shared."""
    choices = gr.State(None)
    folder = gr.State(None)
    shown = [controls.viewer, controls.download, controls.status]
    built = controls.reconstruct.click(
        fn=_choices, inputs=[corridor, *controls.overrides, controls.effort, controls.share],
        outputs=[choices, *shown], api_visibility="private", show_progress="hidden",
    ).success(
        fn=update_wrapper(partial(job, build=build), job), inputs=[corridor, choices], outputs=[*shown, folder],
        api_name="handle_reconstruct", show_progress="hidden", show_progress_on=[controls.viewer],
    )
    built.failure(fn=_failed, outputs=shown, api_visibility="private", show_progress="hidden")
    built.success(fn=_shared, inputs=[folder, choices], outputs=[controls.status], api_visibility="private",
                  show_progress="hidden")
