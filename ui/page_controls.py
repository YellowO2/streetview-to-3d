"""Lay out the controls under the map: the two buttons and what they show."""
import gradio as gr

from ui.Controls import Controls
from ui.tools.gallery.publish import sharing
from ui.tools.viewer import viewer
from streetview_to_3d.reconstruct.tools.gpu_seconds import DEFAULT_EFFORT, EFFORT_SECONDS_PER_SPOT


def page_controls():
    """The Controls, laid out in the order they show, inside a gr.Blocks."""
    with gr.Row(equal_height=True):
        # separate, so the GPU click is a fresh interaction: the ZeroGPU token expires on wall-clock
        prepare = gr.Button("1. Prepare")
        reconstruct = gr.Button("2. Reconstruct")
        # how hard to try: the GPU window per spot
        effort = gr.Dropdown(list(EFFORT_SECONDS_PER_SPOT), value=DEFAULT_EFFORT, label="Effort",
                             show_label=False, container=False, scale=0, min_width=130)

    # hidden per-run overrides, settable over the API (see reconstruct_button.py)
    overrides = [
        gr.Slider(50, 100, value=75, step=5, visible=False),           # keep_pct
        gr.Number(value=0, precision=0, minimum=0, visible=False),     # gpu_seconds
        gr.Number(value=0, precision=0, minimum=0, visible=False),     # view_hfov
        gr.Textbox(value="", visible=False),                           # da3_model
        gr.Textbox(value="", visible=False),                           # masker
        gr.Textbox(value="", visible=False),                           # mask_classes
        gr.Checkbox(value=True, visible=False),                        # fill
        gr.Number(value=0, minimum=0, visible=False),                  # conf_floor
    ]

    # shown where the gallery is set up
    share = gr.Checkbox(value=True, visible=sharing(),
                        label="Add the scene to the public gallery (anyone can view it)")

    status = gr.HTML()

    # shown once there is a scene to download
    download = gr.DownloadButton("Download scene (.zip)", visible=False, variant="primary")
    # a working viewer from page load: a downloaded scene can be dropped in
    scene_view = gr.HTML(viewer())
    return Controls(prepare, reconstruct, effort, overrides, share, status, download, scene_view)
