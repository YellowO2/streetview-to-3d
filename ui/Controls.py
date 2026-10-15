"""The page's controls under the map: its two buttons and what they show."""
from dataclasses import dataclass

import gradio as gr


@dataclass
class Controls:
    """prepare and reconstruct: buttons 1 and 2; effort: how hard to try (the GPU window
    per spot); overrides: the hidden per-run settings, in RunChoices'
    order; share: whether to publish to the gallery; status: what
    the last click says; download: the scene's zip, once there is one; viewer: the scene."""
    prepare: gr.Button
    reconstruct: gr.Button
    effort: gr.Dropdown
    overrides: list
    share: gr.Checkbox
    status: gr.HTML
    download: gr.DownloadButton
    viewer: gr.HTML
