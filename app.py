"""
Gradio interface for Street Builder: reconstruct a walkable street corridor
into a 3D point cloud from Google Street View / Apple Look Around panoramas.

Run locally:  python app.py
HF Spaces:    set as app.py, add `spaces` to requirements, enable ZeroGPU.
"""

# Must be the first project import. ZeroGPU requires `spaces` (imported by
# services.pipeline_runner) to load before anything CUDA-related does.
# services.lookaround_fetch imports streetlevel's `reproject` module, which
# initializes CUDA at import time to pick its default device -- if that runs
# first, `import spaces` fails with "CUDA has been initialized before
# importing the `spaces` package." Importing pipeline_runner here, before
# the services import below, guarantees the required order regardless of
# what order the names in that later `from services import ...` get resolved in.
from services import pipeline_runner  # noqa: F401

import gradio as gr
import pillow_heif

pillow_heif.register_heif_opener()

from paths import IMAGES_DIR, SPLATS_DIR
from street_builder.map_selection.tab import build_tab as build_street_builder_tab, BRIDGE_HEAD_SCRIPT, BRIDGE_CSS

with gr.Blocks(title="Street Builder") as demo:
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
