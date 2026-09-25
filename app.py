"""
Gradio interface for Street Builder: reconstruct a walkable street corridor
into a 3D point cloud from Google Street View / Apple Look Around panoramas.

Run locally:  python app.py
HF Spaces:    set as app.py, add `spaces` to requirements, enable ZeroGPU.
"""

# The package first: importing it imports `spaces` before anything touches
# CUDA, which ZeroGPU requires (see streetview_to_3d/gpu.py).
from streetview_to_3d.paths import DATA_DIR
from streetview_to_3d.ui.tab import build_main_tab
from streetview_to_3d.ui.map_selection.tab import BRIDGE_HEAD_SCRIPT, BRIDGE_CSS

import gradio as gr

with gr.Blocks(title="Street Builder") as demo:
    build_main_tab()


if __name__ == "__main__":
    demo.launch(
        allowed_paths=[DATA_DIR],
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
