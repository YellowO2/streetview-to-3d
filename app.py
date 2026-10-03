"""Street Builder's Gradio app: Street View panoramas to a 3D scene. Run: python app.py"""

# the package first: it imports `spaces` before anything touches CUDA (streetview_to_3d/models/gpu.py)
from streetview_to_3d.common.paths import DATA_DIR
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
        ssr_mode=False,  # HF's default SSR Node proxy was the suspect when the Space hung restarting
    )
