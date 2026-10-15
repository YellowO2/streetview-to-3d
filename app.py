"""Street Builder's Gradio app: Street View panoramas to a 3D scene. Run: python app.py"""
import logging

# the package's progress lines, message only, on stderr: the Space's logs show them
logging.basicConfig(level=logging.INFO, format="%(message)s")
for chatty in ("httpx", "httpx2", "numexpr"):   # a line per request Gradio makes, NumExpr's thread count
    logging.getLogger(chatty).setLevel(logging.WARNING)

# the package first: it imports `spaces` before anything touches CUDA (streetview_to_3d/tools/on_gpu.py)
from streetview_to_3d.tools.paths import DATA_DIR
from main import street_builder
from ui import BRIDGE_CSS, BRIDGE_HEAD_SCRIPT, pano_api

import gradio as gr

with gr.Blocks(title="Street Builder") as demo:
    street_builder()
    pano_api()   # the masker and DA3 on one pano, API only


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
