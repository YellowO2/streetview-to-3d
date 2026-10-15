"""Pick streets on a map, build them in 3D and look around.

Run: python main.py
"""
from streetview_to_3d.build_scene import build_scene  # first: it imports `spaces` before anything touches CUDA
from streetview_to_3d.gather_panoramas import gather_panoramas
from ui.main import ui


def street_builder():
    """The whole app's page, inside a gr.Blocks: the UI, handed the engine's two jobs."""
    ui(prepare=gather_panoramas, build=build_scene)


if __name__ == "__main__":
    import logging

    import gradio as gr

    from streetview_to_3d.tools.paths import DATA_DIR
    from ui import BRIDGE_CSS, BRIDGE_HEAD_SCRIPT

    # the package's progress lines, message only, on stderr: the Space's logs show them
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for chatty in ("httpx", "httpx2", "numexpr"):   # a line per request Gradio makes, NumExpr's thread count
        logging.getLogger(chatty).setLevel(logging.WARNING)

    with gr.Blocks(title="Street Builder") as demo:
        street_builder()
    demo.launch(
        allowed_paths=[DATA_DIR],
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Default(),
        css=".no-pad { padding-left: 0 !important; padding-right: 0 !important; } " + BRIDGE_CSS,
        head=BRIDGE_HEAD_SCRIPT,
        ssr_mode=False,  # HF's default SSR Node proxy was the suspect when the Space hung restarting
    )
