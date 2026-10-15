"""Build a prepared corridor's scene with the chosen settings, and show it."""
from ui.on_reconstruct.run_settings import run_settings
from ui.on_reconstruct.scene_page import scene_page


def on_reconstruct(corridor, choices, *, build):
    """What the page shows once build (the engine's build_scene) has made corridor's
    scene: reconstruct (GPU), then place and fill (CPU)."""
    settings = run_settings(corridor, choices)
    folder = build(corridor, settings)
    page = scene_page(folder)
    return page
