"""Build a prepared corridor's scene with the chosen settings, and show it."""
from ui.build_scene.run_settings import run_settings
from ui.build_scene.scene_page import scene_page


def build_scene(corridor, choices, *, build):
    """What the page shows once build (the engine's build) has made corridor's
    scene: reconstruct (GPU), then place and fill (CPU)."""
    settings = run_settings(corridor, choices)
    folder = build(corridor, settings)
    page = scene_page(folder)
    return page
