"""Pick streets on a map, build them in 3D and look around."""
from streetview_to_3d.main import streetview_to_3d
from streetview_to_3d.panos.main import panos
from ui.main import ui


def street_builder():
    """The whole app's page, inside a gr.Blocks: the UI, handed the engine's two jobs."""
    ui(prepare=panos, build=streetview_to_3d)
