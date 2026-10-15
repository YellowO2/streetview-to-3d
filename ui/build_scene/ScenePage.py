"""What the page shows once a scene is built, and where it is."""
from typing import NamedTuple


class ScenePage(NamedTuple):
    viewer: object      # the viewer, opening the scene
    download: object    # the scene's zip, to download
    status: str
    folder: str         # the scene's folder, for sharing it
