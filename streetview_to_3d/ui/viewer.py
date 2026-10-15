"""Embed the point-cloud viewer in the page, opening a scene or a splat."""
from streetview_to_3d.ui.tools.page import iframe
from streetview_to_3d.visualise.build_viewer import build_viewer


def viewer(*, scene_url: str | None = None, splat_url: str | None = None) -> str:
    """The viewer in an iframe with mouse capture, opening a scene.json or a splat (.spz) first."""
    return iframe(build_viewer({"sceneUrl": scene_url, "splatUrl": splat_url, "editable": False}),
                  pointer_lock=True)
