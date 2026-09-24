"""HTML/iframe builders for the point-cloud viewer. `iframe()` is the canonical sandboxed iframe wrapper --
reconstruct/map_ui.py imports it from here too, rather than keeping its
own copy.
"""
import html as html_lib

from visualise.build_viewer import build_document


def iframe(srcdoc: str, aspect: str = "16/9", *, pointer_lock: bool = False) -> str:
    escaped = html_lib.escape(srcdoc, quote=True)
    permissions = "allow-scripts allow-same-origin"
    if pointer_lock:
        permissions += " allow-pointer-lock allow-downloads"
    return (
        f'<iframe srcdoc="{escaped}" sandbox="{permissions}" '
        f'style="width:100%;aspect-ratio:{aspect};border:none;border-radius:8px;background:#000">'
        "</iframe>"
    )


def file_url(abs_path: str) -> str:
    """Build Gradio's file-serving URL. /gradio_api/file= is the route in Gradio 5+."""
    return f"/gradio_api/file={abs_path}"


def pointcloud_document(ply_url: str | None = None, *, scene_url: str | None = None) -> str:
    """Build the shared modular viewer and inject its initial asset URL."""
    return build_document({"plyUrl": ply_url, "sceneUrl": scene_url, "editable": False})


def build_pointcloud_viewer(ply_url: str | None = None, *, scene_url: str | None = None) -> str:
    """Embed the shared viewer in Gradio, including mouse capture for flight."""
    return iframe(pointcloud_document(ply_url, scene_url=scene_url), pointer_lock=True)
