"""HTML/iframe builders for the viewer: point clouds, scenes and splats.
`iframe()` is the one sandboxed iframe wrapper; the map picker and the
splat tab use it too.
"""
import html as html_lib

from streetview_to_3d.visualise.build_viewer import build_document


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


def viewer_document(ply_url: str | None = None, *, scene_url: str | None = None,
                    splat_url: str | None = None) -> str:
    """Build the shared modular viewer and inject what it opens first: a
    scene.json, a single point-cloud PLY, or a splat (.spz)."""
    return build_document({"plyUrl": ply_url, "sceneUrl": scene_url, "splatUrl": splat_url,
                           "editable": False})


def build_viewer(ply_url: str | None = None, *, scene_url: str | None = None,
                 splat_url: str | None = None) -> str:
    """Embed the shared viewer in Gradio, including mouse capture for flight."""
    return iframe(viewer_document(ply_url, scene_url=scene_url, splat_url=splat_url),
                  pointer_lock=True)
