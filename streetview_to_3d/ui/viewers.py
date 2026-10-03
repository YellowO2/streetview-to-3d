"""The viewer embedded in Gradio, and `iframe()`, the one sandboxed iframe wrapper (also
used by the map picker and the HF app)."""
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
    """Gradio's URL for a file it serves (Gradio 5+)."""
    return f"/gradio_api/file={abs_path}"


def build_viewer(*, scene_url: str | None = None, splat_url: str | None = None) -> str:
    """The viewer in an iframe with mouse capture, opening a scene.json or a splat (.spz) first."""
    return iframe(build_document({"sceneUrl": scene_url, "splatUrl": splat_url, "editable": False}),
                  pointer_lock=True)
