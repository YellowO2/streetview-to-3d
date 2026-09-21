"""HTML/iframe builders for the point-cloud viewer and result
download links. `iframe()` is the canonical sandboxed iframe wrapper --
reconstruct/map_ui.py imports it from here too, rather than keeping its
own copy.
"""
import html as html_lib
import json
from pathlib import Path


def iframe(srcdoc: str, aspect: str = "16/9", *, pointer_lock: bool = False) -> str:
    escaped = html_lib.escape(srcdoc, quote=True)
    permissions = "allow-scripts allow-same-origin"
    if pointer_lock:
        permissions += " allow-pointer-lock"
    return (
        f'<iframe srcdoc="{escaped}" sandbox="{permissions}" '
        f'style="width:100%;aspect-ratio:{aspect};border:none;border-radius:8px;background:#000">'
        "</iframe>"
    )


def file_url(abs_path: str) -> str:
    """Build Gradio's file-serving URL. /gradio_api/file= is the route in Gradio 5+."""
    return f"/gradio_api/file={abs_path}"


VIEWER_PATH = Path(__file__).resolve().parents[1] / "visualise" / "viewer.html"


def pointcloud_document(ply_url: str | None = None, *, scene_url: str | None = None) -> str:
    """Read the directly-openable viewer; inject only its initial asset URL."""
    doc = VIEWER_PATH.read_text(encoding="utf-8")
    # JSON is in a script element: escape '<' so URLs cannot close the tag.
    config = json.dumps({"plyUrl": ply_url, "sceneUrl": scene_url}).replace("<", "\\u003c")
    return doc.replace(
        '<script id="viewer-config" type="application/json">{}</script>',
        f'<script id="viewer-config" type="application/json">{config}</script>',
        1,
    )


def build_pointcloud_viewer(ply_url: str | None = None, *, scene_url: str | None = None) -> str:
    """Embed the shared viewer in Gradio, including mouse capture for flight."""
    return iframe(pointcloud_document(ply_url, scene_url=scene_url), pointer_lock=True)


def summary(lines, note=None) -> str:
    """A plain result list, with an optional line on what to do next."""
    items = "".join(f'<li style="margin:4px 0">{html_lib.escape(str(l))}</li>'
                    for l in lines)
    tail = (f'<p style="color:#aaa;margin:8px 0 0;font:13px sans-serif">'
            f'{html_lib.escape(note)}</p>' if note else "")
    return (f'<div style="padding:12px;background:#1e1e2e;border-radius:8px">'
            f'<ul style="margin:0;padding-left:20px;font:13px sans-serif;'
            f'list-style:none;color:#ddd">{items}</ul>{tail}</div>')
