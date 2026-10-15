"""Wrap a page in a sandboxed iframe, for Gradio and the map picker."""
import html as html_lib


def iframe(srcdoc: str, aspect: str = "16/9", *, pointer_lock: bool = False) -> str:
    """srcdoc in a sandboxed iframe (scripts allowed; pointer_lock: mouse capture and downloads too)."""
    escaped = html_lib.escape(srcdoc, quote=True)
    permissions = "allow-scripts allow-same-origin"
    if pointer_lock:
        permissions += " allow-pointer-lock allow-downloads"
    return (
        f'<iframe srcdoc="{escaped}" sandbox="{permissions}" '
        f'style="width:100%;aspect-ratio:{aspect};border:none;border-radius:8px;background:#000">'
        "</iframe>"
    )
