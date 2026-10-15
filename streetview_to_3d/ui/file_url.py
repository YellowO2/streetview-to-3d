"""Give the URL Gradio serves a file on disk at."""


def file_url(abs_path: str) -> str:
    """Gradio's URL for a file it serves (Gradio 5+)."""
    return f"/gradio_api/file={abs_path}"
