"""Extracts street_builder's own point-cloud viewer (viewers.py's
build_pointcloud_viewer) as a plain, standalone .html file -- no Gradio,
no app.py, no server. It's already a complete self-contained HTML
document under the hood; Gradio just wraps it in an <iframe> for
embedding. Drag-and-drop reads the dropped file via FileReader (not a
server fetch), so this works opened directly via file:// -- just
double-click it.

Usage:
    python -m tests.export_standalone_viewer --out ~/Downloads/viewer.html
"""
import argparse
import os

from visualise import viewers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="~/Downloads/viewer.html")
    args = parser.parse_args()

    original_iframe = viewers.iframe
    viewers.iframe = lambda srcdoc, aspect="16/9": srcdoc  # unwrap: capture the raw doc instead of an <iframe srcdoc=...>
    try:
        doc = viewers.build_pointcloud_viewer()
    finally:
        viewers.iframe = original_iframe

    out_path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(doc)
    print(f"Wrote {out_path} -- open it directly (file://), then drag a .ply onto it.")


if __name__ == "__main__":
    main()
