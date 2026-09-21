"""Copy the shared viewer to a convenient location (no build step needed).

The source visualise/viewer.html can also be opened directly. Both versions
load Three.js from a CDN, so an internet connection is needed.

Usage:
    python -m visualise.export_standalone_viewer --out ~/Downloads/viewer.html
"""
import argparse
from pathlib import Path
import shutil

from ui.viewers import VIEWER_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="~/Downloads/viewer.html")
    args = parser.parse_args()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path != VIEWER_PATH.resolve():
        shutil.copyfile(VIEWER_PATH, out_path)
    print(f"Viewer: {out_path} -- open it directly, then drop a .ply onto it.")


if __name__ == "__main__":
    main()
