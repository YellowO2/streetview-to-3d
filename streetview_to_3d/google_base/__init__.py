"""The Google base: a clean shape of the street from Google's own depth
maps -- exact planes, walls merged across panos, one ground. See build.py
for the steps and why, fetch.py for which panos go in.

    from streetview_to_3d.google_base import gather, build
    base = build(gather(scene))          # scene: a loaded scene.json dict

or from the command line, writing the base as a scene folder:

    python -m streetview_to_3d.google_base SCENE_DIR OUT_DIR
"""
from streetview_to_3d.google_base.build import Base, build
from streetview_to_3d.google_base.fetch import GooglePano, gather

__all__ = ["Base", "GooglePano", "build", "gather"]
