"""Show a built scene: the viewer on it, its download and the status."""
import os
import zipfile

import gradio as gr

from ui.build_scene.ScenePage import ScenePage
from ui.tools.page import file_url
from ui.tools.viewer import viewer
from streetview_to_3d.types.Scene import FILENAME as SCENE_FILENAME


def scene_page(folder):
    """The ScenePage for the scene saved in folder."""
    scene_url = file_url(os.path.join(folder, SCENE_FILENAME))
    return ScenePage(gr.HTML(viewer(scene_url=scene_url), visible=True),
                     gr.DownloadButton(value=_zip(folder), visible=True), "<p>Scene ready.</p>", folder)


def _zip(run_dir):
    """The whole scene folder as save wrote it (scene.json, .ply files, water.json, life.json,
    ground.npz, labels/) as one zip the viewer opens. Stored, not compressed: point data
    barely shrinks."""
    names = sorted(os.listdir(run_dir))
    archive = os.path.join(run_dir, f"scene_{os.path.basename(run_dir)[:8]}.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        for n in names:
            if os.path.isfile(os.path.join(run_dir, n)):
                z.write(os.path.join(run_dir, n), n)
        for folder in names:
            if os.path.isdir(os.path.join(run_dir, folder)):
                for n in sorted(os.listdir(os.path.join(run_dir, folder))):
                    z.write(os.path.join(run_dir, folder, n), f"{folder}/{n}")
    return archive
