"""Read a scene folder back into memory, with its points and class maps."""
import os

import numpy as np
from PIL import Image

from streetview_to_3d.types.Cloud import Cloud
from streetview_to_3d.types.Scene import LABELS, Scene
from streetview_to_3d.tools.ply import read_ply


def load(folder):
    """The Scene saved in folder, every node with a .ply holding its Cloud (grey without colour),
    and every node whose class map was saved (save.py's labels/) holding it again."""
    scene = Scene.load(folder)
    for n in scene.nodes:
        if n.ply:
            points, colours = read_ply(os.path.join(folder, n.ply))
            n.cloud = Cloud(points, colours if colours is not None else np.full((len(points), 3), 0.5))
        labels = os.path.join(folder, LABELS, f"{n.pano.id}.png")
        if os.path.exists(labels):
            n.labels = np.asarray(Image.open(labels))
    return scene
