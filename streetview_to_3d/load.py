"""Read a scene folder back into memory, each node with its points."""
import os

import numpy as np

from streetview_to_3d.types.Cloud import Cloud
from streetview_to_3d.types.Scene import Scene
from streetview_to_3d.tools.ply import read_ply


def load(folder):
    """The Scene saved in folder, every node with a .ply holding its Cloud (grey without colour)."""
    scene = Scene.load(folder)
    for n in scene.nodes:
        if n.ply:
            points, colours = read_ply(os.path.join(folder, n.ply))
            n.cloud = Cloud(points, colours if colours is not None else np.full((len(points), 3), 0.5))
    return scene
