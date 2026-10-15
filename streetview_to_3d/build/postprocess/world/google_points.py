"""Google's 3D Tiles points read from the google.ply the viewer saved."""
import numpy as np

from streetview_to_3d.build.postprocess.world.google.GooglePoints import MERGED, GooglePoints
from streetview_to_3d.tools.ply import read_vertices


def google_points(file):
    """GooglePoints from file, a google.ply saved by the viewer's ?google-save (or merged before)."""
    v, header = read_vertices(file)
    return GooglePoints(np.stack([v["x"], v["y"], v["z"]], 1).astype(float),
                        np.stack([v["red"], v["green"], v["blue"]], 1) / 255.0,
                        v["gap"].astype(float), np.stack([v["nx"], v["ny"], v["nz"]], 1),
                        [l[len("comment "):] for l in header.splitlines() if l.startswith("comment credit")],
                        f"comment {MERGED}\n" in header)
