"""Write a finished scene into a folder the viewer opens."""
import json
import logging
import os

from PIL import Image

from streetview_to_3d.common.paths import new_run_dir
from streetview_to_3d.models.segment import labels_path
from streetview_to_3d.postprocess.ply_io import write_mesh, write_ply

logger = logging.getLogger(__name__)


def save(scene, ground, world, folder=None):
    """scene.json, each node's .ply and class map, the fill's ground and the world's files
    (ground, world: None for none), in folder or a fresh run folder; returns the folder."""
    if folder is None:
        folder = new_run_dir()
        logger.info(f"run dir: {folder}")
    os.makedirs(folder, exist_ok=True)
    for n in scene.nodes:
        if n.cloud is not None and n.ply:
            c = n.cloud
            write_ply(os.path.join(folder, n.ply), c.points, c.colours, sway=c.sway)
        if n.photo is not None:
            path = labels_path(folder, n.pano.id)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            Image.fromarray(n.photo[1]).save(path)
    if ground is not None:
        ground.save(folder)
    if world is not None:
        for name, mesh in world.meshes.items():
            write_mesh(os.path.join(folder, name), **mesh)
        for name, points in world.points.items():
            write_ply(os.path.join(folder, name), **points)
        for name, data in world.json.items():
            with open(os.path.join(folder, name), "w") as f:
                json.dump(data, f, separators=(",", ":"))
    scene.save(folder)
    return folder
