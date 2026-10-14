"""Write a finished scene into a folder the viewer opens."""
import json
import os

from PIL import Image

from streetview_to_3d.models.segment import labels_path
from streetview_to_3d.postprocess.ply_io import write_mesh, write_ply


def save(scene, ground, world, folder):
    """scene.json, each node's .ply and class map, the fill's ground and the world's files
    (ground, world: None for none); returns folder."""
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
