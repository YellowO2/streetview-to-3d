"""A sky sphere around the whole scene: one pano's photo, far out.

Past the scene and its backdrop there is nothing DA3 built; the photo is.
One pano -- the placed Google pano nearest the middle of the scene, where
a sphere's parallax is least wrong -- is put on a sphere RADIUS_PAD past
the furthest backdrop point (at least MIN_RADIUS_M), as points (sky.ply
beside scene.json, its "sky"), in the pano's own directions from its own camera.
Near that camera the sphere lines up with the scene exactly; further off,
it is far enough that the offset reads as distance.

The whole photo, not only what DA3 missed: from anywhere but that camera,
a sphere with the scene cut out of it would show black holes shaped like
buildings. Cars, people and poles (the same mask as the points) are
painted over from the pixels around them, so none sits in the sky.

    python -m streetview_to_3d.postprocess.sky SCENE_DIR
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

from streetview_to_3d import scene as scene_mod
from streetview_to_3d.postprocess.ply_io import read_ply, write_ply

FILENAME = "sky.ply"
WIDTH = 1024                      # points around; half that from top to bottom
MIN_RADIUS_M, RADIUS_PAD = 100.0, 1.2


def middle_node(sc):
    """The placed Google node nearest the middle of the placed cameras."""
    from streetview_to_3d.fill.paint import Camera
    nodes = [n for n in sc.nodes if n.ply and n.transform and n.rotation and n.pano.source == "google"]
    if not nodes:
        return None, None
    cams = [Camera(n) for n in nodes]
    centres = np.array([c.centre for c in cams])
    k = int(np.argmin(np.linalg.norm(centres - centres.mean(0), axis=1)))
    return nodes[k], cams[k]


def cleaned_photo(pano, scene_dir):
    """The pano's photo, WIDTH wide, cars, people and poles painted over."""
    from streetview_to_3d.services.segment import labels_path, pano_labels, pano_mask
    from streetview_to_3d.services.streetview_fetch import DA3_ONLY_ZOOM, download_pano_by_id, run_async
    path = run_async(download_pano_by_id(pano.id, zoom=DA3_ONLY_ZOOM))
    img = cv2.resize(np.asarray(Image.open(path).convert("RGB")), (WIDTH, WIDTH // 2), interpolation=cv2.INTER_AREA)
    drop = pano_mask(pano_labels(path, device="cpu", saved=labels_path(scene_dir, pano.id)))
    drop = cv2.resize(drop.astype(np.uint8), (WIDTH, WIDTH // 2), interpolation=cv2.INTER_NEAREST)
    return cv2.inpaint(img, drop, 5, cv2.INPAINT_TELEA)


def build(scene_dir, log=print):
    """Write scene_dir/FILENAME around the placed scene at scene_dir."""
    from streetview_to_3d.postprocess.backdrop import FILENAME as BACKDROP
    sc = scene_mod.Scene.load(scene_dir)
    node, cam = middle_node(sc)
    if node is None:
        log("sky: no placed Google pano")
        return
    radius = MIN_RADIUS_M
    backdrop = os.path.join(scene_dir, BACKDROP)
    if os.path.exists(backdrop):
        p, _ = read_ply(backdrop)
        if len(p):
            radius = max(radius, RADIUS_PAD * np.linalg.norm(p - cam.centre, axis=1).max())
    img = cleaned_photo(node.pano, scene_dir)
    h, w = img.shape[:2]
    # a pixel's direction in the pano (Camera.look, backwards), then the world's
    lon = ((np.arange(w) + .5) / w - .5) * 2 * np.pi
    lat = ((np.arange(h) + .5) / h - .5) * np.pi              # + is below the horizon
    lon, lat = np.meshgrid(lon, lat)
    d = np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], -1).reshape(-1, 3)
    world = d @ cam.rotation @ cam.R.T
    write_ply(os.path.join(scene_dir, FILENAME), cam.centre + radius * world, img.reshape(-1, 3) / 255.0)
    sc.sky = FILENAME
    sc.save(scene_dir)
    log(f"sky: {node.pano.id}'s photo on a {radius:.0f} m sphere, {h * w} points")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__.splitlines()[-1].strip())
    build(sys.argv[1])
