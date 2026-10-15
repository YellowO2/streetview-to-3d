"""Mark the one ground's points that their painting pano calls road."""
import numpy as np

from streetview_to_3d.build_scene.postprocess.fill.PaintedGround import PaintedGround


def on_road(ground, cameras):
    """The PaintedGround with road set: per point, whether the pano that painted it calls it road
    (life.py keeps cars to it)."""
    road = np.zeros(len(ground.points), bool)
    for k, camera in enumerate(cameras):
        mine = ground.painter == k
        if mine.any():
            road[mine] = camera.classed(ground.points[mine], ("road",), False)
    return PaintedGround(ground.points, ground.colours, ground.painter, road)
