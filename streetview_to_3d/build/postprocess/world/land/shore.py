"""The land shaped into a shore where it meets the water."""
from dataclasses import replace

from streetview_to_3d.build.postprocess.world.Water import QUAY_M
from streetview_to_3d.build.postprocess.world.tools.osm_roads import near


def shore(points, water, roads):
    """LandPoints with their heights carved into the water's bed and banks (Water.carve); no
    bank by the roads (Roads): quays."""
    return replace(points, height=water.carve(points.xy, points.height, near(roads.lines(), QUAY_M)))
