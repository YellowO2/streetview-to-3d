import numpy as np
import shapely

from streetview_to_3d.postprocess import water


def test_a_busy_overpass_leaves_the_shore_to_the_jrc_map():
    # a shoreline along x = 0 across the box; east of it is water
    shore = {"type": "way", "tags": {"natural": "water"}, "geometry": [(0, -200), (0, 200)]}
    to_xy = lambda g: np.array(g, float)

    def busy(xy):
        raise TimeoutError("Overpass busy")
    out = water.outline([shore], to_xy, 100, busy, lambda xy: np.asarray(xy)[:, 0] > 0)
    assert out.contains(shapely.Point(50, 0)) and not out.contains(shapely.Point(-50, 0))
