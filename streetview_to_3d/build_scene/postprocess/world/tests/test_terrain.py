import numpy as np

from streetview_to_3d.build_scene.postprocess.world.HeightMap import HeightMap
from streetview_to_3d.build_scene.postprocess.world.Satellite import Satellite


def test_a_map_asked_about_nothing_answers_nothing_without_a_download():
    none = np.zeros(0)
    assert HeightMap()(none, none).shape == (0,)          # no roads near (Moraine Lake): no heights
    assert Satellite.google_map(18)(none, none).shape == (0, 3)
