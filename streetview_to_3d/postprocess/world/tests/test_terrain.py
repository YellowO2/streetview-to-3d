import numpy as np

from streetview_to_3d.postprocess.world import terrain


def test_a_map_asked_about_nothing_answers_nothing_without_a_download():
    none = np.zeros(0)
    assert terrain.height_map()(none, none).shape == (0,)          # no roads near (Moraine Lake): no heights
    assert terrain.google_map(18)(none, none).shape == (0, 3)
