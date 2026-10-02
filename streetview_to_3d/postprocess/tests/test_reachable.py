import numpy as np

from streetview_to_3d.postprocess import buildings


def _square(x, y, w=10.0):
    return (np.array([[x, y], [x + w, y], [x + w, y + w], [x, y + w], [x, y]]), 10.0, None, None, [])


def test_only_buildings_with_da3_near_their_outline_get_points():
    # DA3's points along x = 0..20 at z = 0 (3D: x, height, z)
    scene = np.c_[np.linspace(0, 20, 200), np.full(200, 5.0), np.zeros(200)]
    outlines = [_square(0, 2), _square(0, 2 + buildings.REACH_M + 2), _square(100, 100)]
    assert buildings.reachable(outlines, scene).tolist() == [True, False, False]
    assert not buildings.reachable(outlines, np.zeros((0, 3))).any()
