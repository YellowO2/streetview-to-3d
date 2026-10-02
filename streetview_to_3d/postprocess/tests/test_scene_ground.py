import numpy as np

from streetview_to_3d.postprocess import seams


def _ground():
    # the fill's ground: 1 m up (y down) over x 0..10, z 0..3, grey
    gx, gz = np.meshgrid(np.arange(0, 10, .05), np.arange(0, 3, .05))
    pts = np.c_[gx.ravel(), np.full(gx.size, -1.0), gz.ravel()]
    return seams.SceneGround.from_points(pts, np.full((len(pts), 3), .4))


def test_near_the_ground_its_height_and_colour_at_the_nearest_square():
    dist, h, c = _ground().at(np.array([[5.0, 1.0], [13.0, 1.5]]))
    assert dist[0] == 0 and 2.5 <= dist[1] <= 3.5
    assert np.allclose(h, 1.0) and np.allclose(c, .4)


def test_kept_beside_the_scene_and_read_back(tmp_path):
    _ground().save(tmp_path)
    dist, h, _ = seams.SceneGround.load(tmp_path).at(np.array([[5.0, 1.0]]))
    assert dist[0] == 0 and np.isclose(h[0], 1.0)


def test_a_scene_without_it_has_no_ground_and_nothing_is_near():
    none = seams.SceneGround.load("/nonexistent")
    dist, h, _ = none.at(np.array([[0.0, 0.0]]))
    assert np.isinf(dist[0]) and np.isnan(h[0]) and not none.covers(np.array([[0.0, 0.0]])).any()
