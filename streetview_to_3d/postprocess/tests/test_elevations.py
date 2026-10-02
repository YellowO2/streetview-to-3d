import numpy as np

from streetview_to_3d.postprocess import elevations


def test_one_height_a_square_and_a_bridge_left_out():
    # a street east along y = 0 climbing 5 m in 100, a pano every 10 m; a bridge's 6 m up over 200-225 m
    x = np.arange(0, 400, 10.0)
    xy = np.c_[x, np.zeros_like(x)]
    slope = 0.05 * x
    at, h = elevations.thinned(xy, slope + np.where((x >= 200) & (x < 225), 6.0, 0.0))
    assert len(at) == len(np.unique(np.floor(x / elevations.CELL_M))) - 1
    assert np.allclose(h, 0.05 * at[:, 0], atol=0.3) and not ((at[:, 0] >= 200) & (at[:, 0] < 225)).any()


def test_nothing_around():
    at, f = elevations.thinned(np.zeros((0, 2)), np.zeros(0))
    assert at.shape == (0, 2) and f.shape == (0,)


def test_panos_on_bridges_and_in_tunnels_are_not_the_ground():
    way = lambda tags, x: {"type": "way", "tags": dict(highway="primary", **tags), "geometry": [(x, -50), (x, 50)]}
    elements = [way({"tunnel": "yes"}, 0), way({"bridge": "yes"}, 100), way({}, 200)]
    xy = np.array([[0, 0], [100, 10], [200, 0], [300, 0]], float)
    on = elevations.on_ground(xy, elements, lambda g: np.array(g, float))
    assert on.tolist() == [False, False, True, True]
