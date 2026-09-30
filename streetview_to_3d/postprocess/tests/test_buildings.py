import numpy as np

from streetview_to_3d.postprocess import buildings
from streetview_to_3d.postprocess.roofs import Roof


def _box(x0, z0, size=10.0):
    xy = np.array([[x0, z0], [x0 + size, z0], [x0 + size, z0 + size], [x0, z0 + size], [x0, z0]], float)
    return (xy, 9.0, False, buildings.Form(Roof(xy, "flat")))


def _settle(outlines, slope):
    """The land a grid 1 m apart rising slope per metre east, plus every
    outline's corners, settled."""
    gx, gz = np.meshgrid(np.arange(-20, 60, 1.0), np.arange(-20, 40, 1.0))
    grid = np.c_[gx.ravel(), gz.ravel()]
    walls, owner = buildings.corners(outlines, lambda xy: np.full(len(xy), 1.0))
    xy = np.concatenate([grid, walls])
    own = np.r_[np.full(len(grid), -1), owner]
    return xy, own, buildings.settle(outlines, xy, xy[:, 0] * slope, own)


def test_settled_building_is_never_buried_nor_floating():
    # a house on a 10 % slope: the map's ground rises 1 m across it
    house = _box(0, 0)
    xy, own, h = _settle([house], 0.1)
    form = house[3]
    assert form.foot_m == 0.0                                   # the land's lowest along it
    assert np.allclose(h[own == 0], form.foot_m)                # level with its foot all round
    assert form.skirt_m == 0.0
    far = np.linalg.norm(xy - [30, 5], axis=1) < 1               # 20 m off: the land its own
    assert np.allclose(h[far], xy[far, 0] * 0.1)
    assert (h <= xy[:, 0] * 0.1 + 1e-9).all()                   # only ever cut down


def test_lower_neighbours_cut_is_reached_by_the_higher_ones_walls():
    # two houses side by side up a steep slope: the lower one's cut reaches the higher one
    low, high = _box(0, 0), _box(10.5, 0)
    xy, own, h = _settle([low, high], 0.3)
    assert high[3].foot_m > low[3].foot_m and high[3].skirt_m > 1.0
    assert high[3].foot_m - high[3].skirt_m <= h[own == 1].min() + 1e-9    # its walls reach the land
    # its floors counted from its own foot: the far solid's facade says so
    v, _, _, facade = buildings.solid([high], lambda p: np.zeros(len(p)), np.ones((1, 3)) * 0.8,
                                      np.array([0.0, -1.0, 0.0]))
    wall = facade[:, 1] > -1000
    assert np.isclose((-v[wall, 1]).min(), high[3].foot_m - high[3].skirt_m)
    assert np.isclose(facade[wall, 1].min(), -high[3].skirt_m)
