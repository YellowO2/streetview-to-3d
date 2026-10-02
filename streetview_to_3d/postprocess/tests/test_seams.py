import numpy as np

from streetview_to_3d.postprocess import seams


def test_points_turn_into_da3s_as_they_come_up_to_them():
    from scipy.spatial import cKDTree
    da3, da3_cols = np.array([[0.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])
    pts = np.array([[0.0, 0.0, 0.0], [seams.BLEND_M / 2, 0.0, 0.0], [seams.BLEND_M + 1, 0.0, 0.0]])
    cols, near, _ = seams.toward(pts, np.full((3, 3), 0.5), np.full(3, .1), cKDTree(da3), da3_cols)
    assert near[0] == 1 and 0 < near[1] < 1 and near[2] == 0       # on one, half BLEND_M off, beyond it
    assert np.allclose(cols[0], [1, 0, 0]) and np.allclose(cols[2], 0.5)
    assert np.allclose(seams.toward(pts, np.full((3, 3), .5), np.full(3, .1), None, da3_cols)[1], 0)
    # a wall 10 cm apart meeting DA3's, 40 cm apart: on it, about as few as DA3's; further off than BLEND_M, all
    g = np.arange(0, 6, .4)
    wall = np.c_[np.repeat(g, len(g)), np.tile(g, len(g)), np.zeros(len(g) ** 2)]
    f = np.arange(0, 6, .1)
    mine = np.c_[np.repeat(f, len(f)), np.tile(f, len(f)), np.zeros(len(f) ** 2)]
    _, near, keep = seams.toward(mine, np.full((len(mine), 3), .5), np.full(len(mine), .1), cKDTree(wall),
                                 np.full((len(wall), 3), .5))
    on = near > 0.99
    assert 0.03 < keep[on].mean() < 0.12                            # (10/40)^2: about one in 16
    off = np.c_[mine[:, :2], np.full(len(mine), seams.BLEND_M + 0.5)]
    assert seams.toward(off, np.full((len(off), 3), .5), np.full(len(off), .1), cKDTree(wall),
                        np.full((len(wall), 3), .5))[2].all()
    # a mesh's corners (no gap): coloured and how near, none left out
    assert seams.toward(mine, np.full((len(mine), 3), .5), None, cKDTree(wall), np.full((len(wall), 3), .5))[2].all()
