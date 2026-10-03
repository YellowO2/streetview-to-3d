import numpy as np

from streetview_to_3d.postprocess.clean import clean


def _sheet(z, step, size=4.0, x0=0.0):
    """A square of points size across, step apart, at z (facing z)."""
    g = np.arange(0, size, step)
    return np.c_[np.repeat(g, len(g)) + x0, np.tile(g, len(g)), np.full(len(g) ** 2, z)]


def test_a_loose_sheet_off_a_wall_goes_the_wall_and_a_sparse_surface_stay():
    wall = _sheet(10.0, 0.04)
    loose = _sheet(9.5, 0.25)                                    # hanging half a metre in front of it
    far = _sheet(10.0, 0.25, x0=20.0)                            # a surface as sparse, on its own
    out = clean.sparse(np.concatenate([wall, loose, far]))
    a, b = len(wall), len(wall) + len(loose)
    assert out[:a].mean() < 0.01
    assert out[a:b].mean() > 0.9
    assert out[b:].mean() < 0.05
