import numpy as np

from streetview_to_3d.fill.one_ground import grounds, one_ground


def _grid(a, b, step=0.05):
    u, v = np.meshgrid(np.arange(*a, step), np.arange(*b, step), indexing="ij")
    return u.ravel(), v.ravel()


def test_blind_disc_filled_up_to_the_walls():
    # an alley 6 m wide: DA3 sees its floor from 4 m out, nothing under the camera
    x, z = _grid((-3, 3), (-15, 15), 0.1)
    far = np.hypot(x, z) > 4
    floor = np.stack([x[far], np.zeros(far.sum()), z[far]], 1)
    y, z = _grid((-5, 0), (-15, 15), 0.1)
    walls = np.concatenate([np.stack([np.full(y.size, s * 3.0), y, z], 1) for s in (-1, 1)])
    cloud = np.concatenate([floor, walls])
    cam = np.array([0.0, -2.45, 0.0])
    keep, surface = one_ground([cloud], cam[None], grounds([cloud], cam[None]))
    assert (np.hypot(surface[:, 0], surface[:, 2]) < 1).any()     # under the camera
    assert np.abs(surface[:, 0]).max() < 3.3                        # never past the walls
    assert keep[0][:len(floor)].mean() < 0.03          # DA3's floor replaced by the one ground
    assert keep[0][len(floor):].mean() > 0.95          # the walls kept (all but their foot on the floor)
