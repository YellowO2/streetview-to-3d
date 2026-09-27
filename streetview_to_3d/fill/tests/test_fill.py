from types import SimpleNamespace

import numpy as np

from streetview_to_3d.fill.google import fill
from streetview_to_3d.fill.one_ground import one_ground
from streetview_to_3d.google_base.build import Base
from streetview_to_3d.postprocess.ground import normals_from_neighbours


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
    keep, surface = one_ground([cloud], cam[None], [normals_from_neighbours(cloud)], np.zeros((0, 3)))
    assert (np.hypot(surface[:, 0], surface[:, 2]) < 1).any()     # under the camera
    assert np.abs(surface[:, 0]).max() < 3.3                        # never past the walls
    assert keep[0][:len(floor)].mean() < 0.03          # DA3's floor replaced by the one ground
    assert keep[0][len(floor):].mean() > 0.95          # the walls kept (all but their foot on the floor)


def test_google_wall_joins_da3_wall_goes_up_not_sideways():
    # DA3 has a wall 8 m long, 3 m tall; Google has the same wall 0.4 m
    # further out, 20 m long and 8 m tall
    y, z = _grid((-3, 0), (-4, 4))
    da3 = np.stack([np.full(y.size, 5.0), y, z], 1)
    y, z = _grid((-8, 0), (-10, 10))
    google = np.stack([np.full(y.size, 5.4), y, z], 1)
    pano = SimpleNamespace(pos=np.array([0.0, -2.45, 0.0]), R=np.eye(3), depth=np.zeros((256, 512)))
    base = Base(panos=[pano], points=[google], ground=np.zeros((0, 3)), ground_owner=np.zeros(0, int),
                surfaces=[np.zeros(len(google), int)], planes=[(np.array([1.0, 0, 0]), 5.4)])
    added = fill(base, da3, np.tile([-1.0, 0, 0], (len(da3), 1)), da3)
    assert len(added)
    assert np.abs(added[:, 0] - 5.0).max() < 0.05     # on DA3's wall, no step
    assert np.abs(added[:, 2]).max() < 4.3             # not past DA3's wall's ends
    assert added[:, 1].min() < -6                      # but up above DA3's top


def test_fill_kept_along_all_of_da3s_wall_cut_at_its_opening():
    # DA3's wall runs 20 m with a 2 m gap it has nothing of, top to bottom
    # (an opening); Google's wall is whole. The whole length of DA3's wall
    # counts, the opening does not.
    y, z = _grid((-3, 0), (-10, 10))
    door = (z > 3) & (z < 5)
    da3 = np.stack([np.full((~door).sum(), 5.0), y[~door], z[~door]], 1)
    y, z = _grid((-8, 0), (-10, 10))
    google = np.stack([np.full(y.size, 5.0), y, z], 1)
    pano = SimpleNamespace(pos=np.array([0.0, -2.45, 0.0]), R=np.eye(3), depth=np.zeros((256, 512)))
    base = Base(panos=[pano], points=[google], ground=np.zeros((0, 3)), ground_owner=np.zeros(0, int),
                surfaces=[np.zeros(len(google), int)], planes=[(np.array([1.0, 0, 0]), 5.0)])
    added = fill(base, da3, np.tile([-1.0, 0, 0], (len(da3), 1)), da3)
    above_wall = added[(added[:, 1] < -3.5)]
    assert (above_wall[:, 2] > 7).any() and (above_wall[:, 2] < -7).any()   # over the whole wall
    assert not ((added[:, 2] > 3.3) & (added[:, 2] < 4.7)).any()   # not in the opening
