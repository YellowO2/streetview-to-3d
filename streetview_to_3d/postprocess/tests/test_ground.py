import numpy as np

from streetview_to_3d.postprocess.ground import ground


def _patch(x0, x1, z0, z1, y, step=0.1, slope=0.0):
    """a horizontal (or sloped along x) patch, y down; normals up."""
    xs, zs = np.meshgrid(np.arange(x0, x1, step), np.arange(z0, z1, step), indexing="ij")
    x, z = xs.ravel(), zs.ravel()
    return np.stack([x, y - slope * (x - x0), z], 1)


def _up(n):
    return np.tile([0.0, -1.0, 0.0], (n, 1))


def test_slope_uphill_stays_ground():
    # camera at 2.45 m over a street that rises 1 m per 10 m for 20 m
    street = _patch(-2, 20, -3, 3, 0.0, slope=0.1)
    cam = np.array([0.0, -2.45, 0.0])
    g = ground(street, cam, normals=_up(len(street)))
    assert g.all()          # still ground where it is within 1 m of camera height


def test_things_on_the_ground_are_not_ground():
    street = _patch(-5, 5, -5, 5, 0.0)
    car_roof = _patch(2, 4, -1, 1, -1.5)         # 1.5 m up, over the street
    planter = _patch(-4, -3, 2, 4, -0.8)          # a 0.8 m block beside it
    x = np.concatenate([street, car_roof, planter])
    g = ground(x, np.array([0.0, -2.45, 0.0]), normals=_up(len(x)))
    n0, n1 = len(street), len(street) + len(car_roof)
    assert g[:n0][np.abs(street[:, 0] - 3) > 1.5].all()
    assert not g[n0:n1].any()
    assert not g[n1:].any()


def test_walls_are_not_ground():
    street = _patch(-5, 5, -5, 5, 0.0)
    ys, zs = np.meshgrid(np.arange(-3, 0, 0.1), np.arange(-5, 5, 0.1), indexing="ij")
    wall = np.stack([np.full(ys.size, 5.0), ys.ravel(), zs.ravel()], 1)
    x = np.concatenate([street, wall])
    n = np.concatenate([_up(len(street)), np.tile([-1.0, 0.0, 0.0], (len(wall), 1))])
    g = ground(x, np.array([0.0, -2.45, 0.0]), normals=n)
    assert not g[len(street):].any()
