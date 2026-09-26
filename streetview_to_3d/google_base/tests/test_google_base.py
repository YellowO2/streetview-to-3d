import numpy as np

from streetview_to_3d.google_base.build import group_planes, shared_ground


def _wall(x, k, tilt=0.0):
    """a wall facing -x at x (y down, z along it), from pano k."""
    ys, zs = np.meshgrid(np.arange(-4, 0, 0.1), np.arange(0, 6, 0.1), indexing="ij")
    pts = np.stack([np.full(ys.size, x) + tilt * ys.ravel(), ys.ravel(), zs.ravel()], 1)
    n = np.array([1.0, -tilt, 0.0]) / np.hypot(1.0, tilt)
    return dict(k=k, n=n, d=float(n @ pts.mean(0)), x=pts)


def _floor(y, k):
    xs, zs = np.meshgrid(np.arange(0, 5, 0.1), np.arange(0, 6, 0.1), indexing="ij")
    pts = np.stack([xs.ravel(), np.full(xs.size, y), zs.ravel()], 1)
    n = np.array([0.0, 1.0, 0.0])
    return dict(k=k, n=n, d=float(n @ pts.mean(0)), x=pts)


def test_same_wall_from_two_panos_becomes_one_in_between():
    (n0, d0), (n1, d1) = group_planes([_wall(5.0, 0), _wall(5.1, 1)])[0]
    assert np.allclose(n0, n1) and d0 == d1
    assert 5.0 < d0 < 5.1


def test_nothing_merges_onto_a_floor():
    steep = _wall(0.0, 1, tilt=0.6)                  # a slanted surface, ~31 deg off vertical
    floor = _floor(0.0, 0)
    (nf, df), (ns, ds) = group_planes([floor, steep])[0]
    assert np.allclose(nf, floor["n"]) and np.allclose(ns, steep["n"])


def test_stacked_floors_become_one_ground():
    xs, zs = np.meshgrid(np.arange(0, 4, 0.1), np.arange(0, 4, 0.1), indexing="ij")
    a = np.stack([xs.ravel(), np.zeros(xs.size), zs.ravel()], 1)
    b = a + [0, 0.3, 0]                              # the same floor 30 cm lower, from pano 1
    cams = np.array([[0.0, -2.45, 0.0], [10.0, -2.15, 10.0]])
    G, owner = shared_ground([a, b], cams)
    # one surface: at each spot a single height, the nearer pano's (0)
    assert np.allclose(G[:, 1], 0.0, atol=0.02)
    assert (owner == 0).all()
    step = np.diff(np.unique(np.round(G[:, 0], 3)))
    assert np.allclose(step, 0.05)
