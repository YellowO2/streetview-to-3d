from types import SimpleNamespace

import numpy as np

from streetview_to_3d.postprocess.place import CAM_H, fit_piece, photo_from_world

ORIGIN = (1.35, 103.68)


def _node(e, n, elev, heading, pitch, roll, world_from_da3, scale):
    """A camera at (e, n, elev) pointing by (heading, pitch, roll), as DA3
    would have it in a frame world_from_da3 away, scale units smaller."""
    pano = SimpleNamespace(lat=ORIGIN[0] + n / 111320.0,
                           lon=ORIGIN[1] + e / (111320.0 * np.cos(np.radians(ORIGIN[0]))),
                           elevation=elev, heading=heading, pitch=pitch, roll=roll)
    world = np.array([e, -(elev + CAM_H), n])
    R, t = world_from_da3
    position = R.T @ (world - t) / scale
    rotation = photo_from_world(pano) @ R            # DA3 frame -> photo
    return SimpleNamespace(pano=pano, position=position.tolist(), rotation=rotation.tolist())


def test_a_tilted_piece_on_a_slope_comes_back_exactly():
    # three cameras up a 5 deg slope, the car pitched with it; DA3's frame
    # turned and tilted arbitrarily
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("yxz", [40, 7, -3], degrees=True).as_matrix()
    t = np.array([3.0, -20.0, -5.0])
    nodes = [_node(0, 10 * k, 20 + 10 * k * np.tan(np.radians(5)), 0.0, np.radians(-5), np.radians(1),
                   (R, t), 1.2) for k in range(3)]
    T, off_m, off_deg = fit_piece(nodes, 1.2, ORIGIN)
    assert np.allclose(T[:3, :3], 1.2 * R, atol=1e-6) and np.allclose(T[:3, 3], t, atol=1e-6)
    assert off_m < 1e-6 and off_deg < 1e-4


def test_a_lone_pano_is_turned_by_its_own_orientation():
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("yxz", [-100, 3, 8], degrees=True).as_matrix()
    node = _node(4, -2, 15, np.radians(30), np.radians(6), np.radians(-2), (R, np.zeros(3)), 1.3)
    T, off_m, off_deg = fit_piece([node], 1.3, ORIGIN)
    assert np.allclose(T[:3, :3], 1.3 * R, atol=1e-6)
    assert off_m < 1e-6 and off_deg < 1e-4
