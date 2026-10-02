import numpy as np

from streetview_to_3d.postprocess import seams


def test_the_road_meets_the_scene_at_its_ground_not_a_wall_at_its_edge():
    # DA3's ground at height 0 over x 0..10 (y is down), and past it only a wall's points, 7 m up
    g = np.array([[x + .5, 0.0, z + .5] for x in range(10) for z in range(3) for _ in range(4)])
    wall = np.array([[10.5, -7.0 - k * .1, z + .5] for z in range(3) for k in range(4)])
    pts = np.concatenate([g, wall])
    land = lambda xy: np.zeros(len(xy))
    _, ground, _ = seams.Footprint(pts, np.full((len(pts), 3), .5), land).at(np.array([[13.0, 1.5]]))
    assert abs(ground[0]) < 0.5
    _, blind, _ = seams.Footprint(pts, np.full((len(pts), 3), .5)).at(np.array([[13.0, 1.5]]))
    assert blind[0] > 6                                  # without the land: the wall's, as it was
