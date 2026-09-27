import numpy as np

from streetview_to_3d.postprocess.blobs import loose_bits


def test_floating_ball_removed_wall_and_what_touches_it_kept():
    y, z = np.meshgrid(np.arange(-3, 0, 0.05), np.arange(-5, 5, 0.05))
    wall = np.stack([np.full(y.size, 5.0), y.ravel(), z.ravel()], 1)
    rng = np.random.default_rng(0)
    ball = rng.normal(0, 0.1, (300, 3)) + [2.0, -4.0, 0.0]       # 3 m from the wall, in the air
    sign = rng.uniform(0, 1, (400, 3)) * [0.1, 0.5, 0.5] + [4.8, -2.0, 1.0]   # small, but on the wall
    loose = loose_bits(np.concatenate([wall, ball, sign]))
    assert not loose[:len(wall)].any()
    assert loose[len(wall):len(wall) + len(ball)].all()
    assert not loose[len(wall) + len(ball):].any()
