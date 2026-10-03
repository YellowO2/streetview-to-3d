import numpy as np

from streetview_to_3d.postprocess.clean import blobs
from streetview_to_3d.postprocess.clean.blobs import loose_bits


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


def test_a_far_wall_stays_only_if_a_pano_near_it_puts_it_there_too():
    g = np.arange(0, 4, 0.05)
    wall = lambda x: np.c_[np.full(g.size ** 2, x), np.repeat(g, g.size), np.tile(g, g.size)]   # a wall x m east
    real = wall(15.0)
    a = np.concatenate([real, wall(16.0)])          # A, 15 m off, sees the real wall, and makes one up behind it
    near = real + 0.05                              # B, 8 m from the wall, sees the real one (a little off)
    sure = blobs.confirmed([a, near], [np.zeros(3), np.array([8.0, 2.0, 2.0])])
    assert sure[0][:len(real)].all()                # the real wall: B near it puts it there too
    assert not sure[0][len(real):].any()            # the made-up one: A's alone
    # across a river: C beside A, as far off, guesses the same made-up wall -- no pano near it, no confirming
    c = wall(16.0) + 0.02
    sure = blobs.confirmed([a, c], [np.zeros(3), np.array([0.0, 0.0, 3.0])])
    assert not sure[0].any() and not sure[1].any()
