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


def test_the_satellite_as_the_panos_see_it_a_grey_stays_grey():
    rng = np.random.default_rng(0)
    # the scene's ground: panos see grass, vivid; the satellite the same grass, paler and lighter
    n = 30
    grass = np.clip([.3, .45, .15] + rng.normal(0, .03, (n, n, 3)), 0, 1)
    ground = seams.SceneGround(np.zeros(2, int), np.zeros((n, n)), grass)
    pale = lambda xy: np.clip(grass.reshape(-1, 3)[:len(xy)] * .6 + .3, 0, 1)
    fix = seams.unhazed(ground, pale)
    L = lambda c: seams.lab(np.atleast_2d(c))[0]
    grey = fix(np.array([[.5, .5, .5]]))[0]
    assert abs(L(grey)[1]) < 1 and abs(L(grey)[2]) < 1                       # a grey stays grey
    sat = np.array([[.55, .6, .5], [.8, .1, .1]])
    out = fix(sat)
    assert (seams.lab(out)[:, 0] >= seams.lab(sat)[:, 0] - .5).all()          # never darker
    chroma = lambda c: np.hypot(*seams.lab(c)[:, 1:].T)
    gain = chroma(out) / chroma(sat)
    assert gain[0] > 1.3 and gain[1] < 1.1                                    # the pale the most, the vivid hardly
    assert np.allclose(seams.unhazed(seams.SceneGround.none(), pale)(sat), sat)   # nothing to learn from: as they are


def _ground():
    # the fill's ground: 1 m up (y down) over x 0..10, z 0..3, grey
    gx, gz = np.meshgrid(np.arange(0, 10, .05), np.arange(0, 3, .05))
    pts = np.c_[gx.ravel(), np.full(gx.size, -1.0), gz.ravel()]
    return seams.SceneGround.from_points(pts, np.full((len(pts), 3), .4))


def test_near_the_ground_its_height_and_colour_at_the_nearest_square():
    dist, h, c = _ground().at(np.array([[5.0, 1.0], [13.0, 1.5]]))
    assert dist[0] == 0 and 2.5 <= dist[1] <= 3.5
    assert np.allclose(h, 1.0) and np.allclose(c, .4)


def test_kept_beside_the_scene_and_read_back(tmp_path):
    _ground().save(tmp_path)
    dist, h, _ = seams.SceneGround.load(tmp_path).at(np.array([[5.0, 1.0]]))
    assert dist[0] == 0 and np.isclose(h[0], 1.0)


def test_a_scene_without_it_has_no_ground_and_nothing_is_near():
    none = seams.SceneGround.load("/nonexistent")
    dist, h, _ = none.at(np.array([[0.0, 0.0]]))
    assert np.isinf(dist[0]) and np.isnan(h[0]) and not none.covers(np.array([[0.0, 0.0]])).any()
