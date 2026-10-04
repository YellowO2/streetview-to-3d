import numpy as np

from streetview_to_3d.postprocess import seams


def test_a_map_point_is_as_big_as_da3s_on_them_and_bigger_further_off_its_spacing_following():
    d = np.array([0, 100, 200])
    assert np.allclose(seams.size_at(d), [seams.DA3_M, seams.DA3_M + 100 * seams.GROW, seams.DA3_M + 200 * seams.GROW])
    assert np.allclose(seams.spacing_at(d) * seams.RATIO, seams.size_at(d))
    assert np.allclose(seams.spacing_at(d), [.08, .78, 1.48], atol=.01)      # the spacing approved, near enough


def test_points_take_da3s_colour_as_they_come_up_to_them():
    from scipy.spatial import cKDTree
    da3, da3_cols = np.array([[0.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])
    pts = np.array([[0.0, 0.0, 0.0], [seams.BLEND_M / 2, 0.0, 0.0], [seams.BLEND_M + 1, 0.0, 0.0]])
    cols, near = seams.toward(pts, np.full((3, 3), 0.5), cKDTree(da3), da3_cols)
    assert near[0] == 1 and 0 < near[1] < 1 and near[2] == 0       # on one, half BLEND_M off, beyond it
    assert np.allclose(cols[0], [1, 0, 0]) and np.allclose(cols[2], 0.5)
    assert np.allclose(seams.toward(pts, np.full((3, 3), .5), None, da3_cols)[1], 0)


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
