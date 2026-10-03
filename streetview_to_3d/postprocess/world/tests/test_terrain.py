import numpy as np

from streetview_to_3d.postprocess.world import terrain


def test_a_map_asked_about_nothing_answers_nothing_without_a_download():
    none = np.zeros(0)
    assert terrain.height_map()(none, none).shape == (0,)          # no roads near (Moraine Lake): no heights
    assert terrain.google_map(18)(none, none).shape == (0, 3)


def test_detail_spacing_matches_approved_distances_and_thins_smoothly():
    assert np.allclose(terrain.point_gap(np.array([0, 100, 200])), [.1, .8, 1.5])
    distances = np.linspace(0, 1000, 10001)
    gaps = terrain.point_gap(distances)
    assert np.isfinite(gaps).all() and np.all(np.diff(gaps) > 0)
    # The transition joins without a jump in spacing or slope at either end.
    for boundary in (250, 400):
        samples = terrain.point_gap(np.array([boundary - .01, boundary, boundary + .01]))
        assert abs(samples[2] - 2 * samples[1] + samples[0]) < 1e-6
