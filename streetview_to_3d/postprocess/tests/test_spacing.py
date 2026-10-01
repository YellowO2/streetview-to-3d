import numpy as np

from streetview_to_3d.postprocess.terrain import point_gap


def test_detail_spacing_matches_approved_distances_and_thins_smoothly():
    assert np.allclose(point_gap(np.array([0, 100, 200])), [.1, .8, 1.5])
    distances = np.linspace(0, 1000, 10001)
    gaps = point_gap(distances)
    assert np.isfinite(gaps).all() and np.all(np.diff(gaps) > 0)
    # The transition joins without a jump in spacing or slope at either end.
    for boundary in (250, 400):
        samples = point_gap(np.array([boundary - .01, boundary, boundary + .01]))
        assert abs(samples[2] - 2 * samples[1] + samples[0]) < 1e-6
