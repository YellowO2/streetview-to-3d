import numpy as np

from streetview_to_3d.panos.candidates import on_open_water


def test_only_a_pano_with_water_all_round_is_out_on_it():
    # the sea east of longitude 0, the land west of it
    at_sea = on_open_water(lambda lat, lon: np.where(np.asarray(lon) > 0, 1.0, 0.0))
    m = 1 / 111320.0
    out = at_sea([0, 0, 0], [500 * m, 10 * m, -500 * m])
    assert out.tolist() == [True, False, False]   # a boat's; one on a pier 10 m out; ashore


def test_without_the_map_nothing_is_left_out():
    def gone(lat, lon):
        raise OSError("no tiles")
    assert not on_open_water(gone)([0, 1], [0, 1]).any()
