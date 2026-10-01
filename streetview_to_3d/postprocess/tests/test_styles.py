import numpy as np
from shapely.geometry import Polygon

from streetview_to_3d.postprocess import buildings, styles


def _element(i, lat, lon, size_m, tags):
    """A square way size_m across at (lat, lon)."""
    d = size_m / 111_000
    ring = [(lat, lon), (lat, lon + d), (lat + d, lon + d), (lat + d, lon), (lat, lon)]
    return {"type": "way", "id": i, "tags": tags, "geometry": [{"lat": a, "lon": b} for a, b in ring]}


def _outlines(lat, lon, *buildings_):
    to_xy = lambda g: np.array([[(p["lon"] - lon) * 111_000, (p["lat"] - lat) * 111_000] for p in g])
    return buildings.outlines([_element(i, lat, lon + i * 0.001, s, t) for i, (s, t) in enumerate(buildings_)],
                              to_xy)


def test_regions():
    assert styles.region(36.24, 137.97) == "japan"            # Matsumoto
    assert styles.region(35.10, 129.04) == "korea"            # Busan
    assert styles.region(59.33, 18.07) == "europe_north"      # Stockholm
    assert styles.region(41.90, 12.48) == "europe_south"      # Rome
    assert styles.region(40.74, -73.99) == "north_america"    # New York
    assert styles.region(13.75, 100.49) == "southeast_asia"   # Bangkok
    assert styles.region(28.61, 77.21) == "south_asia"        # Delhi
    assert styles.region(31.23, 121.47) == "east_asia"        # Shanghai
    assert styles.region(-33.9, 151.2) == "other"             # Sydney


def test_a_japanese_castle_is_a_stone_base_and_tiers_each_smaller():
    out = _outlines(36.2387, 137.9694, (20, {"building": "yes", "building:levels": "5", "historic": "castle",
                                             "castle_type": "shiro"}))
    base, *tiers = out
    assert base[3].tags["building:part"] == "base" and base[3].roof.shape == "flat"
    assert len(tiers) == 5 and all(t[3].stand_on is base[3] for t in tiers)
    areas = [Polygon(t[0]).area for t in tiers]
    assert all(a > b for a, b in zip(areas, areas[1:]))                       # each smaller
    assert all(t[3].roof.shape == "hipped" for t in tiers)
    bases = [t[3].base_m for t in tiers]
    assert bases[0] >= base[1] - 1e-9 and all(a < b for a, b in zip(bases, bases[1:]))   # stacked
    assert all(t[1] > t[3].base_m + t[3].roof.height for t in tiers)          # walls under each roof
    # standing on the base's foot, wherever their own corners are
    xy = np.concatenate([o[0] for o in out])
    owner = np.concatenate([np.full(len(o[0]), i) for i, o in enumerate(out)])
    buildings.settle(out, xy, xy[:, 0] * 0.05, owner)
    assert all(t[3].foot_m == base[3].foot_m for t in tiers)


def test_untagged_roofs_by_region_tagged_ones_stand():
    small, big, office = (8, {"building": "yes"}), (40, {"building": "yes"}), (8, {"building": "office"})
    tagged = (8, {"building": "yes", "roof:shape": "flat"})
    japan = _outlines(36.24, 137.97, small, big, office, tagged)
    assert [o[3].roof.shape for o in japan] == ["hipped", "flat", "flat", "flat"]
    assert japan[0][1] == styles.HOUSE_M                                       # a house, not a 12 m block
    assert 0 < japan[0][3].roof.height < japan[0][1]
    stockholm = _outlines(59.33, 18.07, small, big)
    assert [o[3].roof.shape for o in stockholm] == ["gabled", "flat"]
    assert np.isclose(stockholm[0][3].roof.height, np.tan(np.radians(38)) * 4, atol=0.1)


def test_church_tower_mosque_dome_and_minaret():
    church = _outlines(48.85, 2.35, (30, {"building": "church"}))
    assert len(church) == 2 and church[0][3].roof.shape == "gabled"
    tower = church[1]
    assert tower[3].part and tower[3].roof.shape == "pyramidal" and tower[1] > church[0][1] * 1.5
    assert tower[0][:, 0].mean() < church[0][0][:, 0].mean()                 # at its west end
    mosque = _outlines(41.0, 28.97, (30, {"building": "mosque"}))
    shapes = sorted(o[3].roof.shape for o in mosque)
    assert shapes == ["cone", "dome", "flat"]
    assert max(o[1] for o in mosque) > 2 * mosque[0][1]                        # the minaret stands tallest


def test_east_asian_facades_have_no_european_ornament():
    from streetview_to_3d.postprocess.facade_geometry import profile_for
    house = {"building": "house", "style:region": "japan"}
    assert profile_for(house, 7, 80).bay == 1.8 and not profile_for(house, 7, 80).pilasters
    old = {"building": "yes", "start_date": "1890", "style:region": "japan"}
    assert not profile_for(old, 15, 400).pilasters
    assert profile_for({"building": "yes", "start_date": "1890"}, 15, 400).name == "historic_urban"


def test_a_sharp_satellite_roof_keeps_its_colour_a_coarse_one_is_livened():
    out = _outlines(36.24, 137.97, (20, {"building": "yes"}), (20, {"building": "yes"}))
    grey = lambda en: np.tile([0.4, 0.42, 0.45], (len(en), 1))
    assert buildings.satellite_roofs(out[:1], grey, 1.0, 1.0, lively=False) == 1
    assert np.allclose(out[0][3].roof_colour, [0.4, 0.42, 0.45])
    buildings.satellite_roofs(out[1:], grey)
    assert out[1][3].roof_colour.mean() > 0.43                          # Sentinel-2's: lifted
