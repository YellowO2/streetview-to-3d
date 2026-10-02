import mapbox_vector_tile

from streetview_to_3d.postprocess import openfreemap

EXT = 4096


def _tile(layers):
    """A vector tile's bytes: layers {name: [(properties, wkt in tile units, y down)]}."""
    return mapbox_vector_tile.encode(
        [{"name": n, "features": [{"geometry": g, "properties": p} for p, g in fs]} for n, fs in layers.items()],
        default_options={"y_coord_down": True})


def test_tiles_read_as_overpass_ways():
    # a house cut by the two tiles' edge (x = 4096), a road, a lake with an island
    left = _tile({
        "building": [({"render_height": 5, "render_min_height": 0}, "POLYGON ((4000 100, 4196 100, 4196 200, 4000 200, 4000 100))"),
                     ({"render_height": 12, "render_min_height": 0}, "POLYGON ((100 100, 200 100, 200 200, 100 200, 100 100))")],
        "transportation": [({"class": "minor"}, "LINESTRING (0 1000, 3000 1000)"),
                           ({"class": "path", "subclass": "steps", "brunnel": "bridge"}, "LINESTRING (0 1200, 300 1200)"),
                           ({"class": "rail"}, "LINESTRING (0 1500, 3000 1500)")],
        "water": [({"class": "lake"}, "POLYGON ((0 3000, 2000 3000, 2000 4000, 0 4000, 0 3000), "
                                      "(500 3200, 800 3200, 800 3500, 500 3500, 500 3200))")]})
    right = _tile({"building": [({"render_height": 5, "render_min_height": 0}, "POLYGON ((-96 100, 100 100, 100 200, -96 200, -96 100))")]})
    x, y = 2 ** 13, 2 ** 13            # at the equator, where a tile is ~2.4 km
    decoded = {(x, y): openfreemap.decode(left, x, y), (x + 1, y): openfreemap.decode(right, x + 1, y)}
    lat0, lon0 = openfreemap._ll([[x + 0.5, y + 0.5]])[0].values()
    el = openfreemap.elements(lat0, lon0, 5000, 5000, 111320, 111320, water_m=5000, full_m=5000, decoded=decoded)
    houses = [e for e in el if "building" in e["tags"]]
    assert len(houses) == 2                                   # the cut house whole again
    assert sorted(e["tags"].get("height", "") for e in houses) == ["", "12"]   # 5: OpenMapTiles' guess, dropped
    roads = sorted((e["tags"]["highway"], e["tags"].get("bridge")) for e in el if "highway" in e["tags"])
    assert roads == [("residential", None), ("steps", "yes")]    # the railway left out
    assert sum(e["tags"].get("natural") == "water" for e in el) == 1
    assert sum(1 for e in el if e["tags"] == {}) == 1           # the island, untagged as a member way
    # a point in the lake, one on the island
    pt = lambda tx, ty: list(openfreemap._ll([[x + tx / EXT, y + ty / EXT]])[0].values())
    assert openfreemap.water_at([pt(1500, 3800), pt(650, 3350)], decoded) == [True, False]


def test_far_off_only_the_big_buildings_and_main_roads():
    t = _tile({"building": [({"render_height": 9}, "POLYGON ((10 10, 14 10, 14 14, 10 14, 10 10))")],
               "transportation": [({"class": "minor"}, "LINESTRING (10 20, 40 20)"),
                                  ({"class": "primary"}, "LINESTRING (10 30, 40 30)")]})
    x, y = 2 ** 13, 2 ** 13
    lat0, lon0 = openfreemap._ll([[x + 0.5, y + 0.5]])[0].values()   # ~1.2 km from the tile's corner
    el = openfreemap.elements(lat0, lon0, 5000, 5000, 111320, 111320, decoded={(x, y): openfreemap.decode(t, x, y)})
    assert [e["tags"] for e in el] == [{"highway": "primary"}]


def test_full_detail_near_the_cameras_not_the_centre():
    t = _tile({"transportation": [({"class": "minor"}, "LINESTRING (10 20, 40 20)")]})
    x, y = 2 ** 13, 2 ** 13
    lat0, lon0 = openfreemap._ll([[x + 0.5, y + 0.5]])[0].values()
    cam = openfreemap._ll([[x + 30 / EXT, y + 30 / EXT]])[0]           # a camera by the street
    el = openfreemap.elements(lat0, lon0, 5000, 5000, 111320, 111320, decoded={(x, y): openfreemap.decode(t, x, y)},
                              cams=([cam["lat"]], [cam["lon"]]))
    assert [e["tags"] for e in el] == [{"highway": "residential"}]
