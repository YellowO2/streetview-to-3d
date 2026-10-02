import json

import numpy as np

from streetview_to_3d.postprocess import life


def _way(xy, **tags):
    return {"type": "way", "id": len(xy), "tags": {"highway": "residential", **tags},
            "geometry": [{"x": x, "y": y} for x, y in xy]}


to_xy = lambda g: np.array([(p["x"], p["y"]) for p in g], float)


def test_a_road_keeps_its_room_to_the_walls_and_is_cut_where_no_car_fits():
    # a road east 200 m; a house 3 m off its line at 50-70 m; one on it at 140-150 m
    road = _way([(0, 0), (200, 0)])
    walls = [np.array([(50, 3), (70, 3), (70, 10), (50, 10)]),
             np.array([(140, -5), (150, -5), (150, 5), (140, 5)])]
    out = life.roads([road], to_xy, lambda xy: np.zeros(len(xy)), [], lambda xy: np.ones(len(xy), bool), walls)
    assert len(out) == 2                                    # cut at the house on it
    (w, pts, room), _ = out
    assert w == 6 and pts[-1, 0] < 140
    at = (pts[:, 0] > 52) & (pts[:, 0] < 68)
    assert np.allclose(room[at], 3) and np.allclose(room[pts[:, 0] < 40], 3)   # half its width, or the wall


def test_gulls_and_swallows_by_the_water_pigeons_and_swallows_inland_and_a_boat_round_the_water():
    lake = {"level": 2.0, "outer": [(100, -100), (400, -100), (400, 100), (100, 100)], "holes": []}
    cams = np.array([(0.0, 0.0), (10.0, 0.0)])
    kinds = lambda b: [f["kind"] for f in b["flocks"]]
    assert kinds(life.birds([lake], cams, 5.0)) == ["gull", "swallow"]
    assert kinds(life.birds([lake], cams + 1000, 5.0)) == ["pigeon", "swallow"]
    assert life.birds([], cams, 5.0)["centre"] == [5.0, 0.0, 5.0]
    (course,) = life.boats([lake], cams)["courses"]
    p = np.array(course["points"])
    assert course["level"] == 2.0 and p[:, 0].min() >= 100 + life.SHORE_M - 0.5
    assert np.hypot(*(p - (5, 0)).T).max() <= life.BOATS_M - life.SHORE_M + 1
    assert life.boats([lake], cams + 2000)["courses"] == []


def test_saved_as_the_viewer_reads_it(tmp_path):
    name = life.save(tmp_path, [(6.0, np.zeros((3, 3)), np.full(3, 3.0))], "left",
                     life.birds([], np.zeros((1, 2)), 0.0), life.boats([], np.zeros((1, 2))))
    data = json.loads((tmp_path / name).read_text())
    assert data["cars"]["side"] == "left" and len(data["cars"]["roads"]) == 1
    assert data["birds"]["flocks"][0]["kind"] == "pigeon" and data["boats"]["courses"] == []
