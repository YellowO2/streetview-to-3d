import json

import numpy as np

from streetview_to_3d.postprocess.world import life


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


def test_ducks_on_the_nearest_water_in_from_its_shore():
    lake = {"level": 2.0, "outer": [(100, -100), (400, -100), (400, 100), (100, 100)], "holes": []}
    (home,) = life.ducks([lake], np.array([(0.0, 0.0)]))["homes"]
    assert home["level"] == 2.0 and abs(home["at"][0] - (100 + life.DUCK_SHORE_M)) < 0.5
    assert 0 < home["reach"] <= life.DUCK_SHORE_M - 2 + 1e-6
    assert life.ducks([lake], np.array([(-1000.0, 0.0)]))["homes"] == []
    # the scene on a bridge over it: the ducks off its ground, not under it
    from streetview_to_3d.postprocess.seams import SceneGround
    g = np.mgrid[95:140:0.5, -10:10:0.5].reshape(2, -1).T
    bridge = SceneGround.from_points(np.c_[g[:, 0], -np.full(len(g), 8.0), g[:, 1]], np.full((len(g), 3), 0.5))
    (home,) = life.ducks([lake], np.array([(120.0, 0.0)]), bridge)["homes"]
    assert bridge.at(np.array([home["at"]]))[0][0] >= life.DUCK_CLEAR_M - 0.01 and home["reach"] > 0


def test_cats_sit_on_the_scenes_ground_off_its_roads_near_its_cameras():
    from streetview_to_3d.postprocess.seams import SceneGround
    # ground 40 m square round the cameras, a car road along its middle (east-west)
    g = np.mgrid[-20:20:0.5, -20:20:0.5].reshape(2, -1).T
    ground = SceneGround.from_points(np.c_[g[:, 0], -np.full(len(g), 3.0), g[:, 1]], np.full((len(g), 3), 0.5))
    road = [(6.0, np.c_[np.linspace(-20, 20, 11), np.zeros(11), np.zeros(11)], np.full(11, 3.0))]
    cams = np.array([(0.0, 0.0), (5.0, 0.0)])
    out = life.cats(ground, road, cams)["cats"]
    assert len(out) == 2
    for cat in out:
        p = np.array(cat["spots"])
        assert len(p) == life.CAT_SPOTS and np.allclose(p[:, 2], 3.0)
        assert (np.abs(p[:, 1]) > 3 + life.CAT_ROAD_M - 0.3).all()            # off the road
        assert (np.hypot(*(p[:, :2] - p[0, :2]).T) <= life.CAT_SPOT_M).all()
    assert np.hypot(*(np.array(out[0]["spots"][0][:2]) - out[1]["spots"][0][:2])) >= life.CATS_APART_M
    assert life.cats(SceneGround.none(), road, cams)["cats"] == []


def test_saved_as_the_viewer_reads_it(tmp_path):
    name = life.save(tmp_path, cars=life.cars([(6.0, np.zeros((3, 3)), np.full(3, 3.0))], "left"),
                     birds=life.birds([], np.zeros((1, 2)), 0.0), boats=life.boats([], np.zeros((1, 2))))
    data = json.loads((tmp_path / name).read_text())
    assert data["cars"]["side"] == "left" and len(data["cars"]["roads"]) == 1
    assert data["birds"]["flocks"][0]["kind"] == "pigeon" and data["boats"]["courses"] == []


def test_cars_kept_to_the_scenes_own_road_where_osms_line_runs_along_its_pavement():
    from streetview_to_3d.postprocess.seams import SceneGround
    # ground 60 m along, 20 m across; road from 1 to 7 m north of OSM's line (y = 0), pavement either side
    g = np.mgrid[-30:30:0.5, -10:10:0.5].reshape(2, -1).T + 0.25
    road = (g[:, 1] > 1) & (g[:, 1] < 7)
    ground = SceneGround.from_points(np.c_[g[:, 0], np.zeros(len(g)), g[:, 1]], np.full((len(g), 3), 0.5), road)
    xy = np.c_[np.arange(-60.0, 61, life.STEP_M), np.zeros(31)]
    moved, half = life.onto_road(xy, ground)
    on = np.abs(xy[:, 0]) < 25
    assert np.allclose(moved[on, 1], 4, atol=0.3) and np.allclose(half[on], 3, atol=0.3)   # its middle, 3 m each side
    assert np.allclose(moved[np.abs(xy[:, 0]) > 55, 1], 0, atol=0.3)                     # eased back past it
    assert np.isnan(half[~on & (np.abs(xy[:, 0]) > 35)]).all()


def test_the_ground_keeps_its_road(tmp_path):
    from streetview_to_3d.postprocess.seams import SceneGround
    pts = np.array([[0.1, 0, 0.1], [0.2, 0, 0.2], [5.1, 0, 0.1]])
    g = SceneGround.from_points(pts, np.full((3, 3), 0.5), np.array([True, False, True]))
    g.save(tmp_path)
    back = SceneGround.load(tmp_path)
    assert np.allclose(back.road_at(np.array([[0.1, 0.1], [5.1, 0.1]])), [0.5, 1])
    assert np.isnan(back.road_at(np.array([[2.6, 0.1]])))[0]
    assert np.isnan(SceneGround.from_points(pts, np.full((3, 3), 0.5)).road_at(pts[:, [0, 2]])).all()
