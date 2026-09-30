from types import SimpleNamespace

import numpy as np
import shapely

from streetview_to_3d.postprocess import roads

to_xy = lambda geometry: np.array([[p["lon"], p["lat"]] for p in geometry], float)   # metres, as they are


def _way(i, pts, nodes, **tags):
    return {"type": "way", "id": i, "nodes": nodes, "geometry": [{"lon": x, "lat": y} for x, y in pts],
            "tags": {"highway": "residential", **tags}}


# a street east-west through the scene, a bridge over it north-south, a side
# street joining the street at (30, 0), and another bridge 20 m off, joining nothing
STREET = _way(1, [(-50, 0), (0, 0), (30, 0), (50, 0)], [10, 11, 12, 13])
OVER = _way(2, [(0, -40), (0, 40)], [20, 21], bridge="yes", layer="1")
SIDE = _way(3, [(30, 0), (30, 40)], [12, 30])
PAST = _way(4, [(-50, 20), (50, 20)], [40, 41], bridge="yes", layer="1")
ELEMENTS = [STREET, OVER, SIDE, PAST]
DECKS = [SimpleNamespace(ids={2}, xy=np.array([[0.0, y] for y in range(-40, 41)]), h=np.full(81, 6.0)),
         SimpleNamespace(ids={4}, xy=np.array([[x, 20.0] for x in range(-50, 51)]), h=np.full(101, 6.0))]


def test_the_scene_under_a_bridge_is_the_street_and_on_it_the_bridge():
    under = roads.standing(ELEMENTS, to_xy, np.array([[0.0, 1.0]]), np.array([0.0]), DECKS)
    assert under == {1}
    on = roads.standing(ELEMENTS, to_xy, np.array([[0.0, 1.0]]), np.array([5.8]), DECKS)
    assert on == {2}


def test_a_bridge_passing_by_is_not_the_scenes_road():
    ids = roads.standing(ELEMENTS, to_xy, np.array([[-30.0, 1.0]]), np.array([0.0]), DECKS)
    ids = roads.joined(ELEMENTS, to_xy, ids, lambda xy: np.linalg.norm(xy, axis=1) < 40)
    assert ids == {1, 3}                    # the street, and the side street joining it near the scene
    far = roads.joined(ELEMENTS, to_xy, {1}, lambda xy: np.linalg.norm(xy, axis=1) < 20)
    assert far == {1}                       # joining it only further out: its own height
    met = roads.region(ELEMENTS, to_xy, ids, 1.0)
    assert shapely.contains_xy(met, 30, 20) and not shapely.contains_xy(met, 0, 20)   # a bridge is not in it
