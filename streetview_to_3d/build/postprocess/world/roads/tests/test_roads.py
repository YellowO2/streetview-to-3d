from types import SimpleNamespace

import numpy as np
import shapely

from streetview_to_3d.build.postprocess.world.roads import portals, road_mesh, road_points
from streetview_to_3d.build.postprocess.world.roads import RoadHeight as road_height
from streetview_to_3d.build.postprocess.world.roads.Network import Network
from streetview_to_3d.build.postprocess.world.roads.tools import paving
from streetview_to_3d.build.postprocess.world.tools import osm_roads

to_xy = lambda geometry: np.array([[p["lon"], p["lat"]] for p in geometry], float)   # metres, as they are


def _site(near=True, to_xy=to_xy):
    """A stand-in Site: geometry in metres as it is, every place near a camera or none."""
    return SimpleNamespace(to_xy=to_xy, from_cameras=lambda xy: np.full(len(xy), 0.0 if near else 1e9))


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
    under = road_height._standing(ELEMENTS, to_xy, np.array([[0.0, 1.0]]), np.array([0.0]), DECKS)
    assert under == {1}
    on = road_height._standing(ELEMENTS, to_xy, np.array([[0.0, 1.0]]), np.array([5.8]), DECKS)
    assert on == {2}


def test_a_bridge_passing_by_is_not_the_scenes_road():
    ids = road_height._standing(ELEMENTS, to_xy, np.array([[-30.0, 1.0]]), np.array([0.0]), DECKS)
    ids = road_height._joined(ELEMENTS, to_xy, ids, lambda xy: np.linalg.norm(xy, axis=1) < 40)
    assert ids == {1, 3}                    # the street, and the side street joining it near the scene
    far = road_height._joined(ELEMENTS, to_xy, {1}, lambda xy: np.linalg.norm(xy, axis=1) < 20)
    assert far == {1}                       # joining it only further out: its own height
    met = road_height._region(ELEMENTS, to_xy, ids, 1.0)
    assert shapely.contains_xy(met, 30, 20) and not shapely.contains_xy(met, 0, 20)   # a bridge is not in it


def test_tagged_pavements_are_on_correct_side_raised_and_do_not_cover_roads():
    street = _way(1, [(0, 0), (60, 0)], [1, 2], sidewalk="left", lanes="2")
    net = Network([street], _site())
    assert shapely.contains_xy(net.shapes[paving.PAVEMENT], 20, 4)
    assert not shapely.contains_xy(net.all, 20, -4)
    assert shapely.intersection(net.shapes[paving.PAVEMENT], net.shapes[tuple(osm_roads.colour(street["tags"]))]).area == 0
    ground = lambda xy: np.zeros(len(xy))
    pts, cols = road_points._points(net, lambda xy: np.full(len(xy), .3), ground, lambda xy: np.ones(len(xy), bool))
    assert len(pts) and np.isfinite(pts).all()
    assert np.isclose(-pts[:, 1].max(), paving.LIFT_M)
    assert np.isclose(-pts[:, 1].min(), paving.LIFT_M + paving.KERB_M)
    # Paint is narrow but should still receive points, instead of sampling only its boundary.
    assert (cols[:, 0] > .79).any()
    v, c, f = road_mesh._surface(net, ground, shapely.box(-10, -10, 70, 10))
    assert f.max() < len(v) and np.isfinite(v).all()
    assert np.isclose(-v[:, 1].min(), paving.LIFT_M + paving.KERB_M - road_mesh.BELOW_M, atol=.051)


def test_separate_sidewalk_overrides_and_lane_markings_opt_out():
    street = _way(1, [(0, 0), (60, 0)], [1, 2], sidewalk="both",
                  **{"sidewalk:left": "separate", "sidewalk:right": "no", "lanes": "2", "lane_markings": "no"})
    sidewalk = _way(2, [(0, 5), (60, 5)], [3, 4], highway="footway", footway="sidewalk")
    net = Network([street, sidewalk], _site())
    assert shapely.contains_xy(net.all, 20, 5)
    assert not shapely.contains_xy(net.all, 20, 3.5)
    assert paving.PAINT not in net.shapes


def test_lane_dividers_stop_at_mapped_junctions_and_pavements_avoid_buildings():
    street = _way(1, [(0, 0), (30, 0), (60, 0)], [1, 2, 3], sidewalk="left", lanes="2")
    side = _way(2, [(30, 0), (30, 30)], [2, 4])
    building = {"type": "way", "id": 3, "tags": {"building": "yes"},
                "geometry": [{"lon": x, "lat": y} for x, y in [(10, 3), (20, 3), (20, 6), (10, 6), (10, 3)]]}
    net = Network([street, side, building], _site())
    assert not shapely.intersects(net.shapes[paving.PAINT], shapely.Point(30, 0).buffer(7.9))
    assert not shapely.contains_xy(net.all, 15, 4)


def test_crossing_seams_and_furniture_survive_both_exports():
    street = _way(1, [(0, 0), (60, 0)], [1, 2], sidewalk='both', lanes='2', lit='yes')
    crossing = {'type':'node', 'id':3, 'lon':25, 'lat':0,
                'tags':{'highway':'crossing', 'crossing':'marked'}}
    bench = {'type':'node', 'id':4, 'lon':30, 'lat':4.2, 'tags':{'amenity':'bench'}}
    net = Network([street,crossing,bench], _site())
    assert net.shapes[paving.PAVING_JOINT].area > 0
    assert shapely.contains_xy(net.shapes[paving.PAINT],25,-2.5)
    ground = lambda xy: np.zeros(len(xy))
    pts, cols = road_points._points(net,lambda xy: np.full(len(xy),.2),ground,lambda xy: np.ones(len(xy),bool))
    v,c,f = road_mesh._surface(net,ground,shapely.box(-10,-10,70,10))
    assert -pts[:,1].min() > 5 and -v[:,1].min() > 5
    assert np.isfinite(pts).all() and np.isfinite(v).all()
    assert f.max() < len(v)
    # Pavement joint triangles are still at pavement height, not sunk into the road.
    joints = np.all(np.isclose(c,np.array(paving.PAVING_JOINT),atol=.015),axis=1)
    assert joints.any()
    assert np.allclose(-v[joints,1],paving.LIFT_M+paving.KERB_M-road_mesh.BELOW_M)


def _street(x):
    return {"type": "way", "id": x, "tags": {"highway": "residential", "sidewalk": "both"},
            "geometry": [(x, 0), (x, 50)]}


def test_sidewalks_only_on_streets_near_the_cameras():
    to_xy = lambda g: np.array(g, float)
    near = Network([_street(0)], _site(near=True, to_xy=to_xy))
    far = Network([_street(0)], _site(near=False, to_xy=to_xy))
    assert not near.shapes[paving.PAVEMENT].is_empty
    assert paving.PAVEMENT not in far.shapes and not far.all.is_empty     # the carriageway still


def test_a_mouth_where_a_tunnel_leaves_the_ground_road():
    to_xy = lambda g: np.array(g, float)
    road = {"type": "way", "id": 1, "tags": {"highway": "primary"}, "geometry": [(0, 0), (50, 0)]}
    tunnel = {"type": "way", "id": 2, "tags": {"highway": "primary", "tunnel": "yes"}, "geometry": [(50, 0), (300, 0)]}
    pts, cols, faces = portals._portals([road, tunnel], to_xy, lambda xy: np.full(len(xy), 10.0))
    assert len(faces) == 14                                       # one mouth: frame and dark, 7 quads
    assert np.isclose(-pts[:, 1].max(), 10.0)                     # standing on the road's height (y is down)
    assert (pts[:, 0] >= 50 - 1e-9).all() and pts[:, 0].max() <= 50 + portals.MOUTH_M + 1e-9   # facing out, dark going in
    assert not len(portals._portals([tunnel], to_xy, lambda xy: np.zeros(len(xy)))[2])   # no ground road: no mouth
