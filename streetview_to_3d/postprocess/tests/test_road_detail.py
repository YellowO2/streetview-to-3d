import numpy as np

from streetview_to_3d.postprocess import roads


def _street(x):
    return {"type": "way", "id": x, "tags": {"highway": "residential", "sidewalk": "both"},
            "geometry": [(x, 0), (x, 50)]}


def test_sidewalks_only_on_streets_near_the_cameras():
    to_xy = lambda g: np.array(g, float)
    near = roads.Network([_street(0)], to_xy, lambda xy: True)
    far = roads.Network([_street(0)], to_xy, lambda xy: False)
    assert not near.shapes[roads.PAVEMENT].is_empty
    assert roads.PAVEMENT not in far.shapes and not far.all.is_empty     # the carriageway still


def test_a_mouth_where_a_tunnel_leaves_the_ground_road():
    to_xy = lambda g: np.array(g, float)
    road = {"type": "way", "id": 1, "tags": {"highway": "primary"}, "geometry": [(0, 0), (50, 0)]}
    tunnel = {"type": "way", "id": 2, "tags": {"highway": "primary", "tunnel": "yes"}, "geometry": [(50, 0), (300, 0)]}
    pts, cols, faces = roads.portals([road, tunnel], to_xy, lambda xy: np.full(len(xy), 10.0))
    assert len(faces) == 14                                       # one mouth: frame and dark, 7 quads
    assert np.isclose(-pts[:, 1].max(), 10.0)                     # standing on the road's height (y is down)
    assert (pts[:, 0] >= 50 - 1e-9).all() and pts[:, 0].max() <= 50 + roads.MOUTH_M + 1e-9   # facing out, dark going in
    assert not len(roads.portals([tunnel], to_xy, lambda xy: np.zeros(len(xy)))[2])   # no ground road: no mouth
