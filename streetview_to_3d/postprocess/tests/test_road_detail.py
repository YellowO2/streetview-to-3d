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
