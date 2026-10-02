import types

import numpy as np

from streetview_to_3d.postprocess import water
from streetview_to_3d.ui.map_selection import candidates

STEP = 10 / 111320.0   # a pano every 10 m north along one street


def _street(monkeypatch, n=21):
    """A street of n panos, each linked to the next; the network faked,
    every links lookup counted."""
    panos = {f"p{i}": types.SimpleNamespace(id=f"p{i}", lat=i * STEP, lon=0.0, heading=0.0) for i in range(n)}
    asked = []

    async def fetch(ids):
        asked.extend(ids)
        out = []
        for pid in ids:
            i = int(pid[1:])
            out.append({"neighbors": [{"id": f"p{j}", "lat": j * STEP, "lon": 0.0}
                                      for j in (i - 1, i + 1) if 0 <= j < n]})
        return out

    monkeypatch.setattr(candidates, "google_tile_panos", lambda lat, lon, radius_m=None: dict(panos))
    monkeypatch.setattr(candidates, "_nearest_to", lambda *a: None)
    monkeypatch.setattr(candidates, "fetch_panos_by_id", fetch)
    monkeypatch.setattr(water, "occurrence_map", lambda: (lambda lat, lon: np.zeros(np.shape(lat))))
    candidates._meta_cache.clear()
    return asked


def _area(radius_m):
    return candidates.circle(10 * STEP, 0.0, radius_m, corners=32)


def test_an_area_redrawn_fetches_only_what_it_adds(monkeypatch):
    asked = _street(monkeypatch)
    nodes, edges = candidates.expand_area(10 * STEP, 0.0, area=_area(55))
    walked = set(asked)
    assert walked == {f"p{i}" for i in range(5, 16)}                    # walked from inside it only
    assert {n["id"] for n in nodes} == {f"p{i}" for i in range(4, 17)}  # one past it kept, as a leaf
    asked.clear()
    nodes, _ = candidates.expand_area(10 * STEP, 0.0, area=_area(25))   # pulled in
    assert not asked                                                     # nothing fetched again
    assert {n["id"] for n in nodes} == {f"p{i}" for i in range(7, 14)}
    nodes, _ = candidates.expand_area(10 * STEP, 0.0, area=_area(85))   # pushed out
    assert set(asked) == {"p2", "p3", "p4", "p16", "p17", "p18"}         # only what it adds
    assert {n["id"] for n in nodes} == {f"p{i}" for i in range(1, 20)}


def test_a_circle_and_its_shape_find_the_same(monkeypatch):
    _street(monkeypatch)
    by_radius, _ = candidates.expand_area(10 * STEP, 0.0, 55)
    by_shape, _ = candidates.expand_area(10 * STEP, 0.0, area=_area(55))
    assert {n["id"] for n in by_radius} == {n["id"] for n in by_shape}
