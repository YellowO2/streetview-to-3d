import io
import json
import urllib.error

import pytest

from streetview_to_3d.build_scene.postprocess.world import OpenStreetMap as osm
from streetview_to_3d.tools import paths


def _server(monkeypatch, answers):
    """urlopen giving answers in turn: an HTTP code (refused) or a dict (its JSON); the URLs asked."""
    asked = []

    def urlopen(req, timeout):
        asked.append(req.full_url)
        assert req.get_header("User-agent") == osm.USER_AGENT
        a = answers.pop(0)
        if isinstance(a, int):
            raise urllib.error.HTTPError(req.full_url, a, "busy", {}, None)
        return io.BytesIO(json.dumps(a).encode())
    monkeypatch.setattr(osm.urllib.request, "urlopen", urlopen)
    return asked


def test_every_mirror_busy_waits_once_then_asks_them_again(monkeypatch):
    n = len(osm.OVERPASS_URLS)
    asked = _server(monkeypatch, [504] * n + [429, {"elements": [1]}])
    waits = []
    assert osm._ask("q", sleep=waits.append) == {"elements": [1]}
    assert waits == [osm.BUSY_WAIT_S]
    assert asked == list(osm.OVERPASS_URLS) + list(osm.OVERPASS_URLS[:2])


def test_a_wrong_request_is_not_asked_elsewhere(monkeypatch):
    asked = _server(monkeypatch, [400])
    with pytest.raises(urllib.error.HTTPError):
        osm._ask("q", sleep=lambda s: None)
    assert len(asked) == 1


def test_refused_every_round_raises_the_last_refusal(monkeypatch):
    _server(monkeypatch, [504] * (len(osm.OVERPASS_URLS) * osm.ROUNDS))
    with pytest.raises(urllib.error.HTTPError):
        osm._ask("q", sleep=lambda s: None)


def test_no_answer_within_the_budget_raises(monkeypatch):
    now = [0.0]

    def urlopen(req, timeout):
        now[0] += timeout
        raise TimeoutError("hung")
    monkeypatch.setattr(osm.urllib.request, "urlopen", urlopen)
    with pytest.raises(TimeoutError):
        osm._ask("q", sleep=lambda s: now.__setitem__(0, now[0] + s), budget=30, clock=lambda: now[0])
    assert now[0] <= 31


def test_without_overpass_the_tiles_stand_in_and_are_not_kept(monkeypatch, tmp_path):
    def busy(query, **kw):
        raise TimeoutError("busy")
    tiles = [{"type": "way", "id": -1, "tags": {"building": "yes"}, "geometry": []}]
    monkeypatch.setattr(osm, "_ask", busy)
    monkeypatch.setattr(paths, "DATA_DIR", str(tmp_path))
    stand_in = type("Tiles", (), {"elements": lambda self, *a, **k: tiles})()
    assert osm._fetch(0, 0, 800, 700, 111320, 111320, tiles=stand_in) == tiles
    assert not (tmp_path / "map_cache").exists()


def test_an_answer_is_kept_for_any_later_run_asking_the_same(monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(osm, "_ask", lambda query, **kw: asked.append(query) or {"elements": [{"id": 1}]})
    monkeypatch.setattr(paths, "DATA_DIR", str(tmp_path))
    first = osm._fetch(0, 0, 800, 700, 111320, 111320)
    assert osm._fetch(0, 0, 800, 700, 111320, 111320) == first == [{"id": 1}]
    assert len(asked) == 1
    osm._fetch(1, 1, 800, 700, 111320, 111320)
    assert len(asked) == 2 and len(list((tmp_path / "map_cache").iterdir())) == 2
