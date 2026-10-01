import io
import json
import urllib.error

import pytest

from streetview_to_3d.postprocess import osm


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
    monkeypatch.setattr(osm, "ROUNDS", 2)
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
