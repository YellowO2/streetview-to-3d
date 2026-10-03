import json

from huggingface_hub.errors import EntryNotFoundError

from streetview_to_3d.gallery import publish as gallery


def _scene(tmp_path, panos):
    nodes = [{"pano": {"id": p}, "ply": f"node_{i}.ply"} for i, p in enumerate(panos)]
    (tmp_path / "scene.json").write_text(json.dumps(
        {"center": [1.5, 103.7], "nodes": nodes, "edges": [], "land": "land.ply", "water": None}))
    for name in [n["ply"] for n in nodes] + ["land.ply", "ground.npz"]:
        (tmp_path / name).write_bytes(b"12345")
    return str(tmp_path)


class _Api:
    """HfApi with a dataset of one commit, in memory."""
    token = None

    def __init__(self):
        self.commits = []

    def dataset_info(self, repo):
        return type("Info", (), {"sha": f"sha{len(self.commits)}"})

    def create_commit(self, repo, operations, **kw):
        assert kw["parent_commit"] == f"sha{len(self.commits)}"
        self.commits.append({op.path_in_repo: op.path_or_fileobj for op in operations})


def _index(monkeypatch, tmp_path, scenes):
    def download(*a, **kw):
        if scenes is None:
            raise EntryNotFoundError("no index")
        path = tmp_path / "index_now.json"
        path.write_text(json.dumps({"scenes": scenes}))
        return str(path)
    monkeypatch.setattr(gallery, "hf_hub_download", download)
    monkeypatch.setattr(gallery, "place_name", lambda lat, lon: "Somewhere")


def test_uploads_what_the_viewer_loads_and_lists_it(monkeypatch, tmp_path):
    _index(monkeypatch, tmp_path, None)
    api = _Api()
    entry = gallery.publish(_scene(tmp_path, ["a", "b"]), log=lambda m: None, api=api)
    files = api.commits[0]
    assert sorted(f for f in files if f != "index.json") == [
        f"scenes/{entry['id']}/{n}" for n in ("land.ply", "node_0.ply", "node_1.ply", "scene.json")]
    listed = json.loads(files["index.json"])["scenes"]
    assert listed == [entry] and entry["panos"] == ["a", "b"] and entry["bytes"] > 15


def test_a_run_already_listed_is_not_uploaded_again(monkeypatch, tmp_path):
    there = {"id": "old", "panos": ["a", "b", "c", "d", "e"]}
    _index(monkeypatch, tmp_path, [there])
    api = _Api()
    assert gallery.publish(_scene(tmp_path, ["a", "b", "c", "d", "x"]),
                           log=lambda m: None, api=api) == there
    assert not api.commits


def test_a_run_mostly_new_is_uploaded(monkeypatch, tmp_path):
    _index(monkeypatch, tmp_path, [{"id": "old", "panos": ["a", "b"]}])
    api = _Api()
    gallery.publish(_scene(tmp_path, ["a", "x", "y"]), log=lambda m: None, api=api)
    assert len(json.loads(api.commits[0]["index.json"])["scenes"]) == 2
