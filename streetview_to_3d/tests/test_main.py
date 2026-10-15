from streetview_to_3d import main
from streetview_to_3d.Settings import Settings
from streetview_to_3d.reconstruct.WalkSettings import WalkSettings


def test_a_run_goes_reconstruct_postprocess_save_and_the_folder_comes_last(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "reconstruct", lambda corridor, walk, depth_model, masker:
                        calls.append(("reconstruct", corridor, walk, depth_model.repo, masker.model_id)) or "scene")
    monkeypatch.setattr(main, "postprocess", lambda scene, post:
                        calls.append(("postprocess", scene, post)) or ("done", "ground", "world"))
    monkeypatch.setattr(main, "save", lambda scene, ground, world:
                        calls.append(("save", scene, ground, world)) or "runs/1")
    settings = Settings(walk=WalkSettings(model="some/da3", masker="some/segformer"))
    assert main.streetview_to_3d("corridor", settings) == "runs/1"
    assert calls == [("reconstruct", "corridor", settings.walk, "some/da3", "some/segformer"),
                     ("postprocess", "scene", settings.post),
                     ("save", "done", "ground", "world")]
