from streetview_to_3d import main
from streetview_to_3d.Settings import Settings


def test_a_run_goes_reconstruct_postprocess_save_and_the_folder_comes_last(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "reconstruct", lambda corridor, walk: calls.append(("reconstruct", corridor, walk)) or "scene")
    monkeypatch.setattr(main, "postprocess", lambda scene, post:
                        calls.append(("postprocess", scene, post)) or ("done", "ground", "world"))
    monkeypatch.setattr(main, "save", lambda scene, ground, world:
                        calls.append(("save", scene, ground, world)) or "runs/1")
    cfg = Settings()
    assert main.streetview_to_3d("corridor", cfg) == "runs/1"
    assert calls == [("reconstruct", "corridor", cfg.walk), ("postprocess", "scene", cfg.post),
                     ("save", "done", "ground", "world")]
