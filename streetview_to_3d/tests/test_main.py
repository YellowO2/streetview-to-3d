from streetview_to_3d import main
from streetview_to_3d.Settings import Settings


def test_a_run_goes_reconstruct_postprocess_save_into_its_own_folder(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "run_folder", lambda log: "runs/1")
    monkeypatch.setattr(main, "reconstruct", lambda prep, walk: calls.append(("reconstruct", prep, walk)) or "scene")
    monkeypatch.setattr(main, "postprocess", lambda scene, post, folder, log:
                        calls.append(("postprocess", scene, post, folder)) or ("done", "ground", "world"))
    monkeypatch.setattr(main, "save", lambda scene, ground, world, folder:
                        calls.append(("save", scene, ground, world, folder)) or folder)
    cfg = Settings()
    assert main.streetview_to_3d("prep", cfg, log=lambda m: None) == "runs/1"
    assert calls == [("reconstruct", "prep", cfg.walk), ("postprocess", "scene", cfg.post, "runs/1"),
                     ("save", "done", "ground", "world", "runs/1")]
