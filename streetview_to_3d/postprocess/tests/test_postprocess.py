import logging
from types import SimpleNamespace

from streetview_to_3d.postprocess import main
from streetview_to_3d.postprocess.Settings import Settings
from streetview_to_3d.postprocess.world.Settings import Settings as WorldSettings


def _stages(monkeypatch, done, fill=None, world=None):
    """Every stage of main.postprocess replaced by one noting its name."""
    def stage(name):
        def run(scene, *a):
            done.append(name)
            return scene
        return run
    for name in ("place", "drop_blobs", "drop_sheets", "photos"):
        monkeypatch.setattr(main, name, stage(name))
    monkeypatch.setattr(main, "fill", fill or (lambda scene: (done.append("fill") or scene, "ground")))
    monkeypatch.setattr(main, "lay_world", world or (lambda scene, ground, settings: (scene, ("world", ground))))


def test_stages_run_in_order_and_settings_leave_out_their_part(monkeypatch):
    done = []
    _stages(monkeypatch, done)
    scene, ground, world = main.postprocess(SimpleNamespace(), Settings(), None)
    assert done == ["place", "drop_blobs", "drop_sheets", "photos", "fill"]
    assert ground == "ground" and world == ("world", "ground")
    done.clear()
    main.postprocess(SimpleNamespace(), Settings(clean=False), None)
    assert "drop_sheets" not in done
    done.clear()
    assert main.postprocess(SimpleNamespace(), Settings(fill=False), None)[1:] == (None, None)
    assert done == ["place"]


def test_a_failing_fill_or_world_is_skipped_and_said_so(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    done = []

    def broken(*a):
        raise ValueError("no")
    _stages(monkeypatch, done, fill=broken, world=broken)
    scene, ground, world = main.postprocess(SimpleNamespace(), Settings(), None)
    lines = caplog.messages
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert ground is None and world is None
    assert warned == ["fill skipped: ValueError('no')", "terrain skipped: ValueError('no')"]
    assert [l.split()[1] for l in lines if l.startswith("timing:")] == [
        "placement", "floating", "loose", "fill", "terrain"]


def test_the_world_gets_its_settings_and_names_its_files_on_the_scene(monkeypatch):
    asked = []

    def world(scene, ground, settings):
        asked.append(settings)
        return SimpleNamespace(named=True), "world"
    _stages(monkeypatch, [], world=world)
    google = WorldSettings(google="google.ply", bend=False)
    scene, ground, laid = main.postprocess(SimpleNamespace(), Settings(world=google), None)
    assert asked == [google] and scene.named and laid == "world"
    assert Settings().world == WorldSettings() and WorldSettings().google is None
