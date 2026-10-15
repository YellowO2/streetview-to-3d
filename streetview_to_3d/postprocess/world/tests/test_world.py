import logging
from types import SimpleNamespace

import numpy as np

from streetview_to_3d.types.Node import Node
from streetview_to_3d.types.Pano import Pano
from streetview_to_3d.types.Scene import Scene
from streetview_to_3d.postprocess.World import World
from streetview_to_3d.postprocess.world import OpenStreetMap as osm
from streetview_to_3d.postprocess.world import driving_side as side
from streetview_to_3d.postprocess.world import elevations_around as around
from streetview_to_3d.postprocess.world.main import world
from streetview_to_3d.postprocess.world.scene_files import scene_files


def _site():
    """A stand-in Site: one camera at the origin, its pano with an elevation."""
    return SimpleNamespace(origin=(51.5, -0.12), m_per_lon=69300.0, cameras=np.zeros((1, 2)),
                           to_ll=lambda xy: (51.5 + xy[:, 1] / 111320.0, -0.12 + xy[:, 0] / 69300.0),
                           panos=[Pano("google", "a", 51.5, -0.12, elevation=10.0)])


def test_without_a_placed_camera_there_is_no_world(caplog):
    caplog.set_level(logging.WARNING)
    scene = Scene(center=[51.5, -0.12], nodes=[Node(pano=Pano("google", "a", 51.5, -0.12))], land="land.ply")
    assert world(scene, None) == (scene, None)
    assert scene.land == "land.ply"                                       # its names as they were
    assert caplog.messages == ["terrain: no placed cameras"]


def test_the_scene_names_the_files_its_world_holds():
    scene = Scene(center=[0.0, 0.0], terrain="terrain.ply", google="google.ply")
    laid = World(meshes={"land.ply": {}}, points={"buildings.ply": {}}, json={"life.json": {}})
    scene = scene_files(scene, laid)
    assert (scene.land, scene.buildings, scene.life) == ("land.ply", "buildings.ply", "life.json")
    assert scene.terrain is scene.roads is scene.water is scene.blocks is None
    assert scene.google == "google.ply"                                   # not the map's to name


def test_downloads_that_fail_leave_the_world_without_them(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)

    def offline(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(osm, "_fetch", offline)
    monkeypatch.setattr(around, "_around", offline)
    monkeypatch.setattr(side, "_side", offline)
    site = _site()
    assert osm.OpenStreetMap(site).elements() == []
    elevations = around.elevations_around(site)
    assert elevations.xy.shape == (0, 2) and elevations.metres.shape == (0,)
    assert side.driving_side(site) == "right"
    assert caplog.messages == ["terrain: no OpenStreetMap (OSError('offline'))",
                               "terrain: no other panos' elevation (OSError('offline'))",
                               "terrain: no country for the cars' side (OSError('offline'))"]


def test_other_panos_elevations_in_metres_from_the_scenes_origin(monkeypatch):
    monkeypatch.setattr(around, "_around", lambda lats, lons: (np.array([51.5 + 100 / 111320.0]),
                                                               np.array([-0.12 + 50 / 69300.0]), np.array([12.0])))
    elevations = around.elevations_around(_site())
    assert np.allclose(elevations.xy, [[50, 100]]) and elevations.metres.tolist() == [12.0]
