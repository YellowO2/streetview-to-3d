import copy

import numpy as np

from streetview_to_3d.types.Cloud import Cloud
from streetview_to_3d.types.Node import Node
from streetview_to_3d.types.Pano import Pano
from streetview_to_3d.types.Scene import Scene
from streetview_to_3d.postprocess.World import World
from streetview_to_3d.postprocess.world.Settings import Settings
from streetview_to_3d.postprocess.world.Site import Site
from streetview_to_3d.postprocess.world.google.GooglePoints import GooglePoints
from streetview_to_3d.postprocess.world.google.main import google
from streetview_to_3d.postprocess.world.google_points import google_points
from streetview_to_3d.tools.ply import write_ply

UP = np.array([0.0, -1.0, 0.0])     # y down


def _grid(xs, ys, zs):
    g = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([a.ravel() for a in g], 1)


def _scene():
    """One pano at the origin, its DA3 points a wall 10 m north and the ground round it."""
    wall = _grid(np.arange(-5, 5, 0.1), np.arange(-3, 1.5, 0.1), [10.0])
    ground = _grid(np.arange(-5, 5, 0.25), [1.6], np.arange(0, 10, 0.25))
    pts = np.concatenate([wall, ground])
    node = Node(pano=Pano("google", "a", 0.0, 0.0, elevation=10.0), ply="node_0.ply", position=[0.0, 0.0, 0.0],
                rotation=np.eye(3).tolist(), transform=np.eye(4).tolist())
    node.cloud = Cloud(pts, np.full((len(pts), 3), 0.5))
    return Scene(center=[0.0, 0.0], nodes=[node], land="land.ply", terrain="terrain.ply", roads="roads.ply",
                 buildings="buildings.ply", blocks="blocks.ply", water="water.json", life="life.json")


def _world():
    """The map's world with one lake (45-85 m east and north, its level 0.5 m)."""
    size, cell, lo = 20, 10.0, -100.0
    centres = lo + cell * (np.arange(size) + 0.5)
    lake = (centres[None, :] > 45) & (centres[None, :] < 85) & (centres[:, None] > 45) & (centres[:, None] < 85)
    shore = {"lo": lo, "cell": cell, "size": size, "metres": np.where(lake, 60, 0).ravel().tolist()}
    body = {"level": 0.5, "outer": [[45, 45], [85, 45], [85, 85], [45, 85], [45, 45]], "holes": []}
    return World(meshes={"land.ply": {}}, json={"water.json": {"surfaces": [body], "shore": shore},
                                                 "life.json": {"cars": {}}})


def _google():
    """Google's points: DA3's wall, something standing in front of it, a wall behind it, and the lake
    1 m up (with its quay a metre higher)."""
    wall = _grid(np.arange(-5, 5, 0.1), np.arange(-3, 1.5, 0.1), [10.0])
    air = _grid(np.arange(-1, 1, 0.1), np.arange(-1, 0, 0.1), [5.0])
    behind = _grid(np.arange(-5, 5, 0.2), np.arange(-3, 1.5, 0.2), [14.0])
    lake = _grid(np.arange(55, 75, 0.5), [-1.0], np.arange(55, 75, 0.5))
    quay = _grid(np.arange(55, 75, 1.0), [-2.5], [56.0])
    pts = np.concatenate([wall, air, behind, lake, quay])
    normal = np.tile(UP, (len(pts), 1))
    return GooglePoints(pts, np.full((len(pts), 3), 0.4), np.full(len(pts), 0.3), normal, ["credit Google"]), \
        dict(wall=len(wall), air=len(air), behind=len(behind), lake=len(lake), quay=len(quay))


def test_merged_google_stands_for_the_map_and_takes_the_water():
    scene, world = _scene(), _world()
    before = copy.deepcopy(world.json)
    points, n = _google()
    scene, merged = google(scene, Site(scene), world, points, Settings(google="google.ply"))
    assert (scene.google, scene.land, scene.roads, scene.buildings, scene.blocks) == ("google.ply", None, None, None, None)
    assert scene.water == "water.json" and scene.life == "life.json"
    kept = merged.points["google.ply"]
    assert kept["comments"] == ["credit Google", "merged"]
    assert len(kept["pts"]) == n["wall"] + n["behind"] + n["quay"]            # in front of the pano and on the water: gone
    level = merged.json["water.json"]["surfaces"][0]["level"]
    assert 0.9 < level < 1.5                     # the lake at Google's height now (1 m, as slid onto DA3's)
    assert world.json == before and merged.json["life.json"] == before["life.json"]   # the map's own world as it was
    assert set(kept) == {"pts", "cols", "gap", "normal", "near", "comments"}


def test_as_is_leaves_the_panos_points_alone_and_bent_keeps_a_true_wall():
    scene = _scene()
    was = scene.nodes[0].cloud.points
    scene, _ = google(scene, Site(scene), _world(), _google()[0], Settings(google="google.ply", bend=False))
    assert scene.nodes[0].cloud.points is was
    scene = _scene()
    scene, _ = google(scene, Site(scene), _world(), _google()[0], Settings(google="google.ply"))
    assert np.allclose(scene.nodes[0].cloud.points, was, atol=1e-6)            # Google agrees: nothing to bend


def test_already_merged_points_leave_the_world_as_it_is(tmp_path):
    points, _ = _google()
    path = str(tmp_path / "google.ply")
    write_ply(path, points.pts, points.cols, gap=points.gap, normal=points.normal, comments=["credit Google", "merged"])
    loaded = google_points(path)
    assert loaded.merged and loaded.credit == ["credit Google"] and np.allclose(loaded.pts, points.pts, atol=1e-5)
    scene, world = _scene(), _world()
    scene, out = google(scene, Site(scene), world, loaded, Settings(google=path))
    assert out is world and scene.google == "google.ply" and scene.land is None
