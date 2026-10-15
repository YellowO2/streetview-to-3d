import json

import numpy as np

from streetview_to_3d.types.Cloud import Cloud
from streetview_to_3d.types.Node import Node
from streetview_to_3d.types.Pano import Pano
from streetview_to_3d.types.Scene import Scene
from streetview_to_3d.load import load
from streetview_to_3d.build_scene.save import save


def test_a_scene_saved_and_loaded_keeps_its_nodes_points_and_leaves_memory_out_of_scene_json(tmp_path):
    pts = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    cols = np.array([[0.2, 0.4, 0.6], [1.0, 0.0, 0.5]])
    labels = np.zeros((4, 8), np.uint8)
    node = Node(pano=Pano(source="google", id="x", lat=1.0, lon=2.0), ply="node_0.ply",
                cloud=Cloud(pts, cols, np.array([0.0, 1.0])), photo=("img.jpg", labels))
    save(Scene(center=[1.0, 2.0], nodes=[node, Node(pano=Pano(source="google", id="y", lat=1.0, lon=2.0))]),
         None, None, str(tmp_path))
    data = json.loads((tmp_path / "scene.json").read_text())
    assert "cloud" not in data["nodes"][0] and "photo" not in data["nodes"][0]
    assert (tmp_path / "labels" / "x.png").exists() and not (tmp_path / "labels" / "y.png").exists()
    back = load(str(tmp_path))
    assert back.nodes[1].cloud is None
    assert np.allclose(back.nodes[0].cloud.points, pts)
    assert np.allclose(back.nodes[0].cloud.colours, cols, atol=1 / 255)
