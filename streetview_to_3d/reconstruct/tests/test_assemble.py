import numpy as np

from streetview_to_3d.panos.Corridor import Corridor
from streetview_to_3d.reconstruct.assemble import assemble
from streetview_to_3d.reconstruct.open_scene import open_scene
from streetview_to_3d.reconstruct.pieces import pieces
from streetview_to_3d.reconstruct.thinned import thinned


def _corridor():
    catalog = {f"google:{k}": {"dot": dot, "source": "google", "id": k, "lat": 1.0 + dot, "lon": 2.0,
                               "date": "2020-01", "heading": 0.1, "pitch": 0.0, "roll": 0.0}
               for k, dot in (("a", 0), ("b", 1), ("b2", 1), ("c", 2))}
    return Corridor(date_graphs=[], points=[], adjacency={0: [1], 1: [0, 2], 2: [1]},
                    elevations=[5.0, 6.0, 7.0], catalog=catalog, start=(1.0, 2.0), center=(1.0, 2.0))


def _segment(keys, n_points):
    clouds = {k: (np.random.default_rng(0).uniform(0, 1, (n_points, 3)), np.full((n_points, 3), 0.5))
              for k in keys}
    poses = {k: (np.zeros(3), np.eye(3), "path", 1.0, 2.0, 10, 12) for k in keys}
    edges = [(a, b, [10, 12], [11, 12]) for a, b in zip(keys, keys[1:])]
    return clouds, edges, "2020-01", poses


def test_a_scene_opens_with_each_dots_first_candidate_and_the_dots_links():
    sc = open_scene(_corridor())
    assert [n.pano.id for n in sc.nodes] == ["a", "b", "c"]
    assert [n.pano.elevation for n in sc.nodes] == [5.0, 6.0, 7.0]
    assert sc.adjacency == {"0": [1], "1": [0, 2], "2": [1]}


def test_the_bigger_piece_keeps_a_shared_place_and_nodes_hold_their_thinned_points():
    corridor = _corridor()
    big = _segment(["google:a", "google:b"], 50)
    small = _segment(["google:b2"], 50)
    sc = assemble(open_scene(corridor), pieces([small, big]), corridor)
    assert [n.pano.id for n in sc.nodes] == ["a", "b", "c"]           # b2 lost dot 1 to the bigger piece's b
    assert [(e.a, e.b) for e in sc.edges] == [(0, 1)]
    assert [n.ply for n in sc.nodes] == ["node_0.ply", "node_1.ply", None]   # c: no piece reached it
    assert all(len(n.cloud.points) == len(n.cloud.colours) <= 50 for n in sc.nodes[:2])


def test_thinned_as_panoramic_da3s_saver_thins(tmp_path):
    from panoramic_da3 import Saver
    from streetview_to_3d.tools.ply import read_ply, write_ply
    rng = np.random.default_rng(1)
    pts = np.repeat(rng.uniform(0, 1, (300, 3)), 3, axis=0) + rng.normal(0, 0.004, (900, 3))
    cols = rng.uniform(0, 1, (900, 3))
    Saver(pts, str(tmp_path / "saver.ply"), colors=cols)
    cloud = thinned(pts, cols)
    write_ply(str(tmp_path / "ours.ply"), cloud.points, cloud.colours)
    assert (tmp_path / "saver.ply").read_bytes() == (tmp_path / "ours.ply").read_bytes()
    assert len(read_ply(str(tmp_path / "ours.ply"))[0]) < 900
