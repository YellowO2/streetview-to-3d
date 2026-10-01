import numpy as np

from streetview_to_3d.reconstruct.walk_graph import run_pathfind_reconstruction

M = 1 / 111320  # degrees of latitude per metre

# pano -> (solo score, views kept of 12)
RATINGS = {"A0": (5, 12), "A1": (1, 12), "B1": (9, 1)}


def _rate(path):
    score, kept = RATINGS[path]
    return score, (np.zeros(3), np.eye(3)), np.zeros((1, 3)), np.zeros((1, 3)), kept, 12


def _cand(key, dot):
    return (key, key, dot * 100 * M, 0.0)


def test_a_spot_no_link_reached_keeps_its_best_single_across_dates():
    # two spots 100 m apart, no links. Date A ranks first (keeps every view),
    # but its photo of spot 1 rates worst; date B's, ranked second, rates best
    points = [(0.0, 0.0), (100 * M, 0.0)]
    date_graphs = [
        {"date": "A", "dot_candidates": {0: [_cand("A0", 0)], 1: [_cand("A1", 1)]}},
        {"date": "B", "dot_candidates": {1: [_cand("B1", 1)]}},
    ]
    segments = run_pathfind_reconstruction(
        date_graphs, points, {0: [], 1: []}, 0.0, 0.0,
        test_edge=lambda *a: None, rate_pano=_rate)
    picked = sorted(key for s in segments for key in s[5])
    assert picked == ["A0", "B1"]
