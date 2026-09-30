from streetview_to_3d.build_street_graph.date_ranking import date_connects
from streetview_to_3d.build_street_graph.fetch_nodes import corridor_points

M = 1 / 111320  # degrees of latitude per metre


def _north(m):
    return (m * M, 0.0)


def test_close_graphs_get_one_bridge():
    # a street 0-20 m north, and a path 30-50 m north that Google never
    # linked to it: 10 m apart at their nearest, so one bridge, there
    street = [(_north(0), _north(10)), (_north(10), _north(20))]
    path = [(_north(30), _north(40)), (_north(40), _north(50))]
    points, adjacency = corridor_points(street + path)
    by_m = {round(p[0] / M): i for i, p in enumerate(points)}
    assert by_m[30] in adjacency[by_m[20]]
    assert sum(len(ns) for ns in adjacency.values()) == 2 * 5


def test_far_graphs_stay_apart():
    street = [(_north(0), _north(10))]
    path = [(_north(100), _north(110))]
    points, adjacency = corridor_points(street + path)
    by_m = {round(p[0] / M): i for i, p in enumerate(points)}
    assert adjacency[by_m[10]] == [by_m[0]]
    assert adjacency[by_m[100]] == [by_m[110]]


def test_date_covering_only_the_far_graph_still_connects():
    # two graphs; this date has panos only on the far one, which holds a goal
    points = [_north(0), _north(10), _north(100), _north(110)]
    adjacency = {0: [1], 1: [0], 2: [3], 3: [2]}
    goals = [_north(10), _north(110)]
    assert date_connects({2: ["p"], 3: ["p"]}, adjacency, points, 0.0, 0.0, goals)
    # but a lone dot reaching no goal still doesn't
    assert not date_connects({2: ["p"]}, adjacency, points, 0.0, 0.0, [_north(10)])
