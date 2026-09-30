from streetview_to_3d.build_street_graph.date_ranking import date_connects, rank_dates
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


def test_a_date_covering_new_dots_ranks_above_repeat_drives():
    # a road (dots 0-3) driven three times, a park path (4-6) walked once
    def pano(date):
        return {"date": date}
    buckets = {i: [pano("2023-04"), pano("2022-09"), pano("2017-09")] for i in range(4)}
    buckets.update({i: [pano("2015-12")] for i in range(4, 7)})
    assert rank_dates(buckets) == ["2023-04", "2015-12", "2022-09", "2017-09"]
