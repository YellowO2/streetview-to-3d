"""Reusable end-to-end test harness for the pathfind algorithm
(street_builder/reconstruction/walk_graph.py): a list of mock scenarios
(dots/dot_candidates/adjacency/fail-pairs), each run through the REAL
algorithm with a fake test_edge callback (a deterministic pass/fail table
-- no GPU, no DA3, since the algorithm has zero GPU dependency after the
ZeroGPU-driven split described in walk_graph.py's own module docstring).

Add a new scenario by appending a dict to SCENARIOS -- no new boilerplate
needed.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from street_builder.reconstruction.walk_graph import run_pathfind_reconstruction

M_PER_DEG_LAT = 111320.0


def latlon(offset_m):
    """Fake positions along a single line, offset_m meters from a fixed
    origin -- real haversine math still applies, just along one axis."""
    return (1.0 + offset_m / M_PER_DEG_LAT, 103.0)


def build_scenario_graph(dot_specs, dot_edges):
    """dot_specs: {dot_index: offset_m} -- one dot per corridor position.
    dot_edges: [(dot_a, dot_b), ...] -- structural dot-to-dot adjacency
    (same shape fetch_nodes.interpolate_points produces). Returns
    (points, adjacency): points is dot_index-ordered, adjacency is
    {dot_index: [neighbor_dot_index, ...]}."""
    n = max(dot_specs) + 1
    points = [latlon(dot_specs[i]) for i in range(n)]
    adjacency = {i: [] for i in range(n)}
    for a, b in dot_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    return points, adjacency


def build_date_graphs(node_specs, dot_of, points):
    """node_specs: {key: (dot_index, date)}. Returns [{"date",
    "dot_candidates"}, ...], dot_candidates values as (key, path, lat,
    lon) tuples, one per real node, grouped isolated per date."""
    by_date: dict[str, dict[int, list]] = {}
    for key, (dot, date) in node_specs.items():
        lat, lon = points[dot]
        by_date.setdefault(date, {}).setdefault(dot, []).append((key, f"/fake/{key}", lat, lon))
    return [{"date": date, "dot_candidates": dc} for date, dc in by_date.items()]


def run_scenario(name, node_specs, dot_specs, dot_edges, fail_pairs, start_dot=0, **kwargs):
    """node_specs: {key: (dot_index, date)}. dot_specs/dot_edges: see
    build_scenario_graph. fail_pairs: set of frozenset({key_a, key_b})
    that should always fail. Returns (segments, test_log)."""
    points, adjacency = build_scenario_graph(dot_specs, dot_edges)
    date_graphs = build_date_graphs(node_specs, None, points)
    test_log = []

    def fake_test_edge(path_a, path_b, test_id):
        id_a, id_b = os.path.basename(path_a), os.path.basename(path_b)
        test_log.append((id_a, id_b))
        if frozenset({id_a, id_b}) in fail_pairs:
            return None
        pose_a = (np.zeros(3), np.eye(3))
        pose_b = (np.zeros(3), np.eye(3))
        per_pano_pts = {id_a: np.zeros((1, 3)), id_b: np.zeros((1, 3))}
        per_pano_cols = {id_a: np.zeros((1, 3)), id_b: np.zeros((1, 3))}
        return pose_a, pose_b, np.zeros((2, 3)), np.zeros((2, 3)), per_pano_pts, per_pano_cols

    start_lat, start_lon = points[start_dot]

    print(f"\n{'=' * 60}\nScenario: {name}\n{'=' * 60}")
    segments = run_pathfind_reconstruction(
        date_graphs, points, adjacency, start_lat, start_lon, fake_test_edge, **kwargs,
    )
    print(f"-> {len(test_log)} test call(s): {test_log}")
    print(f"-> {len(segments)} segment(s): " +
          ", ".join(f"{d}({len(pe)} hops, {'full' if ra else 'partial'})" for _, _, pe, d, ra, _, _ in segments))
    return segments, test_log


# ---- scenarios ------------------------------------------------------
# Dot spacing kept well beyond the default point_cover_tolerance_m (15m)
# and edge_max_dist_m (18m) where the scenario needs those to matter --
# otherwise tolerance overlap alone would cover everything without the
# search actually walking anywhere.
SCENARIOS = [
    {
        "name": "dead-end chain, one point unreachable by any date (live-log reproduction)",
        "dot_specs": {0: 0, 1: 20, 2: 40, 3: 300},
        "dot_edges": [(0, 1), (1, 2), (2, 3)],
        "node_specs": {
            "zF8": (0, "2022-05"), "4zMR": (1, "2022-05"), "A1jj": (2, "2022-05"),
        },
        "fail_pairs": set(),
        "point_cover_tolerance_m": 1.0,
        "check": lambda segs, log: (
            len(log) == 2,
            f"expected exactly 2 test calls, got {len(log)}: {log}",
        ),
    },
    {
        "name": "closest candidate fails, second choice works, then dead-ends; second date bridges the gap",
        "dot_specs": {0: 0, 1: 20, 2: 40, 3: 100},
        "dot_edges": [(0, 1), (1, 2), (0, 3)],
        "node_specs": {
            "n1": (0, "2022-05"), "n2_bad": (1, "2022-05"), "n2": (1, "2022-05"), "n3": (2, "2022-05"),
            "m1": (0, "2020-07"), "m2": (3, "2020-07"),
        },
        "fail_pairs": {frozenset({"n1", "n2_bad"})},
        "point_cover_tolerance_m": 1.0,
        "check": lambda segs, log: (
            sum(1 for e in log if frozenset(e) == frozenset({"n1", "n2_bad"})) == 1
            and {s[3] for s in segs} == {"2022-05", "2020-07"},
            f"expected dead edge tested once, both dates combined; got {len(log)} calls: {log}, dates: {[s[3] for s in segs]}",
        ),
    },
    {
        "name": "skip-one bridges a single empty dot within edge_max_dist_m",
        "dot_specs": {0: 0, 1: 20, 2: 40, 3: 60},
        "dot_edges": [(0, 1), (1, 2), (2, 3)],
        "node_specs": {
            "N0": (0, "A"), "N1": (1, "A"), "N3": (3, "A"),  # dot 2 deliberately empty
        },
        "fail_pairs": set(),
        "point_cover_tolerance_m": 1.0,
        "edge_max_dist_m": 45.0,  # dot1 -> dot3 skip is 40m, must clear this
        "check": lambda segs, log: (
            len(segs) == 1 and set(segs[0][2]) == {("N0", "N1"), ("N1", "N3")},
            f"expected skip-one to bridge the empty dot and connect all 3 real panos in one piece; got segments: {[(s[3], s[2]) for s in segs]}",
        ),
    },
]


def run_fuzz(seed, fail_rate, n_dates=3, dots_per_date=6, spacing_m=20.0):
    """Randomized stress test: a synthetic multi-date corridor (chain
    topology, some extra branch dots), pass/fail decided by a coin flip
    PER PAIR (same pair always gets the same outcome). Raises whatever the
    real algorithm raises -- fuzzing is exactly for catching crashes a
    curated scenario wouldn't think to construct."""
    import random
    rng = random.Random(seed)

    n_dots = n_dates * dots_per_date
    points = [latlon(i * spacing_m) for i in range(n_dots)]
    adjacency = {i: [] for i in range(n_dots)}
    for i in range(n_dots - 1):
        adjacency[i].append(i + 1)
        adjacency[i + 1].append(i)
    for _ in range(n_dots // 3):
        a, b = rng.sample(range(n_dots), 2)
        if b not in adjacency[a]:
            adjacency[a].append(b)
            adjacency[b].append(a)

    date_graphs = []
    for d in range(n_dates):
        date = f"date{d}"
        dot_candidates = {}
        for i in range(n_dots):
            if rng.random() < 0.3:
                continue
            lat, lon = points[i]
            n_cands = rng.randint(1, 5)
            dot_candidates[i] = [(f"d{d}n{i}c{c}", f"/fake/d{d}n{i}c{c}", lat, lon) for c in range(n_cands)]
        if dot_candidates:
            date_graphs.append({"date": date, "dot_candidates": dot_candidates})

    tested_pairs = {}
    test_log = []

    def fake_test_edge(path_a, path_b, test_id):
        id_a, id_b = os.path.basename(path_a), os.path.basename(path_b)
        pair = frozenset({id_a, id_b})
        test_log.append((id_a, id_b))
        if pair not in tested_pairs:
            tested_pairs[pair] = rng.random() >= fail_rate
        if not tested_pairs[pair]:
            return None
        per_pano_pts = {id_a: np.zeros((1, 3)), id_b: np.zeros((1, 3))}
        per_pano_cols = {id_a: np.zeros((1, 3)), id_b: np.zeros((1, 3))}
        return (np.zeros(3), np.eye(3)), (np.zeros(3), np.eye(3)), np.zeros((2, 3)), np.zeros((2, 3)), per_pano_pts, per_pano_cols

    start_lat, start_lon = points[0]

    segments = run_pathfind_reconstruction(
        date_graphs, points, adjacency, start_lat, start_lon, fake_test_edge,
    )
    return segments, test_log, tested_pairs


if __name__ == "__main__":
    import contextlib
    import io

    failures = []
    for sc in SCENARIOS:
        kwargs = {k: v for k, v in sc.items()
                  if k not in ("name", "node_specs", "dot_specs", "dot_edges", "fail_pairs", "check")}
        segs, log = run_scenario(
            sc["name"], sc["node_specs"], sc["dot_specs"], sc["dot_edges"], sc["fail_pairs"],
            **kwargs,
        )
        ok, msg = sc["check"](segs, log)
        status = "PASS" if ok else "FAIL"
        print(f"-> {status}: {msg}" if not ok else f"-> {status}")
        if not ok:
            failures.append(sc["name"])

    print(f"\n{'=' * 60}\nFuzz: randomized graphs x randomized pass/fail\n{'=' * 60}")
    fuzz_failures = []
    for fail_rate in (0.2, 0.4, 0.6, 0.8):
        crashes = 0
        dupes = 0
        max_calls = 0
        for seed in range(30):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    segs, log, tested_pairs = run_fuzz(seed=seed * 1000 + int(fail_rate * 10), fail_rate=fail_rate)
            except Exception as e:
                crashes += 1
                print(f"  fail_rate={fail_rate} seed={seed}: CRASH {type(e).__name__}: {e}")
                continue
            dup = len(log) - len(set(frozenset(e) for e in log))
            dupes += dup
            max_calls = max(max_calls, len(log))
        print(f"fail_rate={fail_rate}: 30 seeds, crashes={crashes}, duplicate_calls={dupes}, max_calls_seen={max_calls}")
        if crashes or dupes:
            fuzz_failures.append(fail_rate)

    print(f"\n{'=' * 60}")
    if failures or fuzz_failures:
        if failures:
            print(f"{len(failures)}/{len(SCENARIOS)} curated scenario(s) FAILED: {failures}")
        if fuzz_failures:
            print(f"fuzz FAILED at fail_rate(s): {fuzz_failures}")
        sys.exit(1)
    print(f"All {len(SCENARIOS)} curated scenarios + fuzz (120 randomized runs) passed.")
