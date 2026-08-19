"""Reusable end-to-end test harness for the pathfind algorithm
(street_builder/reconstruction/walk_graph.py): a list of mock scenarios
(nodes/edges/points/fail-pairs), each run through the REAL algorithm with
a fake test_edge callback (a deterministic pass/fail table -- no GPU, no
DA3, since the algorithm has zero GPU dependency after the ZeroGPU-driven
split described in walk_graph.py's own module docstring).

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


def build_date_graphs(node_specs, edge_specs):
    """node_specs: {key: (offset_m, date)}. edge_specs: {key: [(other_key,
    dist_m), ...]} -- same shape build_corridor_graphs would hand the
    algorithm, just pre-isolated per date here since test fixtures are
    already written one date at a time. Returns [{"date", "nodes", "edges"}, ...],
    nodes as (key, path, lat, lon) tuples."""
    by_date: dict[str, list[str]] = {}
    for key, (off, date) in node_specs.items():
        by_date.setdefault(date, []).append(key)

    date_graphs = []
    for date, keys in by_date.items():
        key_set = set(keys)
        nodes = [(k, f"/fake/{k}", *latlon(node_specs[k][0])) for k in keys]
        edges = {k: [(o, d) for o, d in edge_specs.get(k, []) if o in key_set] for k in keys}
        date_graphs.append({"date": date, "nodes": nodes, "edges": edges})
    return date_graphs


def run_scenario(name, node_specs, edge_specs, point_offsets, fail_pairs, start_offset=0, **kwargs):
    """point_offsets: [offset_m, ...] -- corridor spine. fail_pairs: set of
    frozenset({key_a, key_b}) that should always fail.
    Returns (segments, test_log)."""
    date_graphs = build_date_graphs(node_specs, edge_specs)
    points = [latlon(off) for off in point_offsets]
    test_log = []

    def fake_test_edge(path_a, path_b, test_id):
        id_a, id_b = os.path.basename(path_a), os.path.basename(path_b)
        test_log.append((id_a, id_b))
        if frozenset({id_a, id_b}) in fail_pairs:
            return None
        pose_a = (np.zeros(3), np.eye(3))
        pose_b = (np.zeros(3), np.eye(3))
        return pose_a, pose_b, np.zeros((2, 3)), np.zeros((2, 3))

    start_lat, start_lon = latlon(start_offset)

    print(f"\n{'=' * 60}\nScenario: {name}\n{'=' * 60}")
    segments = run_pathfind_reconstruction(
        date_graphs, points, start_lat, start_lon, fake_test_edge, **kwargs,
    )
    print(f"-> {len(test_log)} test call(s): {test_log}")
    print(f"-> {len(segments)} segment(s): " +
          ", ".join(f"{d}({len(pe)} hops, {'full' if ra else 'partial'})" for _, _, pe, d, ra, _ in segments))
    return segments, test_log


# ---- scenarios ------------------------------------------------------
SCENARIOS = [
    {
        "name": "dead-end chain, one point unreachable by any date (live-log reproduction)",
        "node_specs": {
            "zF8": (0, "2022-05"), "4zMR": (10.8, "2022-05"), "A1jj": (21.6, "2022-05"),
        },
        "edge_specs": {
            "zF8": [("4zMR", 10.8)],
            "4zMR": [("zF8", 10.8), ("A1jj", 10.8)],
            "A1jj": [("4zMR", 10.8)],
        },
        "point_offsets": [0, 10.8, 21.6, 200],
        "fail_pairs": set(),
        "check": lambda segs, log: (
            len(log) == 2,
            f"expected exactly 2 test calls, got {len(log)}: {log}",
        ),
    },
    {
        "name": "closest candidate fails, second choice works, then dead-ends; second date bridges the gap",
        "node_specs": {
            "n1": (0, "2022-05"), "n2_bad": (10, "2022-05"), "n2": (10, "2022-05"), "n3": (20, "2022-05"),
            "m1": (0, "2020-07"), "m2": (50, "2020-07"),
        },
        "edge_specs": {
            "n1": [("n2_bad", 10.0), ("n2", 10.0)],
            "n2_bad": [("n1", 10.0)],
            "n2": [("n1", 10.0), ("n3", 10.0)],
            "n3": [("n2", 10.0)],
            "m1": [("m2", 50.0)],
            "m2": [("m1", 50.0)],
        },
        "point_offsets": [0, 10, 20, 50],
        "fail_pairs": {frozenset({"n1", "n2_bad"})},
        "check": lambda segs, log: (
            sum(1 for e in log if frozenset(e) == frozenset({"n1", "n2_bad"})) == 1
            and len(log) == 4
            and {s[3] for s in segs} == {"2022-05", "2020-07"}
            and any(s[4] for s in segs),
            f"expected 4 calls total (dead edge once), both dates combined, full coverage; got {len(log)} calls: {log}, dates: {[s[3] for s in segs]}",
        ),
    },
    {
        # Spacing matters: node offsets must clear point_cover_tolerance_m
        # (default 15m) and start_zone_m (default 5m) from each other, or
        # tolerance-based over-coverage or extra roots silently change what
        # the scenario is actually testing.
        "name": "two separate gaps on one date (x-x-o-o-o-x), bridged by a different pairing",
        "node_specs": {
            "N0": (0, "A"), "N1": (10, "A"), "N2": (60, "A"), "N3": (100, "A"), "N4": (140, "A"),
        },
        "edge_specs": {
            "N0": [("N1", 10.0), ("N2", 60.0)],
            "N1": [("N0", 10.0), ("N3", 90.0)],
            "N2": [("N0", 60.0), ("N3", 40.0)],
            "N3": [("N2", 40.0), ("N1", 90.0), ("N4", 40.0)],
            "N4": [("N3", 40.0)],
        },
        "point_offsets": [0, 10, 60, 100, 140],
        "fail_pairs": {frozenset({"N0", "N1"})},
        "check": lambda segs, log: (
            sum(1 for e in log if frozenset(e) == frozenset({"N0", "N1"})) == 1
            and len(log) <= 5
            and any(s[4] for s in segs),
            f"expected dead edge tested once, <=5 calls total, full coverage; got {len(log)} calls: {log}",
        ),
    },
]


def run_fuzz(seed, fail_rate, n_dates=3, nodes_per_date=6, spacing_m=15.0):
    """Randomized stress test: a synthetic multi-date graph (chain topology
    per date, some extra cross-links), pass/fail decided by a coin flip
    PER PAIR (same pair always gets the same outcome). Raises whatever the
    real algorithm raises -- fuzzing is exactly for catching crashes a
    curated scenario wouldn't think to construct."""
    import random
    rng = random.Random(seed)

    date_graphs = []
    all_offsets = []
    for d in range(n_dates):
        date = f"date{d}"
        offsets = sorted(rng.uniform(0, (nodes_per_date - 1) * spacing_m) for _ in range(nodes_per_date))
        all_offsets.extend(offsets)
        keys = [f"d{d}n{i}" for i in range(nodes_per_date)]
        nodes = [(k, f"/fake/{k}", *latlon(off)) for k, off in zip(keys, offsets)]
        edges = {k: [] for k in keys}
        for i in range(nodes_per_date - 1):
            dist = offsets[i + 1] - offsets[i]
            edges[keys[i]].append((keys[i + 1], dist))
            edges[keys[i + 1]].append((keys[i], dist))
        for _ in range(nodes_per_date // 2):
            i, j = rng.sample(range(nodes_per_date), 2)
            dist = abs(offsets[i] - offsets[j])
            edges[keys[i]].append((keys[j], dist))
            edges[keys[j]].append((keys[i], dist))
        date_graphs.append({"date": date, "nodes": nodes, "edges": edges})

    point_offsets = sorted(all_offsets)

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
        return (np.zeros(3), np.eye(3)), (np.zeros(3), np.eye(3)), np.zeros((2, 3)), np.zeros((2, 3))

    points = [latlon(off) for off in point_offsets]
    start_lat, start_lon = latlon(point_offsets[0])

    segments = run_pathfind_reconstruction(
        date_graphs, points, start_lat, start_lon, fake_test_edge,
    )
    return segments, test_log, tested_pairs


if __name__ == "__main__":
    import contextlib
    import io

    failures = []
    for sc in SCENARIOS:
        segs, log = run_scenario(
            sc["name"], sc["node_specs"], sc["edge_specs"], sc["point_offsets"], sc["fail_pairs"],
            early_exit_segments=4,
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
