"""One-shot real-data pathfind test: fetch real metadata for a list of
clicked pano IDs, build the real isolated per-date graphs (same functions
prepare_pathfind uses), then run the real pathfind algorithm
(street_builder/reconstruction/walk_graph.py) with a fake test_edge
callback (no GPU needed -- the algorithm has zero GPU dependency) and
print a clean summary.

Usage: edit NODE_IDS below (first = start, rest = goals), then:
    ./.venv/bin/python3 scripts/mock_pathfind_run.py

FAIL_IDS: pano-id pairs to force-fail (simulating a known real DA3
result). Leave empty for "everything succeeds" (best case, tests search
logic only). Set RANDOM_FAIL_RATE instead for an unbiased randomized run.
"""
import asyncio
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from street_builder.build_graph.build_graph import build_corridor_graphs
from street_builder.reconstruction.walk_graph import run_pathfind_reconstruction

# ---- edit these for a new run -----------------------------------------
NODE_IDS = [
    "zF8DjYZpq8oGwq3wsxNYiw",   # start
    "H1fiiMDUmDsLIf0ULHh-vg",
    "4zMRdf_w2rkbckAgocFovg",
    "FiAceHrWAhROA925sezwrg",
    "A1jjoks7Qr_3Ut6kFvsEKw",
    "DlBCRcQInkAJTiXBrRM5Qg",
    "zQUKcenJ6Ppxj0GvKR9SWA",
    "3LNiBS5HSLzmeBVKypKZcQ",
    "m78sM9kVcLAKDBfeyfSepg",
    "C674PiypQiQqUbIoQRI7Yw",
]
FAIL_IDS: set[frozenset] = set()  # e.g. {frozenset({"zF8...", "H1fii..."})}
RANDOM_FAIL_RATE = None  # e.g. 0.4 for a randomized run instead of FAIL_IDS
RANDOM_SEED = 0
# -------------------------------------------------------------------------


def fetch_real_nodes(node_ids):
    from services.streetview_fetch import fetch_pano_by_id

    async def _fetch_all():
        return {pid: await fetch_pano_by_id(pid) for pid in node_ids}

    metas = asyncio.run(_fetch_all())
    missing = [pid for pid, m in metas.items() if m is None]
    if missing:
        raise RuntimeError(f"Failed to fetch metadata for: {missing}")
    return metas


def real_corridor_edges(node_ids, metas):
    """Edge exists between two of the given nodes iff they're REAL
    Street View neighbors (pano.links) -- matches how the actual UI only
    ever records an edge on a real click, never a guess."""
    neighbor_ids = {pid: {n["id"] for n in metas[pid]["neighbors"]} for pid in node_ids}
    edges = []
    for i, a in enumerate(node_ids):
        for b in node_ids[i + 1:]:
            if b in neighbor_ids[a] or a in neighbor_ids[b]:
                edges.append(((metas[a]["lat"], metas[a]["lon"]), (metas[b]["lat"], metas[b]["lon"])))
    return edges


def make_fake_test_edge(test_log, tested_pairs, rng=None):
    def fake_test_edge(path_a, path_b, test_id):
        id_a, id_b = os.path.basename(path_a), os.path.basename(path_b)
        pair = frozenset({id_a, id_b})
        test_log.append((id_a, id_b))
        if pair not in tested_pairs:
            if rng is not None:
                tested_pairs[pair] = rng.random() >= RANDOM_FAIL_RATE
            else:
                tested_pairs[pair] = pair not in FAIL_IDS
        if not tested_pairs[pair]:
            return None
        pose = (np.zeros(3), np.eye(3))
        return pose, pose, np.zeros((2, 3)), np.zeros((2, 3))
    return fake_test_edge


def main():
    print(f"=== fetching real metadata for {len(NODE_IDS)} node(s) ===")
    metas = fetch_real_nodes(NODE_IDS)
    for pid in NODE_IDS:
        m = metas[pid]
        print(f"  {pid}: ({m['lat']:.6f}, {m['lon']:.6f}) date={m['date']}, {len(m['neighbors'])} real neighbor(s)")

    corridor_edges = real_corridor_edges(NODE_IDS, metas)
    print(f"\n=== real edges among these {len(NODE_IDS)} nodes: {len(corridor_edges)} ===")
    id_by_latlon = {(metas[pid]["lat"], metas[pid]["lon"]): pid for pid in NODE_IDS}
    for a, b in corridor_edges:
        print(f"  {id_by_latlon[a]} -- {id_by_latlon[b]}")
    if not corridor_edges:
        raise RuntimeError("No real edges found among these nodes -- can't build a corridor.")

    start = (metas[NODE_IDS[0]]["lat"], metas[NODE_IDS[0]]["lon"])
    goals = [(metas[pid]["lat"], metas[pid]["lon"]) for pid in NODE_IDS[1:]]

    print(f"\n=== building isolated per-date graphs (network calls to Google/Apple) ===")
    date_graphs, points, adjacency = build_corridor_graphs(corridor_edges, start[0], start[1], goals)
    n_candidates_by_date = {g["date"]: sum(len(b) for b in g["dot_candidates"].values()) for g in date_graphs}
    print(f"{len(points)} corridor spine point(s), {len(date_graphs)} date graph(s) built:")
    for i, g in enumerate(date_graphs, 1):
        print(f"  {i}. {g['date']} ({n_candidates_by_date[g['date']]} candidate(s))")

    # Real download step, skipped here (no GPU/network image fetch needed
    # for this mock) -- fake paths keyed by real pano id, same shape
    # main.py's _download_date_graphs would produce.
    fake_date_graphs = [
        {"date": g["date"],
         "dot_candidates": {
             dot_idx: [(n["key"], f"/fake/{n['key']}", n["lat"], n["lon"]) for n in bucket]
             for dot_idx, bucket in g["dot_candidates"].items()
         }}
        for g in date_graphs
    ]

    test_log = []
    tested_pairs = {}
    if RANDOM_FAIL_RATE is not None:
        import random
        rng = random.Random(RANDOM_SEED)
        test_edge = make_fake_test_edge(test_log, tested_pairs, rng=rng)
        print(f"\n=== running pathfind (test_edge mocked, random fail_rate={RANDOM_FAIL_RATE}, seed={RANDOM_SEED}) ===")
    else:
        test_edge = make_fake_test_edge(test_log, tested_pairs)
        print(f"\n=== running pathfind (test_edge mocked, FAIL_IDS={FAIL_IDS or 'none -- everything succeeds'}) ===")

    segments, _ = run_pathfind_reconstruction(fake_date_graphs, points, adjacency, start[0], start[1], test_edge)

    print(f"\n=== RESULT ===")
    print(f"{len(test_log)} test call(s), {len(segments)} segment(s)")
    for pts, cols, path_edges, date, reached_all, node_positions, frame_poses in segments:
        print(f"  date={date}: {len(path_edges)} hop(s), {'FULL' if reached_all else 'partial'} coverage")
        for a, b in path_edges:
            print(f"    {a} -> {b}")


if __name__ == "__main__":
    main()
