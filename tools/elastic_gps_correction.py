"""Elastic (pose-graph-style) drift correction.

Per-island independent GPS fits (tools/piece_gps_test.py) already place
each island close to GPS -- within an island, DA3 didn't misbehave, it
just drifted a bit, and the per-island fit already absorbs that. But
each island's fit eats its own independent dose of GPS noise, so
boundaries between islands don't connect smoothly even though each side
is individually "close enough."

This builds ON TOP of the per-island fits (not the raw DA3 positions):
  - x_i = each node's already-good per-island fitted position
  - WITHIN-island edges: high weight (near-rigid) -- this shape is
    already validated against GPS, don't let it move
  - ACROSS-island edges (the joints Phase B rejected): kept in the
    graph, not cut, but with a LOW weight -- elastic, allowed to
    stretch/compress to absorb whatever drift separates the two islands
  - every node also gets a weak GPS pull, just to anchor the whole graph
    (otherwise it's only defined up to a floating global translation)

This is a convex quadratic in the per-node translation, so it's one
sparse linear solve -- no rotation search, no iteration.

Usage:
    python -m tools.elastic_gps_correction --in /tmp/gps_alignment.json \
        --pieces /tmp/gps_pieces_15c.json --out /tmp/gps_elastic.json
"""
import argparse
import json

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from postprocess.gps_fit.discover_pieces import build_node_adjacency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="/tmp/gps_alignment.json")
    parser.add_argument("--pieces", default="/tmp/gps_pieces_15c.json", help="island assignments from piece_gps_test.py")
    parser.add_argument("--within-weight", type=float, default=1000.0, help="how rigidly to trust already-good within-island shape")
    parser.add_argument("--across-weight", type=float, default=5.0, help="how elastic the inter-island joints are")
    parser.add_argument("--gps-weight", type=float, default=0.1, help="weak anchor pull toward raw GPS")
    parser.add_argument("--out", default="/tmp/gps_elastic.json")
    args = parser.parse_args()

    with open(args.in_path) as f:
        raw_data = json.load(f)
    with open(args.pieces) as f:
        piece_data = json.load(f)

    by_key = {d["key"]: d for d in raw_data}
    island_of = {d["key"]: d["island"] for d in piece_data}
    before_fitted = {d["key"]: d["fitted_en"] for d in piece_data}

    print("Building real node-to-node adjacency from the source dot graph...")
    node_adj = build_node_adjacency(raw_data)

    keys = [k for k in by_key if k in island_of]
    idx = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    print(f"{n} node(s) with both raw DA3 data and an island assignment.")

    # Start from the already-good per-island fitted positions, not the
    # raw DA3 frame -- that fit is what's already validated against GPS.
    x = np.array([before_fitted[k] for k in keys])
    real = np.array([by_key[k]["real_en"] for k in keys])

    within_edges, across_edges = [], []
    for k in keys:
        i = idx[k]
        for nb in node_adj.get(k, ()):
            if nb not in idx:
                continue
            j = idx[nb]
            if j <= i:
                continue
            (within_edges if island_of[k] == island_of[nb] else across_edges).append((i, j))
    print(f"{len(within_edges)} within-island edge(s) (rigid), {len(across_edges)} across-island joint(s) (elastic).")

    # Build normal equations for: sum_edges w_e ||(y_i-y_j)-(x_i-x_j)||^2
    # + sum_i w_g ||y_i - real_i||^2, separately per (x, y) coordinate.
    rows, cols, vals = [], [], []
    b = np.zeros((n, 2))
    wg = args.gps_weight

    def add(r, c, v):
        rows.append(r); cols.append(c); vals.append(v)

    for i in range(n):
        add(i, i, wg)
        b[i] += wg * real[i]

    for we, edge_list in ((args.within_weight, within_edges), (args.across_weight, across_edges)):
        for i, j in edge_list:
            add(i, i, we); add(j, j, we)
            add(i, j, -we); add(j, i, -we)
            diff = we * (x[i] - x[j])
            b[i] += diff
            b[j] -= diff

    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    print("Solving sparse linear system...")
    y = np.zeros((n, 2))
    y[:, 0] = spla.spsolve(A, b[:, 0])
    y[:, 1] = spla.spsolve(A, b[:, 1])

    before_err = np.linalg.norm(np.array([before_fitted[k] for k in keys]) - real, axis=1)
    after_err = np.linalg.norm(y - real, axis=1)
    print(f"\nMedian error vs GPS -- before (per-island fit): {np.median(before_err):.1f}m, "
          f"after (elastic correction): {np.median(after_err):.1f}m")
    print(f"Max error vs GPS -- before: {before_err.max():.1f}m, after: {after_err.max():.1f}m")

    out = []
    for i, k in enumerate(keys):
        d = by_key[k]
        out.append({
            "key": k,
            "chunk_id": d.get("chunk_id"),
            "island": island_of[k],
            "real_en": real[i].tolist(),
            "before_en": before_fitted[k],
            "after_en": y[i].tolist(),
            "before_residual_m": float(before_err[i]),
            "after_residual_m": float(after_err[i]),
        })
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"\nWrote {len(out)} node result(s) to {args.out}")


if __name__ == "__main__":
    main()
