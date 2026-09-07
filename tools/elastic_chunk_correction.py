"""Elastic drift correction at CHUNK granularity -- the unknown is one 2D
translation per CHUNK (not per node), applied uniformly to every node in
that chunk. This guarantees perfect within-chunk rigidity by construction
(a chunk's own shape literally cannot change, matching the fact that its
point cloud can never be split finer than itself -- see chunk_gps_test.py's
docstring) instead of relying on a large edge weight to approximate it.

x_i = node i's position under its chunk's own already-good fit
      (tools/chunk_gps_test.py's fitted_en)
y_i = x_i + delta[chunk(i)]   <- what we solve for

For a real-adjacency edge between node i (chunk a) and node j (chunk b):
  (y_i - y_j) - (x_i - x_j) = delta_a - delta_b
so preserving the edge's DA3-implied relative offset reduces to simply
wanting adjacent chunks' corrections to agree -- a per-edge smoothness
term on delta, weighted high within an island (already-trusted joints)
and low across islands (the actual drift-absorbing joints).

Every node also gets a weak GPS pull, purely as an anchor (otherwise the
whole graph floats under a free global translation).

Usage:
    python -m tools.elastic_chunk_correction --in /tmp/gps_alignment.json \
        --chunks /tmp/gps_chunks_15.json --out /tmp/gps_chunk_elastic.json
"""
import argparse
import json

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from tools.piece_gps_test import build_node_adjacency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="/tmp/gps_alignment.json")
    parser.add_argument("--chunks", default="/tmp/gps_chunks_15.json", help="chunk_gps_test.py output")
    parser.add_argument("--within-weight", type=float, default=10.0, help="how rigidly same-island chunk joints agree")
    parser.add_argument("--across-weight", type=float, default=5.0, help="how elastic inter-island joints are")
    parser.add_argument("--gps-weight", type=float, default=0.1, help="weak anchor pull toward raw GPS")
    parser.add_argument("--out", default="/tmp/gps_chunk_elastic.json")
    args = parser.parse_args()

    with open(args.in_path) as f:
        raw_data = json.load(f)
    with open(args.chunks) as f:
        chunk_data = json.load(f)

    by_key = {d["key"]: d for d in raw_data}
    island_of = {d["key"]: d["island"] for d in chunk_data}
    chunk_of = {d["key"]: d["chunk_id"] for d in chunk_data}
    before_fitted = {d["key"]: d["fitted_en"] for d in chunk_data}

    print("Building real node-to-node adjacency from the source dot graph...")
    node_adj = build_node_adjacency(raw_data)

    keys = [k for k in by_key if k in island_of and chunk_of[k]]
    idx = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    print(f"{n} node(s) with chunk/island assignment.")

    chunks = sorted({chunk_of[k] for k in keys})
    chunk_idx = {c: i for i, c in enumerate(chunks)}
    n_chunks = len(chunks)
    print(f"{n_chunks} chunk(s) -> {n_chunks * 2} unknown(s) (2D translation per chunk).")

    x = np.array([before_fitted[k] for k in keys])
    real = np.array([by_key[k]["real_en"] for k in keys])

    within_edges, across_edges = [], []
    seen_pairs = set()
    for k in keys:
        ca = chunk_idx[chunk_of[k]]
        for nb in node_adj.get(k, ()):
            if nb not in chunk_of or nb not in idx:
                continue
            cb = chunk_idx[chunk_of[nb]]
            if ca == cb:
                continue
            pair = tuple(sorted((ca, cb)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            (within_edges if island_of[k] == island_of[nb] else across_edges).append(pair)
    print(f"{len(within_edges)} within-island chunk joint(s) (rigid-ish), {len(across_edges)} across-island joint(s) (elastic).")

    # Normal equations for unknowns delta[0..n_chunks-1] (2D each):
    #   sum_edges w_e ||delta_a - delta_b||^2 + sum_i w_g ||delta[chunk(i)] - (real_i - x_i)||^2
    rows, cols, vals = [], [], []
    b = np.zeros((n_chunks, 2))
    wg = args.gps_weight

    def add(r, c, v):
        rows.append(r); cols.append(c); vals.append(v)

    gps_target = real - x  # per-node desired correction
    for i, k in enumerate(keys):
        c = chunk_idx[chunk_of[k]]
        add(c, c, wg)
        b[c] += wg * gps_target[i]

    for we, edge_list in ((args.within_weight, within_edges), (args.across_weight, across_edges)):
        for a, b_ in edge_list:
            add(a, a, we); add(b_, b_, we)
            add(a, b_, -we); add(b_, a, -we)

    A = sp.coo_matrix((vals, (rows, cols)), shape=(n_chunks, n_chunks)).tocsr()
    print("Solving sparse linear system...")
    delta = np.zeros((n_chunks, 2))
    delta[:, 0] = spla.spsolve(A, b[:, 0])
    delta[:, 1] = spla.spsolve(A, b[:, 1])

    y = np.array([x[i] + delta[chunk_idx[chunk_of[k]]] for i, k in enumerate(keys)])

    before_err = np.linalg.norm(x - real, axis=1)
    after_err = np.linalg.norm(y - real, axis=1)
    print(f"\nMedian error vs GPS -- before (per-chunk fit): {np.median(before_err):.1f}m, "
          f"after (elastic correction): {np.median(after_err):.1f}m")
    print(f"Max error vs GPS -- before: {before_err.max():.1f}m, after: {after_err.max():.1f}m")

    out = []
    for i, k in enumerate(keys):
        d = by_key[k]
        out.append({
            "key": k,
            "chunk_id": chunk_of[k],
            "island": island_of[k],
            "real_en": real[i].tolist(),
            "before_en": x[i].tolist(),
            "after_en": y[i].tolist(),
            "before_residual_m": float(before_err[i]),
            "after_residual_m": float(after_err[i]),
        })
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"\nWrote {len(out)} node result(s) to {args.out}")


if __name__ == "__main__":
    main()
