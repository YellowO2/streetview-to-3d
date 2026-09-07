"""Two-phase GPS correctness check, replacing the per-chunk/per-date/
per-edge tests before it:

Phase A -- split each chunk internally if it's already broken: fit a
chunk's own nodes to its own GPS positions; if some nodes don't fit,
cut them off (using the REAL underlying dot-adjacency graph, not just
chunk membership) and recurse on what's left, bottoming out once a
piece is too small (<3 nodes) to fit at all.

Phase B -- grow outward from those pieces (which are usually a whole
chunk, occasionally a fragment of one): repeatedly try merging a piece
with an adjacent piece, refit the combination, and keep the merge only
if everything still fits; otherwise that boundary stays cut. Repeats
from every not-yet-claimed piece until everything is assigned.

Node-level adjacency (not just chunk-level) comes from the SAME real
dot-adjacency graph street_builder/build_graph/global_dates.py uses to
build chunks in the first place -- so cuts happen at real gaps in the
walking path, not at arbitrary chunk boundaries.

Usage:
    python -m postprocess.gps_fit.discover_pieces --in /tmp/gps_alignment.json
"""
import argparse
import json
import os

import numpy as np

from paths import NTU_DIR
from postprocess.gps_fit.fit import fit_similarity_2d

MIN_NODES = 3
FIT_THRESHOLD_M = 20.0



def residuals_for(nodes):
    if len(nodes) < 2:
        return None, None
    src = np.array([n["da3_xz"] for n in nodes])
    dst = np.array([n["real_en"] for n in nodes])
    R, scale, t = fit_similarity_2d(src, dst)
    fitted = scale * (src @ R.T) + t
    return np.linalg.norm(fitted - dst, axis=1), (R, scale, t)


def connected_components(keys, node_adj):
    keys = set(keys)
    seen = set()
    comps = []
    for start in keys:
        if start in seen:
            continue
        comp = {start}
        frontier = [start]
        seen.add(start)
        while frontier:
            c = frontier.pop()
            for nb in node_adj.get(c, ()):
                if nb in keys and nb not in seen:
                    seen.add(nb)
                    comp.add(nb)
                    frontier.append(nb)
        comps.append(comp)
    return comps


def resolve(nodes_by_key, node_adj, threshold=FIT_THRESHOLD_M, min_nodes=MIN_NODES):
    """Recursively fits `nodes_by_key` (dict key->node) to GPS, cuts off
    badly-fitting nodes along real adjacency, and recurses on the
    resulting fragments. Returns a list of {key: node} dicts, each one
    either internally consistent or too small to say either way."""
    keys = list(nodes_by_key)
    if len(keys) < min_nodes:
        return [dict(nodes_by_key)]

    nodes = [nodes_by_key[k] for k in keys]
    residuals, _ = residuals_for(nodes)
    good_keys = {k for k, r in zip(keys, residuals) if r <= threshold}
    bad_keys = set(keys) - good_keys

    if not bad_keys:
        return [dict(nodes_by_key)]

    results = []
    for comp in connected_components(good_keys, node_adj):
        results.extend(resolve({k: nodes_by_key[k] for k in comp}, node_adj, threshold, min_nodes))
    for comp in connected_components(bad_keys, node_adj):
        results.extend(resolve({k: nodes_by_key[k] for k in comp}, node_adj, threshold, min_nodes))
    return results


def build_node_adjacency(data, warn_tolerance_m=10.0):
    """Node-level adjacency from the SAME real dot graph chunking used --
    matches each node's (lat, lon) back to its nearest source dot_id
    (small floating-point drift accumulates through the pipeline, so
    exact float equality misses most nodes -- nearest-neighbor is robust
    to that). Every real node was captured at some real dot, so there's
    no distance cutoff -- always match to the nearest one; a distance
    beyond warn_tolerance_m is just flagged as suspicious, not dropped."""
    from scipy.spatial import cKDTree

    with open(os.path.join(NTU_DIR, "fetch_metadata.json")) as f:
        m = json.load(f)
    points = m["points"]
    dot_adjacency = {int(k): v for k, v in m["adjacency"].items()}

    points_arr = np.array(points)
    tree = cKDTree(points_arr)
    m_per_deg = 111320.0

    node_latlon = np.array([[d["lat"], d["lon"]] for d in data])
    dists, idxs = tree.query(node_latlon)
    key_to_dot = {d["key"]: int(idx) for d, idx in zip(data, idxs)}

    far = [(d["key"], dist * m_per_deg) for d, dist in zip(data, dists) if dist * m_per_deg > warn_tolerance_m]
    if far:
        print(f"  ({len(far)} node(s) matched but >{warn_tolerance_m:.0f}m from their nearest dot -- worth a look: "
              f"{[(k, round(d, 1)) for k, d in far[:5]]}{'...' if len(far) > 5 else ''})")

    dot_to_keys = {}
    for key, dot_id in key_to_dot.items():
        dot_to_keys.setdefault(dot_id, []).append(key)

    node_adj = {}
    for d in data:
        key = d["key"]
        dot_id = key_to_dot[key]
        neighbors = set()
        for nb_dot in dot_adjacency.get(dot_id, []):
            neighbors.update(dot_to_keys.get(nb_dot, []))
        node_adj[key] = neighbors - {key}
    return node_adj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="/tmp/gps_alignment.json")
    parser.add_argument("--threshold", type=float, default=FIT_THRESHOLD_M)
    parser.add_argument("--out", default="/tmp/gps_pieces.json")
    args = parser.parse_args()

    with open(args.in_path) as f:
        data = json.load(f)

    print("Building real node-to-node adjacency from the source dot graph...")
    node_adj = build_node_adjacency(data)

    by_key = {d["key"]: d for d in data}
    by_chunk = {}
    for d in data:
        if d["chunk_id"] is None:
            continue
        by_chunk.setdefault(d["chunk_id"], []).append(d["key"])

    print(f"\n=== Phase A: splitting each of {len(by_chunk)} chunk(s) internally where broken ===")
    pieces = []
    for chunk_id, keys in by_chunk.items():
        sub = resolve({k: by_key[k] for k in keys}, node_adj, args.threshold)
        if len(sub) > 1:
            print(f"  {chunk_id} ({len(keys)} node(s)) split into {len(sub)} piece(s): {[len(s) for s in sub]}")
        pieces.extend(sub)

    print(f"\nPhase A done: {len(by_chunk)} chunk(s) -> {len(pieces)} piece(s).")

    print(f"\n=== Phase B: growing outward, merging adjacent pieces where the combined fit still holds ===")
    piece_id_of = {}
    for i, p in enumerate(pieces):
        for k in p:
            piece_id_of[k] = i
    piece_keys = {i: set(p) for i, p in enumerate(pieces)}

    def piece_adjacent(i, j):
        return any(nb in piece_keys[j] for k in piece_keys[i] for nb in node_adj.get(k, ()))

    claimed = set()
    islands = []
    for i in list(piece_keys):
        if i in claimed:
            continue
        island = {i}
        claimed.add(i)
        island_keys = set(piece_keys[i])
        _, fit = residuals_for([by_key[k] for k in island_keys])
        grown = True
        while grown:
            grown = False
            neighbor_ids = {j for i2 in island for j in piece_keys if j not in island
                             and piece_adjacent(i2, j)}
            for j in sorted(neighbor_ids):
                if j in claimed and j not in island:
                    continue
                cand_keys = piece_keys[j]
                cand_nodes = [by_key[k] for k in cand_keys]
                if fit is None:
                    # Island's own fit isn't established yet (too few points
                    # so far) -- can't test anything against it; just take
                    # the candidate and try to establish a fit next pass.
                    accept = True
                else:
                    R, scale, t = fit
                    src = np.array([n["da3_xz"] for n in cand_nodes])
                    dst = np.array([n["real_en"] for n in cand_nodes])
                    predicted = scale * (src @ R.T) + t
                    # Does the island's OWN, already-established transform
                    # correctly place the candidate's real-world positions?
                    # No joint refit involved -- if applying what we already
                    # trust to the candidate's points lands close to where
                    # they really are, they agree; if it takes real
                    # shifting to reconcile them, keep them separate.
                    error = np.linalg.norm(predicted - dst, axis=1)
                    accept = np.median(error) <= args.threshold
                if accept:
                    island.add(j)
                    island_keys |= cand_keys
                    claimed.add(j)
                    grown = True
                    _, fit = residuals_for([by_key[k] for k in island_keys])
        islands.append(island)

    print(f"\nPhase B done: {len(pieces)} piece(s) -> {len(islands)} island(s).")
    island_sizes = sorted((sum(len(piece_keys[i]) for i in isl) for isl in islands), reverse=True)
    print(f"Island sizes (node counts): {island_sizes[:20]}{'...' if len(island_sizes) > 20 else ''}")

    out_nodes = []
    for isl_idx, isl in enumerate(islands):
        combined_keys = {k for i in isl for k in piece_keys[i]}
        nodes = [by_key[k] for k in combined_keys]
        chunks = sorted({n["chunk_id"] for n in nodes if n["chunk_id"]})
        if len(nodes) >= 2:
            residuals, (R, scale, t) = residuals_for(nodes)
            src = np.array([n["da3_xz"] for n in nodes])
            fitted = scale * (src @ R.T) + t
        else:
            fitted = np.array([n["real_en"] for n in nodes])
            residuals = np.zeros(len(nodes))
        for n, f, r in zip(nodes, fitted, residuals):
            out_nodes.append({**n, "island": isl_idx, "island_chunks": chunks, "fitted_en": f.tolist(), "residual_m": float(r)})

    with open(args.out, "w") as f:
        json.dump(out_nodes, f)
    print(f"\nWrote {len(out_nodes)} node result(s) to {args.out}")


if __name__ == "__main__":
    main()
