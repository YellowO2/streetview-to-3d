"""Chunk-granularity GPS correctness check -- like tests/piece_gps_test.py,
but treating each CHUNK as the atomic, rigid unit instead of splitting it
into sub-pieces. The stored point-cloud data has no per-node/per-frame
tag (see street_builder/reconstruction/join_segments.py's
_read_ply_points -- a chunk's .ply is just a fused (points, colors) blob),
so a chunk can never actually be cut finer than itself when exporting real
geometry. Building the GPS-consistency graph at chunk granularity from the
start keeps the analysis consistent with what we can actually produce.

Phase A: fit each chunk's own nodes to its own GPS positions -- ONE
similarity transform per chunk, no internal splitting.

Phase B: grow islands by merging adjacent chunks (real chunk-to-chunk
adjacency) wherever the island's own already-established transform,
applied directly to the candidate chunk's own nodes, still lands close to
that chunk's real GPS positions. Exactly the same accept/reject rule as
piece_gps_test.py's Phase B, just with chunks as the unit instead of
pieces.

Usage:
    python -m tests.chunk_gps_test --in /tmp/gps_alignment.json --threshold 15 --out /tmp/gps_chunks.json
"""
import argparse
import json

import numpy as np

from tests.robust_gps_alignment import chunk_adjacency, fit_similarity_2d

FIT_THRESHOLD_M = 15.0


def residuals_for(nodes):
    if len(nodes) < 2:
        return None, None
    src = np.array([n["da3_xz"] for n in nodes])
    dst = np.array([n["real_en"] for n in nodes])
    R, scale, t = fit_similarity_2d(src, dst)
    fitted = scale * (src @ R.T) + t
    return np.linalg.norm(fitted - dst, axis=1), (R, scale, t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="/tmp/gps_alignment.json")
    parser.add_argument("--threshold", type=float, default=FIT_THRESHOLD_M)
    parser.add_argument("--out", default="/tmp/gps_chunks.json")
    args = parser.parse_args()

    with open(args.in_path) as f:
        data = json.load(f)

    by_chunk = {}
    for d in data:
        if d["chunk_id"] is None:
            continue
        by_chunk.setdefault(d["chunk_id"], []).append(d)

    adj = chunk_adjacency()
    print(f"{len(by_chunk)} chunk(s) with node data, real adjacency graph has {len(adj)} chunk(s) with a neighbor.")

    print(f"\n=== Phase A: fitting each of {len(by_chunk)} chunk(s) whole (no internal splitting) ===")
    chunk_fit = {}
    for chunk_id, nodes in by_chunk.items():
        residuals, fit = residuals_for(nodes)
        med = float(np.median(residuals)) if residuals is not None else 0.0
        chunk_fit[chunk_id] = fit
        if residuals is not None and med > args.threshold:
            print(f"  {chunk_id} ({len(nodes)} node(s)): median residual {med:.1f}m (INTERNALLY INCONSISTENT, kept whole anyway)")

    print(f"\n=== Phase B: growing islands by merging adjacent chunks wherever the island's own fit still holds ===")
    claimed = set()
    islands = []
    for c in list(by_chunk):
        if c in claimed:
            continue
        island = {c}
        claimed.add(c)
        fit = chunk_fit[c]
        grown = True
        while grown:
            grown = False
            neighbor_ids = {nb for c2 in island for nb in adj.get(c2, set()) if nb not in island and nb in by_chunk}
            for j in sorted(neighbor_ids):
                if j in claimed and j not in island:
                    continue
                cand_nodes = by_chunk[j]
                if fit is None:
                    accept = True
                else:
                    R, scale, t = fit
                    src = np.array([n["da3_xz"] for n in cand_nodes])
                    dst = np.array([n["real_en"] for n in cand_nodes])
                    predicted = scale * (src @ R.T) + t
                    error = np.linalg.norm(predicted - dst, axis=1)
                    accept = np.median(error) <= args.threshold
                if accept:
                    island.add(j)
                    claimed.add(j)
                    grown = True
                    combined_nodes = [n for c2 in island for n in by_chunk[c2]]
                    _, fit = residuals_for(combined_nodes)
        islands.append(island)

    print(f"\nPhase B done: {len(by_chunk)} chunk(s) -> {len(islands)} island(s).")
    island_sizes = sorted((sum(len(by_chunk[c]) for c in isl) for isl in islands), reverse=True)
    print(f"Island sizes (node counts): {island_sizes[:20]}{'...' if len(island_sizes) > 20 else ''}")

    out_nodes = []
    for isl_idx, isl in enumerate(islands):
        nodes = [n for c in isl for n in by_chunk[c]]
        if len(nodes) >= 2:
            residuals, (R, scale, t) = residuals_for(nodes)
            src = np.array([n["da3_xz"] for n in nodes])
            fitted = scale * (src @ R.T) + t
        else:
            fitted = np.array([n["real_en"] for n in nodes])
            residuals = np.zeros(len(nodes))
        for n, f, r in zip(nodes, fitted, residuals):
            out_nodes.append({**n, "island": isl_idx, "island_chunks": sorted(isl), "fitted_en": f.tolist(), "residual_m": float(r)})

    with open(args.out, "w") as f:
        json.dump(out_nodes, f)
    print(f"\nWrote {len(out_nodes)} node result(s) to {args.out}")


if __name__ == "__main__":
    main()
