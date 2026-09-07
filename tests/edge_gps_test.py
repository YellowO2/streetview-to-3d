"""Tests every declared-adjacent chunk PAIR independently against GPS
(not a seed-grown region) -- for each edge (chunk_a, chunk_b) in the real
adjacency graph, combine BOTH chunks' own full node sets (not just the
shared boundary -- 2 points can always be fit perfectly by a similarity
transform, so a meaningful test needs each chunk's full ~12-20 nodes),
fit one similarity transform to GPS, and mark the edge GOOD or BAD by
residual. Then islands = connected components of the graph using only
GOOD edges. Deterministic, no seed/order dependence, unlike
tests/robust_gps_alignment.py's region-growing version.

Usage:
    python -m tests.edge_gps_test --in /tmp/gps_alignment.json
"""
import argparse
import json

import numpy as np

from tests.robust_gps_alignment import chunk_adjacency, fit_similarity_2d

EDGE_THRESHOLD_M = 20.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="/tmp/gps_alignment.json")
    parser.add_argument("--threshold", type=float, default=EDGE_THRESHOLD_M)
    parser.add_argument("--out", default="/tmp/gps_edge_test.json")
    args = parser.parse_args()

    with open(args.in_path) as f:
        data = json.load(f)

    by_chunk = {}
    for d in data:
        if d["chunk_id"] is None:
            continue
        by_chunk.setdefault(d["chunk_id"], []).append(d)

    adj = chunk_adjacency()
    edges = sorted({tuple(sorted((a, b))) for a in adj for b in adj[a]})
    edges = [(a, b) for a, b in edges if a in by_chunk and b in by_chunk]
    print(f"{len(by_chunk)} chunk(s) with node data, {len(edges)} testable edge(s).")

    good_edges, bad_edges = [], []
    for a, b in edges:
        nodes = by_chunk[a] + by_chunk[b]
        src = np.array([n["da3_xz"] for n in nodes])
        dst = np.array([n["real_en"] for n in nodes])
        R, scale, t = fit_similarity_2d(src, dst)
        fitted = scale * (src @ R.T) + t
        residuals = np.linalg.norm(fitted - dst, axis=1)
        med = float(np.median(residuals))
        if med <= args.threshold:
            good_edges.append((a, b, med))
        else:
            bad_edges.append((a, b, med))

    print(f"\n{len(good_edges)} good edge(s), {len(bad_edges)} bad edge(s) (threshold {args.threshold:.0f}m).")
    print("\nWorst edges:")
    for a, b, med in sorted(bad_edges, key=lambda e: -e[2])[:20]:
        print(f"  {a} -- {b}: median residual {med:.1f}m")

    # connected components using only good edges
    parent = {c: c for c in by_chunk}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    for a, b, _ in good_edges:
        union(a, b)

    islands = {}
    for c in by_chunk:
        islands.setdefault(find(c), []).append(c)
    islands = sorted(islands.values(), key=lambda isl: -len(isl))

    print(f"\n{len(islands)} island(s):")
    for isl in islands[:15]:
        print(f"  {len(isl)} chunk(s): {sorted(isl)[:8]}{'...' if len(isl) > 8 else ''}")

    with open(args.out, "w") as f:
        json.dump({
            "good_edges": good_edges, "bad_edges": bad_edges,
            "islands": [sorted(isl) for isl in islands],
            "threshold": args.threshold,
        }, f)
    print(f"\nWrote result to {args.out}")


if __name__ == "__main__":
    main()
