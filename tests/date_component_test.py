"""Like tests/date_segment_test.py, but splits each date further: chunks
sharing a date but NOT graph-connected to each other (via the real
declared-adjacency graph) are separate segments, not one. Lumping
unconnected same-date chunks into one fit was misleading -- it forces a
single transform onto two genuinely separate physical clusters, which
looks like drift but is really just a bad grouping.

Usage:
    python -m tests.date_component_test --in /tmp/gps_alignment.json
"""
import argparse
import json

import numpy as np

from tests.date_segment_test import build_key_to_date, fit_similarity_2d
from tests.robust_gps_alignment import chunk_adjacency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="/tmp/gps_alignment.json")
    parser.add_argument("--out", default="/tmp/gps_date_components.json")
    args = parser.parse_args()

    with open(args.in_path) as f:
        data = json.load(f)

    print("Building node -> date map from every chunk's raw metadata...")
    key_to_date = build_key_to_date()
    adj = chunk_adjacency()

    by_chunk = {}
    for d in data:
        date = key_to_date.get(d["key"])
        if date is None or d["chunk_id"] is None:
            continue
        by_chunk.setdefault(d["chunk_id"], {"date": date, "nodes": []})["nodes"].append(d)

    # connected components restricted to same-date chunk pairs
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
    for a in by_chunk:
        for b in adj.get(a, set()):
            if b in by_chunk and by_chunk[a]["date"] == by_chunk[b]["date"]:
                union(a, b)

    components = {}
    for c in by_chunk:
        components.setdefault(find(c), []).append(c)
    components = sorted(components.values(), key=lambda comp: -sum(len(by_chunk[c]["nodes"]) for c in comp))

    print(f"\n{len(by_chunk)} chunk(s) -> {len(components)} date-connected component(s).")

    out_nodes = []
    summary = []
    for i, comp in enumerate(components):
        date = by_chunk[comp[0]]["date"]
        nodes = [n for c in comp for n in by_chunk[c]["nodes"]]
        label = f"{date} #{i}"
        if len(nodes) < 3:
            for n in nodes:
                out_nodes.append({**n, "component": label, "date": date, "fitted_en": n["real_en"], "residual_m": 0.0})
            summary.append((label, len(comp), len(nodes), None))
            continue
        src = np.array([n["da3_xz"] for n in nodes])
        dst = np.array([n["real_en"] for n in nodes])
        R, scale, t = fit_similarity_2d(src, dst)
        fitted = scale * (src @ R.T) + t
        residuals = np.linalg.norm(fitted - dst, axis=1)
        med = float(np.median(residuals))
        summary.append((label, len(comp), len(nodes), med))
        for n, f, r in zip(nodes, fitted, residuals):
            out_nodes.append({**n, "component": label, "date": date, "fitted_en": f.tolist(), "residual_m": float(r)})

    summary.sort(key=lambda s: -(s[3] or 0))
    print("\nComponent(s), worst first:")
    for label, n_chunks, n_nodes, med in summary:
        med_str = f"{med:.1f}m" if med is not None else "n/a (<3 nodes)"
        print(f"  {label}: {n_chunks} chunk(s), {n_nodes} node(s), median residual {med_str}")

    with open(args.out, "w") as f:
        json.dump(out_nodes, f)
    print(f"\nWrote {len(out_nodes)} node result(s) to {args.out}")


if __name__ == "__main__":
    main()
