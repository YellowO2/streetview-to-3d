"""Sanity check for corridor_points' merge logic -- no network calls,
just re-derives dots at several merge thresholds straight from the
cached raw selection graph (ntu/graph_full.json) and plots the RAW
(pre-merge) nodes as a light backdrop with each threshold's MERGED
centroids overlaid, so a merge that's clustering the wrong things is
visible directly (e.g. a centroid sitting off to the side of its own
raw members, or two clearly-unrelated raw clusters merged into one).

Usage:
    python -m tests.visualize_merge_comparison
"""
import json
import os

import matplotlib.pyplot as plt

import street_builder.build_graph.fetch_nodes as fetch_nodes

NTU_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ntu")
THRESHOLDS = [0.0, 5.0, 8.0]


def main():
    with open(os.path.join(NTU_DIR, "graph_full.json")) as f:
        g = json.load(f)
    nodes, key_edges = g["nodes"], [tuple(e) for e in g["edges"]]
    by_key = {n["key"]: n for n in nodes}
    edges = [((by_key[a]["lat"], by_key[a]["lon"]), (by_key[b]["lat"], by_key[b]["lon"])) for a, b in key_edges]
    raw_points = [(n["lat"], n["lon"]) for n in nodes]

    fig, axes = plt.subplots(1, len(THRESHOLDS), figsize=(8 * len(THRESHOLDS), 8), sharex=True, sharey=True)

    for ax, threshold in zip(axes, THRESHOLDS):
        fetch_nodes.MERGE_DIST_M = threshold
        points, adjacency = fetch_nodes.corridor_points(edges)

        raw_lats, raw_lons = zip(*raw_points)
        ax.scatter(raw_lons, raw_lats, c="lightgray", s=6, label=f"raw ({len(raw_points)})", zorder=1)

        merged_lats, merged_lons = zip(*points)
        ax.scatter(merged_lons, merged_lats, c="crimson", s=10, alpha=0.6, label=f"merged dots ({len(points)})", zorder=2)

        ax.set_title(f"MERGE_DIST_M = {threshold:g}m -> {len(points)} dot(s)")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(NTU_DIR, "merge_comparison.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
