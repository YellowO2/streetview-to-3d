"""Sanity-check report for the global (whole-NTU) date cover -- pure
metadata, no GPU. Run after tests/fetch_ntu_metadata.py has produced
ntu/fetch_metadata.json.

Usage:
    python tests/inspect_global_date_cover.py
"""
import json
import os

from street_builder.build_graph.date_ranking import rank_dates
from street_builder.build_graph.global_dates import build_date_cover, connected_components

META_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ntu", "fetch_metadata.json")
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ntu", "date_cover.json")


def main():
    with open(META_PATH) as f:
        m = json.load(f)
    points = m["points"]
    adjacency = {int(k): v for k, v in m["adjacency"].items()}
    buckets = {int(k): v for k, v in m["buckets"].items()}

    total_candidates = sum(len(b) for b in buckets.values())
    non_empty_dots = sum(1 for b in buckets.values() if b)
    print(f"{len(points)} dot(s) total, {non_empty_dots} non-empty, {total_candidates} candidate(s)")

    ranked = rank_dates(buckets)
    print(f"\n{len(ranked)} distinct date(s) found. Top 10 by coverage span/count:")
    for d in ranked[:10]:
        n = sum(1 for b in buckets.values() if any(x["date"] == d for x in b))
        print(f"  {d}: {n} dot(s)")

    cover = build_date_cover(points, adjacency, buckets, ranked)
    with open(OUT_PATH, "w") as f:
        json.dump({str(k): v for k, v in cover.items()}, f)
    print(f"\nSaved cover to {OUT_PATH}")

    reachable = {i for i, b in buckets.items() if b}
    unassigned = reachable - set(cover)
    print(f"\n{len(cover)}/{len(reachable)} reachable dot(s) assigned a date ({len(unassigned)} unassigned)")

    dates_used = {}
    for d in cover.values():
        dates_used[d] = dates_used.get(d, 0) + 1
    print(f"\n{len(dates_used)} distinct date(s) actually used in the cover:")
    for d, n in sorted(dates_used.items(), key=lambda t: -t[1]):
        print(f"  {d}: {n} dot(s)")

    seams = sum(1 for i in cover for j in adjacency.get(i, []) if j in cover and j > i and cover[i] != cover[j])
    total_edges = sum(1 for i in cover for j in adjacency.get(i, []) if j in cover and j > i)
    print(f"\n{seams}/{total_edges} adjacent-dot edge(s) are cross-date seams ({seams / total_edges * 100:.1f}%)" if total_edges else "\nno edges among assigned dots")

    regions = connected_components(set(cover), adjacency)
    same_date_regions = []
    for r in regions:
        by_date_in_region = {}
        for d in r:
            by_date_in_region.setdefault(cover[d], set()).add(d)
        same_date_regions.extend(by_date_in_region.values())
    print(f"{len(same_date_regions)} same-date contiguous region(s) total (this is roughly how many GPU walks/segments to expect, before any real DA3 failures)")
    sizes = sorted((len(r) for r in same_date_regions), reverse=True)
    print(f"region sizes: {sizes[:20]}{'...' if len(sizes) > 20 else ''}")


if __name__ == "__main__":
    main()
