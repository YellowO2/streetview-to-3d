"""One-time metadata-only fetch (no images) across the ENTIRE cached NTU
graph -- gathers, for every real node, which Google/Apple panos exist
nearby and which real capture dates they're on. Same per-dot logic as
fetch_nodes.fetch_corridor_nodes, but with CHECKPOINTING -- at ~2852
dots and no progress logging, a single run takes long enough that losing
it to an interruption (confirmed: lost a ~50 minute run with zero
output, since the plain version only wrote its file once at the very
end) is a real risk, not a hypothetical one. Saves progress every
CHECKPOINT_EVERY dots and resumes from wherever it left off.

This is the metadata this session's date-selection redesign needs: pick
the best 1-2 dates for the WHOLE campus once (instead of each chunk
independently ranking its own dates, which is why chunk-boundary nodes
kept ending up on mismatched dates), before ever downloading a single
image.

Usage:
    python tests/fetch_ntu_metadata.py
    python tests/fetch_ntu_metadata.py --refetch   # ignore any existing checkpoint
"""
import argparse
import asyncio
import json
import os
import time

from services.geo import haversine_m
from services.streetview_fetch import fetch_pano_by_id, format_date
from street_builder.build_graph.fetch_nodes import POINT_MAX_DIST_M, corridor_points
from street_builder.map_selection.candidates import MAX_NODES, apple_tile_panos, nearby_nodes, node_key

GRAPH_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ntu", "graph_full.json")
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ntu", "fetch_metadata.partial.json")
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ntu", "fetch_metadata.json")
CHECKPOINT_EVERY = 25


def save_checkpoint(points, adjacency, buckets, seen_google_ids, seen_apple_ids, done_dots):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({
            "points": points, "adjacency": adjacency,
            "buckets": {str(i): [{k: v for k, v in n.items() if k != "_pano"} for n in b] for i, b in buckets.items()},
            "seen_google_ids": sorted(seen_google_ids), "seen_apple_ids": sorted(seen_apple_ids),
            "done_dots": sorted(done_dots),
        }, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refetch", action="store_true", help="ignore any existing checkpoint, start over")
    args = parser.parse_args()

    with open(GRAPH_PATH) as f:
        g = json.load(f)
    nodes, key_edges = g["nodes"], [tuple(e) for e in g["edges"]]
    by_key = {n["key"]: n for n in nodes}
    edges = [((by_key[a]["lat"], by_key[a]["lon"]), (by_key[b]["lat"], by_key[b]["lon"])) for a, b in key_edges]
    points, adjacency = corridor_points(edges)
    print(f"Loaded {len(nodes)} node(s), {len(key_edges)} edge(s) -> {len(points)} dot(s)")

    buckets = {i: [] for i in range(len(points))}
    seen_google_ids, seen_apple_ids, done_dots = set(), set(), set()

    if not args.refetch and os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            ck = json.load(f)
        if ck["points"] == [list(p) for p in points]:
            buckets = {int(k): v for k, v in ck["buckets"].items()}
            for i in range(len(points)):
                buckets.setdefault(i, [])
            seen_google_ids = set(ck["seen_google_ids"])
            seen_apple_ids = set(ck["seen_apple_ids"])
            done_dots = set(ck["done_dots"])
            print(f"Resuming from checkpoint: {len(done_dots)}/{len(points)} dot(s) already done")
        else:
            print("Checkpoint doesn't match this graph -- starting fresh")

    t0 = time.monotonic()
    since_checkpoint = 0
    for i, (lat, lon) in enumerate(points):
        if i in done_dots:
            continue

        try:
            google_candidates, _ = nearby_nodes(lat, lon, radius_m=POINT_MAX_DIST_M, max_nodes=MAX_NODES)
        except Exception as e:
            print(f"Google lookup failed near ({lat}, {lon}): {e}")
            google_candidates = []
        for gc in google_candidates:
            if gc["id"] in seen_google_ids:
                continue
            seen_google_ids.add(gc["id"])
            try:
                meta = asyncio.run(fetch_pano_by_id(gc["id"]))
            except Exception as e:
                print(f"Google date lookup failed for {gc['id']}: {e}")
                continue
            if not meta:
                continue
            for entry in meta["dates"]:
                buckets[i].append({
                    "key": node_key("google", entry["id"]), "source": "google", "id": entry["id"],
                    "lat": gc["lat"], "lon": gc["lon"], "date": entry["label"],
                })

        try:
            apple_candidates = apple_tile_panos(lat, lon)
        except Exception as e:
            print(f"Apple lookup failed near ({lat}, {lon}): {e}")
            apple_candidates = {}
        for p in apple_candidates.values():
            if p.id in seen_apple_ids:
                continue
            if haversine_m(lat, lon, p.lat, p.lon) > POINT_MAX_DIST_M:
                continue
            seen_apple_ids.add(p.id)
            buckets[i].append({
                "key": node_key("apple", p.id), "source": "apple", "id": p.id,
                "lat": p.lat, "lon": p.lon, "date": format_date(p.date),
            })

        done_dots.add(i)
        since_checkpoint += 1
        if since_checkpoint >= CHECKPOINT_EVERY:
            save_checkpoint(points, adjacency, buckets, seen_google_ids, seen_apple_ids, done_dots)
            since_checkpoint = 0
            elapsed = time.monotonic() - t0
            rate = len(done_dots) / elapsed if elapsed else 0
            eta_s = (len(points) - len(done_dots)) / rate if rate else float("inf")
            print(f"[checkpoint] {len(done_dots)}/{len(points)} dot(s) done, {elapsed:.0f}s elapsed, ~{eta_s:.0f}s remaining")

    save_checkpoint(points, adjacency, buckets, seen_google_ids, seen_apple_ids, done_dots)
    total_candidates = sum(len(b) for b in buckets.values())
    print(f"Done: {total_candidates} total candidate(s) across all dots")

    with open(OUT_PATH, "w") as f:
        json.dump({"points": points, "adjacency": adjacency, "buckets": {str(i): b for i, b in buckets.items()}}, f)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
