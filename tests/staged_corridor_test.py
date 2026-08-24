"""CLI driver for staged, chunked corridor reconstruction against a
deployed street-view-to-3d Space -- see street_builder/tab.py's
handle_cli_run_chunk/handle_cli_join docstrings for the API this drives.

Fetches the full real Street View graph for an area once (pure network,
no GPU) and caches it to ntu/graph.json so re-running doesn't re-hit the
network every time. Slices its edges into fixed-size chunks in discovery
order, then runs the first --max-chunks of them (default 2) against the
Space's cli_run_chunk endpoint and joins them via cli_join -- a small,
cheap first look at whether two adjacent chunks reconstruct and stitch
correctly, before ever trying the whole area.

Usage:
    python tests/staged_corridor_test.py --center 1.3483,103.6831 --radius 750
    python tests/staged_corridor_test.py --center 1.3483,103.6831 --radius 750 --refetch
"""
import argparse
import json
import os
import sys

from gradio_client import Client

from street_builder.map_selection.candidates import expand_area

CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ntu", "graph.json")


def fetch_or_load_graph(center_lat, center_lon, radius_m, refetch):
    if not refetch and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cached = json.load(f)
        print(f"Loaded cached graph from {CACHE_PATH}: {len(cached['nodes'])} node(s), {len(cached['edges'])} edge(s).")
        return cached["nodes"], [tuple(e) for e in cached["edges"]]

    print(f"Fetching real graph within {radius_m}m of ({center_lat}, {center_lon})...")
    nodes, edges = expand_area(center_lat, center_lon, radius_m)
    print(f"Found {len(nodes)} node(s), {len(edges)} edge(s).")
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump({"center": [center_lat, center_lon], "radius_m": radius_m, "nodes": nodes, "edges": list(edges)}, f)
    print(f"Cached to {CACHE_PATH}.")
    return nodes, edges


def build_chunks(nodes, edges, chunk_size):
    by_key = {n["key"]: n for n in nodes}
    chunks = []
    for i in range(0, len(edges), chunk_size):
        chunk_edges = edges[i:i + chunk_size]
        keys = sorted({k for e in chunk_edges for k in e})
        if len(keys) < 2:
            continue
        start_key, goal_keys = keys[0], keys[1:]
        chunks.append({
            "chunk_id": i // chunk_size,
            "start": [by_key[start_key]["lat"], by_key[start_key]["lon"]],
            "goals": [[by_key[k]["lat"], by_key[k]["lon"]] for k in goal_keys],
            "edges": [[[by_key[a]["lat"], by_key[a]["lon"]], [by_key[b]["lat"], by_key[b]["lon"]]]
                      for a, b in chunk_edges],
        })
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="potato-bug/street-view-to-3d")
    parser.add_argument("--center", required=True, help="lat,lon")
    parser.add_argument("--radius", type=float, required=True, help="meters")
    parser.add_argument("--chunk-size", type=int, default=12, help="max edges per chunk")
    parser.add_argument("--max-chunks", type=int, default=2, help="how many chunks to actually run")
    parser.add_argument("--refetch", action="store_true", help="ignore the local cache and re-fetch the graph")
    args = parser.parse_args()

    center_lat, center_lon = (float(x) for x in args.center.split(","))
    nodes, edges = fetch_or_load_graph(center_lat, center_lon, args.radius, args.refetch)
    if not edges:
        print("No edges found -- nothing to do.")
        sys.exit(1)

    chunks = build_chunks(nodes, edges, args.chunk_size)[:args.max_chunks]
    print(f"Running {len(chunks)} chunk(s) of up to {args.chunk_size} edges each.")

    client = Client(args.space)
    pairs = []
    prev_chunk_id = None
    for idx, chunk in enumerate(chunks):
        print(f"\n--- chunk {chunk['chunk_id']} ({idx + 1}/{len(chunks)}, "
              f"{len(chunk['goals']) + 1} node(s), {len(chunk['edges'])} edge(s)) ---")
        status = client.predict(json.dumps(chunk), api_name="/cli_run_chunk")
        print(status)
        if prev_chunk_id is not None:
            pairs.append([prev_chunk_id, chunk["chunk_id"]])
        prev_chunk_id = chunk["chunk_id"]

    print("\nJoining...")
    result = client.predict(json.dumps(pairs), api_name="/cli_join")
    print(result)


if __name__ == "__main__":
    main()
