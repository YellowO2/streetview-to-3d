"""CLI driver for staged, chunked corridor reconstruction against a
deployed street-view-to-3d Space -- see street_builder/tab.py's
handle_cli_add_chunk docstring for the API this drives.

Fetches the full real Street View graph for an area once (pure network,
no GPU) and caches it to ntu/graph.json so re-running doesn't re-hit the
network every time. Splits it into connected chunks via
map_selection.candidates.split_into_chunks -- NOT a flat slice of the
raw discovery-order edge list, which can put two unconnected branches
back-to-back purely by BFS queue timing (see that function's own
docstring) -- then grows a chain of --max-chunks mutually-adjacent
chunks (starting from one real known_adjacent_chunk_pairs entry) and
runs them one at a time via cli_add_chunk, each incrementally bridged
onto whatever's currently checkpointed in the dataset repo -- no
re-verifying earlier merges, unlike the old all-at-once join. The
checkpoint is already the final, viewable result after every call -- no
separate finalize step needed.

Usage:
    python tests/staged_corridor_test.py --center 1.3481742,103.6836485 --radius 750
    python tests/staged_corridor_test.py --center 1.3481742,103.6836485 --radius 750 --refetch
"""
import argparse
import json
import os
import sys

from gradio_client import Client

from street_builder.map_selection.candidates import expand_area, split_into_chunks

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


def to_payload(chunk):
    start_lat, start_lon = chunk["start"]
    return {
        "chunk_id": chunk["chunk_id"],
        "start": [start_lat, start_lon],
        "goals": [[lat, lon] for lat, lon in chunk["goals"]],
        "edges": [[[a[0], a[1]], [b[0], b[1]]] for a, b in chunk["corridor_edges"]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="potato-bug/street-view-to-3d")
    parser.add_argument("--center", required=True, help="lat,lon")
    parser.add_argument("--radius", type=float, required=True, help="meters")
    parser.add_argument("--chunk-size", type=int, default=12, help="max nodes per chunk")
    parser.add_argument("--max-chunks", type=int, default=2, help="how many (mutually-adjacent) chunks to actually run")
    parser.add_argument("--refetch", action="store_true", help="ignore the local cache and re-fetch the graph")
    args = parser.parse_args()

    center_lat, center_lon = (float(x) for x in args.center.split(","))
    nodes, edges = fetch_or_load_graph(center_lat, center_lon, args.radius, args.refetch)
    if not edges:
        print("No edges found -- nothing to do.")
        sys.exit(1)

    all_chunks, known_adjacent_chunk_pairs = split_into_chunks(nodes, edges, chunk_size=args.chunk_size)
    print(f"Split into {len(all_chunks)} connected chunk(s), {len(known_adjacent_chunk_pairs)} known-adjacent pair(s).")
    if not known_adjacent_chunk_pairs:
        print("No adjacent chunk pairs found -- can't run a join test.")
        sys.exit(1)

    by_id = {c["chunk_id"]: c for c in all_chunks}
    # neighbors_of[cid] -- every OTHER chunk id known-adjacent to cid,
    # from the real graph -- used to compute each new chunk's
    # adjacent_ids argument (only the ones already added so far).
    neighbors_of = {}
    for a, b in known_adjacent_chunk_pairs:
        neighbors_of.setdefault(a, set()).add(b)
        neighbors_of.setdefault(b, set()).add(a)

    # Grow a chain of mutually-adjacent chunks, starting from one real
    # pair, so every chunk run is genuinely connected to the growing set.
    a, b = known_adjacent_chunk_pairs[0]
    chosen = [a, b]
    remaining_pairs = list(known_adjacent_chunk_pairs[1:])
    while len(chosen) < args.max_chunks:
        grown = False
        for pair in remaining_pairs:
            x, y = pair
            if x in chosen and y not in chosen:
                chosen.append(y)
                remaining_pairs.remove(pair)
                grown = True
                break
            if y in chosen and x not in chosen:
                chosen.append(x)
                remaining_pairs.remove(pair)
                grown = True
                break
        if not grown:
            break

    chunks = [by_id[cid] for cid in chosen]
    print(f"Running {len(chunks)} mutually-adjacent chunk(s): {chosen}")

    client = Client(args.space)
    added_so_far = []
    for idx, chunk in enumerate(chunks):
        cid = chunk["chunk_id"]
        payload = to_payload(chunk)
        adjacent_ids = sorted(neighbors_of.get(cid, set()) & set(added_so_far))
        print(f"\n--- {cid} ({idx + 1}/{len(chunks)}, {len(chunk['goals']) + 1} node(s), "
              f"{len(chunk['corridor_edges'])} edge(s), adjacent_ids={adjacent_ids}) ---")
        run_status = client.predict(json.dumps(payload), api_name="/cli_run_chunk")
        print(run_status)
        bridge_status = client.predict(json.dumps(adjacent_ids), api_name="/cli_bridge_chunk")
        print(bridge_status)
        added_so_far.append(cid)

    print("\nDone -- the checkpoint above (cli_join/current/ in the dataset repo) is already the current, viewable result.")


if __name__ == "__main__":
    main()
