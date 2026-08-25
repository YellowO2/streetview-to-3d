"""CLI driver for staged, chunked corridor reconstruction against a
deployed street-view-to-3d Space -- see street_builder/tab.py's
handle_cli_run_chunk/handle_cli_bridge_chunk docstrings for the API this
drives.

Loads the pre-computed whole-NTU metadata + date cover (see
tests/fetch_ntu_metadata.py, tests/inspect_global_date_cover.py) and
splits it into connected, SINGLE-DATE chunks via
global_dates.split_cover_into_chunks -- not the raw selection graph, so
a chunk never straddles a date seam internally (see that function's own
docstring). Then grows a chain of --max-chunks mutually-adjacent chunks
(starting from one real known_adjacent_chunk_pairs entry) and runs them
one at a time via cli_run_chunk + cli_bridge_chunk (two separate GPU
calls, see handle_cli_run_chunk's own docstring for why), each
incrementally bridged onto whatever's currently checkpointed in the
dataset repo -- no re-verifying earlier merges, unlike the old
all-at-once join. The checkpoint is already the final, viewable result
after every call -- no separate finalize step needed.

Usage:
    python tests/staged_corridor_test.py
    python tests/staged_corridor_test.py --max-chunks 4 --chunk-size 20
"""
import argparse
import json
import os
import time

from gradio_client import Client

from street_builder.build_graph.global_dates import split_cover_into_chunks

NTU_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ntu")


def _summary(status_html):
    """Strips the <ul>...</ul> file-link dump each cli_* call returns --
    scannable progress in a long run needs the one-line status, not a
    wall of dataset URLs repeated on every single chunk."""
    return status_html.split("</p>")[0].removeprefix("<p>")


def to_payload(chunk, protected_positions=()):
    return {
        "chunk_id": chunk["chunk_id"],
        "dots": chunk["dots"],
        "date": chunk["date"],
        "protected_positions": sorted(protected_positions),
        "use_global_cover": True,
    }


def boundary_positions_by_chunk(all_chunks, points, adjacency):
    """Every dot COORDINATE that has a real adjacency edge crossing into
    a DIFFERENT chunk -- these are exactly the locations a later
    cli_bridge_chunk call needs to survive set_cover's coverage-only
    selection (see walk_graph.run_pathfind_reconstruction's
    protected_positions docstring). Returns {chunk_id: {(lat, lon), ...}}."""
    dot_to_chunk = {d: c["chunk_id"] for c in all_chunks for d in c["dots"]}
    result = {c["chunk_id"]: set() for c in all_chunks}
    for d, cid in dot_to_chunk.items():
        for nb in adjacency.get(d, []):
            cb = dot_to_chunk.get(nb)
            if cb and cb != cid:
                result[cid].add(tuple(points[d]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="potato-bug/street-view-to-3d")
    parser.add_argument("--metadata", default=os.path.join(NTU_DIR, "fetch_metadata.json"))
    parser.add_argument("--cover", default=os.path.join(NTU_DIR, "date_cover.json"))
    parser.add_argument("--chunk-size", type=int, default=20, help="max dots per chunk")
    parser.add_argument("--max-chunks", type=int, default=2, help="how many (mutually-adjacent) chunks to actually run")
    args = parser.parse_args()

    with open(args.metadata) as f:
        m = json.load(f)
    points = m["points"]
    adjacency = {int(k): v for k, v in m["adjacency"].items()}
    with open(args.cover) as f:
        cover = {int(k): v for k, v in json.load(f).items()}
    print(f"Loaded {len(points)} dot(s), {len(cover)} assigned to a date.")

    all_chunks, known_adjacent_chunk_pairs = split_cover_into_chunks(points, adjacency, cover, chunk_size=args.chunk_size)
    print(f"Split into {len(all_chunks)} single-date chunk(s), {len(known_adjacent_chunk_pairs)} known-adjacent pair(s).")
    if not known_adjacent_chunk_pairs:
        print("No adjacent chunk pairs found -- can't run a join test.")
        raise SystemExit(1)

    by_id = {c["chunk_id"]: c for c in all_chunks}
    boundary_positions = boundary_positions_by_chunk(all_chunks, points, adjacency)
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
        payload = to_payload(chunk, protected_positions=boundary_positions.get(cid, set()))
        adjacent_ids = sorted(neighbors_of.get(cid, set()) & set(added_so_far))
        print(f"\n--- {cid} ({idx + 1}/{len(chunks)}, {len(chunk['dots'])} dot(s), date={chunk['date']}, "
              f"adjacent_ids={adjacent_ids}, protected_positions={len(payload['protected_positions'])}) ---")
        t0 = time.monotonic()
        run_status = client.predict(json.dumps(payload), api_name="/cli_run_chunk")
        print(_summary(run_status))
        bridge_status = client.predict(cid, json.dumps(adjacent_ids), api_name="/cli_bridge_chunk")
        print(_summary(bridge_status))
        print(f"[timing] {cid}: {time.monotonic() - t0:.1f}s wall-clock (run+bridge, includes network)")
        added_so_far.append(cid)

    print("\nDone -- the checkpoint above (cli_join/current/ in the dataset repo) is already the current, viewable result.")


if __name__ == "__main__":
    main()
