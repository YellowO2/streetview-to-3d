"""CLI driver for binary-tree, metadata-only chunk merging against a
deployed street-view-to-3d Space -- see street_builder/tab.py's
handle_cli_merge_group/handle_cli_assemble docstrings for the API this
drives.

Same chunk-growing logic as tests/staged_corridor_test.py (real
known_adjacent_chunk_pairs, chained from one real pair), but instead of
incrementally bridging chunk-by-chunk onto one growing point cloud
(cli_bridge_chunk), pairs up N mutually-adjacent chunks into a binary
tree and merges level by level, metadata-only (cli_merge_group) --
N -> N/2 -> N/4 -> ... -> 1, then a single cli_assemble at the very end
resolves the root into an actual point cloud. No point-cloud data moves
until that last step, regardless of how many chunks are in the tree.

Usage:
    python tests/tree_merge_test.py
    python tests/tree_merge_test.py --num-chunks 8 --chunk-size 20
"""
import argparse
import json
import os
import time

from gradio_client import Client

from street_builder.build_graph.global_dates import split_cover_into_chunks

NTU_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ntu")


def _summary(status_html):
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
    dot_to_chunk = {d: c["chunk_id"] for c in all_chunks for d in c["dots"]}
    result = {c["chunk_id"]: set() for c in all_chunks}
    for d, cid in dot_to_chunk.items():
        for nb in adjacency.get(d, []):
            cb = dot_to_chunk.get(nb)
            if cb and cb != cid:
                result[cid].add(tuple(points[d]))
    return result


def grow_chain(known_adjacent_chunk_pairs, n):
    """Same growth rule as staged_corridor_test.py: start from one real
    pair, repeatedly attach any remaining pair with exactly one end
    already in the chosen set -- guarantees every chunk added is
    genuinely adjacent to the growing set, and (since pairs are only
    ever consumed once they're used to grow) that consecutive
    tree-pairing below reuses only real declared-adjacent edges."""
    a, b = known_adjacent_chunk_pairs[0]
    chosen = [a, b]
    remaining = list(known_adjacent_chunk_pairs[1:])
    while len(chosen) < n:
        grown = False
        for pair in remaining:
            x, y = pair
            if x in chosen and y not in chosen:
                chosen.append(y)
                remaining.remove(pair)
                grown = True
                break
            if y in chosen and x not in chosen:
                chosen.append(x)
                remaining.remove(pair)
                grown = True
                break
        if not grown:
            break
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="potato-bug/street-view-to-3d")
    parser.add_argument("--metadata", default=os.path.join(NTU_DIR, "fetch_metadata.json"))
    parser.add_argument("--cover", default=os.path.join(NTU_DIR, "date_cover.json"))
    parser.add_argument("--chunk-size", type=int, default=20, help="max dots per chunk")
    parser.add_argument("--num-chunks", type=int, default=8, help="how many (mutually-adjacent) chunks to run, tree-merged down to 1")
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
        print("No adjacent chunk pairs found -- can't run a tree-merge test.")
        raise SystemExit(1)

    by_id = {c["chunk_id"]: c for c in all_chunks}
    boundary_positions = boundary_positions_by_chunk(all_chunks, points, adjacency)

    chosen = grow_chain(known_adjacent_chunk_pairs, args.num_chunks)
    chunks = [by_id[cid] for cid in chosen]
    print(f"Running {len(chunks)} mutually-adjacent chunk(s): {chosen}")

    client = Client(args.space)

    # Step 1: run every chunk (its own GPU call, saves raw output --
    # unchanged from staged_corridor_test.py's own step 1).
    for idx, chunk in enumerate(chunks):
        cid = chunk["chunk_id"]
        payload = to_payload(chunk, protected_positions=boundary_positions.get(cid, set()))
        print(f"\n--- run {cid} ({idx + 1}/{len(chunks)}, {len(chunk['dots'])} dot(s), date={chunk['date']}) ---")
        t0 = time.monotonic()
        run_status = client.predict(json.dumps(payload), api_name="/cli_run_chunk")
        print(_summary(run_status))
        print(f"[timing] {cid}: {time.monotonic() - t0:.1f}s")

    # Step 2: pair up REAL-adjacent groups and merge level by level --
    # N -> N/2 -> ... -> 1. Position in `chosen` does NOT imply adjacency
    # between neighbors: grow_chain only guarantees each chunk is
    # adjacent to the growing SET as a whole, not to its immediate list
    # predecessor -- pairing by list position can (and did, in an earlier
    # version of this script) pair two chunks that were never actually
    # declared adjacent, which handle_cli_merge_group correctly rejects
    # with NoBridgeCandidatesError (a declared-adjacent pair with zero
    # real candidates raises loudly -- see join_segments.py; it does NOT
    # silently leave them as 2 pieces). So pairing here must only ever
    # match groups with a REAL declared-adjacent chunk pair between them.
    leaf_pairs = {frozenset(p) for p in known_adjacent_chunk_pairs}
    group_chunk_ids = {cid: frozenset({cid}) for cid in chosen}

    def groups_adjacent(ids_a, ids_b):
        return any(frozenset((x, y)) in leaf_pairs for x in ids_a for y in ids_b)

    level = list(chosen)
    level_num = 0
    while len(level) > 1:
        level_num += 1
        unmatched = list(level)
        next_level = []
        print(f"\n=== tree level {level_num}: {len(level)} group(s) ===")
        while len(unmatched) > 1:
            a = unmatched.pop(0)
            partner_idx = next((i for i, b in enumerate(unmatched)
                                 if groups_adjacent(group_chunk_ids[a], group_chunk_ids[b])), None)
            if partner_idx is None:
                print(f"--- {a} has no adjacent unmatched partner this level -- carries over ---")
                next_level.append(a)
                continue
            b = unmatched.pop(partner_idx)
            new_id = f"g_L{level_num}_{len(next_level)}"
            print(f"\n--- merge {a} + {b} -> {new_id} ---")
            t0 = time.monotonic()
            status = client.predict(a, b, new_id, api_name="/cli_merge_group")
            print(_summary(status))
            print(f"[timing] {new_id}: {time.monotonic() - t0:.1f}s")
            group_chunk_ids[new_id] = group_chunk_ids[a] | group_chunk_ids[b]
            next_level.append(new_id)
        next_level.extend(unmatched)  # last leftover if the level had an odd count
        if len(next_level) == len(level):
            print("No more adjacent pairs found this pass -- stopping (the chosen chunks don't form one connected tree).")
            break
        level = next_level

    if len(level) == 1:
        root = level[0]
        print(f"\n=== assembling root group {root} ===")
        t0 = time.monotonic()
        status = client.predict(root, api_name="/cli_assemble")
        print(_summary(status))
        print(f"[timing] assemble: {time.monotonic() - t0:.1f}s")
        print(f"\nDone -- root group '{root}' assembled into cli_join/current/ in the dataset repo.")
    else:
        print(f"\nEnded with {len(level)} disconnected group(s): {level} -- assemble each separately via cli_assemble if needed.")


if __name__ == "__main__":
    main()
