"""CLI driver for the full-NTU chunked reconstruction campaign: runs every
still-ungenerated chunk from the whole-NTU date cover (see
tests/fetch_ntu_metadata.py / tests/inspect_global_date_cover.py), then
tree-merges every successfully-run chunk (metadata-only, see
street_builder/tab.py's handle_cli_merge_group/handle_cli_assemble) down
to as few connected root groups as the real declared-adjacency graph
allows, and assembles the largest one into the viewable checkpoint
(cli_join/current).

Resumable by construction: which chunks have already been run is read
back from the dataset repo itself (cli_raw/<chunk_id>/), not tracked
locally -- re-running this script after a crash/interruption skips
whatever's already there and just continues. A chunk that fails to run
(caught, logged, skipped) is simply excluded from the merge forest below
rather than aborting the whole campaign.

Usage:
    python -m tests.full_ntu_campaign
    python -m tests.full_ntu_campaign --chunk-size 20
"""
import argparse
import json
import os
import time

from gradio_client import Client
from huggingface_hub import HfApi

from street_builder.build_graph.global_dates import split_cover_into_chunks

NTU_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ntu")
DATASET_REPO = "potato-bug/ntu-reconstruction"


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


def already_run_chunk_ids():
    """Reads back which chunks already have raw output saved
    (cli_raw/<chunk_id>/...) -- the source of truth for resumability,
    not anything tracked locally."""
    api = HfApi()
    files = api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset")
    ids = set()
    for f in files:
        parts = f.split("/")
        if len(parts) >= 3 and parts[0] == "cli_raw" and parts[2].startswith("pathfind_metadata") and parts[2].endswith(".json"):
            ids.add(parts[1])
    return ids


class NoBridgeCandidates(Exception):
    """A declared-adjacent pair genuinely had zero real candidates within
    range -- a real-world coverage gap (e.g. the true boundary dot's pano
    wasn't available for one side's assigned date, so that chunk's real
    reconstructed node ended up anchored further away), not a bug.
    Deterministic -- retrying changes nothing, unlike a transient
    CancelledError -- so retry_predict raises this immediately instead of
    burning 3 retries finding out again."""


def retry_predict(client, *args, api_name, retries=3, backoff_s=15.0):
    """gradio_client calls occasionally hit a transient CancelledError
    (observed on a real run -- looked like a ZeroGPU queue hiccup, not a
    logic bug: the retry itself succeeded immediately after). Retries the
    exact same call a few times before giving up for real -- except a
    genuine NoBridgeCandidatesError (see NoBridgeCandidates above), which
    is never transient and raises immediately."""
    for attempt in range(retries):
        try:
            return client.predict(*args, api_name=api_name)
        except Exception as e:
            if "NoBridgeCandidatesError" in str(e):
                raise NoBridgeCandidates(str(e)) from e
            print(f"    attempt {attempt + 1}/{retries} failed: {type(e).__name__}: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(backoff_s)


def run_remaining_chunks(client, all_chunks, boundary_positions, batch_size=None):
    """Runs every not-yet-run chunk, or just the next `batch_size` of them
    if given -- lets a caller check in after a bounded amount of work
    (and roughly bounded wall-clock time) instead of committing to the
    whole remaining set blind. Resumable regardless: which chunks are
    "done" is always re-read from the dataset repo (see
    already_run_chunk_ids), so running this in batches across several
    separate invocations is exactly as safe as one big run."""
    by_id = {c["chunk_id"]: c for c in all_chunks}
    done = already_run_chunk_ids()
    todo = [c["chunk_id"] for c in all_chunks if c["chunk_id"] not in done]
    if batch_size is not None:
        todo = todo[:batch_size]
    print(f"\n{len(done)} chunk(s) already run, {len(todo)} in this batch.")

    failed = []
    for idx, cid in enumerate(todo):
        chunk = by_id[cid]
        payload = to_payload(chunk, protected_positions=boundary_positions.get(cid, set()))
        print(f"\n--- run {cid} ({idx + 1}/{len(todo)}, {len(chunk['dots'])} dot(s), date={chunk['date']}) ---")
        t0 = time.monotonic()
        try:
            status = retry_predict(client, json.dumps(payload), api_name="/cli_run_chunk")
            print(_summary(status))
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e} -- skipping, excluded from merge forest.")
            failed.append(cid)
            continue
        print(f"[timing] {cid}: {time.monotonic() - t0:.1f}s")

    if failed:
        print(f"\n{len(failed)} chunk(s) failed to run and will be excluded from merging: {failed}")
    return sorted(already_run_chunk_ids())


def existing_group_chunk_ids():
    """{group_id: frozenset(chunk_ids)} for every group already saved
    under cli_meta/ -- lets merge_forest skip re-doing a merge whose
    exact result is already sitting there from an earlier (interrupted
    or since-superseded) run, instead of blindly re-paying real
    GPU-billed DA3 work for something already done. Checked against the
    ACTUAL chunk coverage read back from the saved group, not just
    whether the group_id happens to exist -- group ids are deterministic
    given the same chunk_ids/known_adjacent_chunk_pairs inputs, but this
    is a free, cheap safety check against ever silently trusting a
    mismatched leftover (see this session's own cli_raw staleness bug
    for exactly this class of mistake, just one level up)."""
    from huggingface_hub import hf_hub_download

    api = HfApi()
    files = api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset")
    group_ids = sorted({f.split("/")[1] for f in files if f.startswith("cli_meta/") and f.endswith("/group_meta.json")})
    result = {}
    for gid in group_ids:
        path = hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=f"cli_meta/{gid}/group_meta.json")
        with open(path) as f:
            data = json.load(f)
        result[gid] = frozenset(ref[0] for piece in data["pieces"] for ref in piece["leaf_refs"])
    return result


def merge_forest(client, chunk_ids, known_adjacent_chunk_pairs):
    """Greedily pairs only REAL declared-adjacent groups at each level
    (N -> N/2 -> ... down to as few roots as the graph allows) -- see
    tests/tree_merge_test.py's own docstring for why list-position
    pairing is wrong. Resumable: a computed pairing whose target
    group_id already exists in cli_meta/ WITH the exact same chunk
    coverage is reused as-is rather than re-merged (see
    existing_group_chunk_ids). Returns a list of final root group ids
    (usually 1, more if the declared-adjacency graph itself has multiple
    connected components -- see this script's own module docstring)."""
    leaf_pairs = {frozenset(p) for p in known_adjacent_chunk_pairs}
    group_chunk_ids = {cid: frozenset({cid}) for cid in chunk_ids}
    existing = existing_group_chunk_ids()
    print(f"\n{len(existing)} already-merged group(s) found in cli_meta/ -- reused wherever they match this run's pairing.")

    def groups_adjacent(ids_a, ids_b):
        return any(frozenset((x, y)) in leaf_pairs for x in ids_a for y in ids_b)

    level = list(chunk_ids)
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
                next_level.append(a)
                continue
            b = unmatched.pop(partner_idx)
            new_id = f"g_L{level_num}_{len(next_level)}"
            expected_ids = group_chunk_ids[a] | group_chunk_ids[b]
            if existing.get(new_id) == expected_ids:
                print(f"--- {a} + {b} -> {new_id}: already merged, reusing (skipped GPU call) ---")
                group_chunk_ids[new_id] = expected_ids
                next_level.append(new_id)
                continue
            print(f"--- merge {a} + {b} -> {new_id} ---")
            t0 = time.monotonic()
            try:
                status = retry_predict(client, a, b, new_id, json.dumps(known_adjacent_chunk_pairs), api_name="/cli_merge_group")
            except NoBridgeCandidates:
                # Declared adjacent by the dot graph, but genuinely no real
                # candidate pair within range once actually reconstructed
                # (e.g. the true boundary dot's pano was missing for one
                # side's date) -- a real-world coverage gap, not a bug (see
                # this session's own chunk14<->chunk31 investigation: 40m
                # apart, just over BRIDGE_MAX_DIST_M). Leave both as
                # separate pieces rather than crashing the whole campaign.
                print(f"    {a} <-> {b}: genuine real-world gap (declared adjacent, but no real candidates in range) -- leaving separate.")
                next_level.append(a)
                next_level.append(b)
                continue
            print(f"    {_summary(status)}")
            print(f"    [timing] {new_id}: {time.monotonic() - t0:.1f}s")
            group_chunk_ids[new_id] = group_chunk_ids[a] | group_chunk_ids[b]
            next_level.append(new_id)
        next_level.extend(unmatched)
        if len(next_level) == len(level):
            print("No more adjacent pairs found this pass -- stopping.")
            break
        level = next_level

    return sorted(level, key=lambda g: -len(group_chunk_ids[g]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="potato-bug/street-view-to-3d")
    parser.add_argument("--metadata", default=os.path.join(NTU_DIR, "fetch_metadata.json"))
    parser.add_argument("--cover", default=os.path.join(NTU_DIR, "date_cover.json"))
    parser.add_argument("--chunk-size", type=int, default=20, help="max dots per chunk")
    parser.add_argument("--batch-size", type=int, default=None,
                         help="run at most this many NEW chunks, then stop WITHOUT merging -- "
                              "safe to re-run repeatedly (resumable) to check progress in bounded steps. "
                              "Omit to run every remaining chunk then merge+assemble everything.")
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

    boundary_positions = boundary_positions_by_chunk(all_chunks, points, adjacency)
    client = Client(args.space)

    t_start = time.monotonic()
    run_chunk_ids = run_remaining_chunks(client, all_chunks, boundary_positions, batch_size=args.batch_size)
    print(f"\n[timing] this batch's chunk runs: {(time.monotonic() - t_start) / 60:.1f} min total")

    if args.batch_size is not None:
        still_todo = len(all_chunks) - len(run_chunk_ids)
        print(f"\nBatch done -- {len(run_chunk_ids)}/{len(all_chunks)} chunk(s) have raw output, "
              f"{still_todo} remaining. Re-run with --batch-size to continue, or without it once "
              f"everything's run to merge+assemble.")
        return

    print(f"\n{len(run_chunk_ids)}/{len(all_chunks)} chunk(s) have raw output -- building merge forest.")
    t_merge = time.monotonic()
    roots = merge_forest(client, run_chunk_ids, known_adjacent_chunk_pairs)
    print(f"\n[timing] merge forest: {(time.monotonic() - t_merge) / 60:.1f} min total")
    print(f"\nEnded with {len(roots)} root group(s): {roots}")

    if roots:
        biggest = roots[0]
        print(f"\n=== assembling largest root group {biggest} into cli_join/current ===")
        t0 = time.monotonic()
        status = retry_predict(client, biggest, api_name="/cli_assemble")
        print(_summary(status))
        print(f"[timing] assemble: {time.monotonic() - t0:.1f}s")
        if len(roots) > 1:
            print(f"\n{len(roots) - 1} smaller disconnected root group(s) NOT assembled into the checkpoint "
                  f"(no declared adjacency found to the main group): {roots[1:]} -- assemble manually via cli_assemble if needed.")

    print(f"\n[timing] TOTAL: {(time.monotonic() - t_start) / 60:.1f} min")
    print("Done.")


if __name__ == "__main__":
    main()
