"""Combine independently-reconstructed pathfind segments into one merged
point cloud using real DA3 tests -- no GPS placement (see tests/gps.py
for the old GPS-anchoring approach, kept only for reference, no longer
used: nothing here needs the result oriented to true north/real-world
coordinates, only internally consistent).

Each pair of segments known (or suspected, in the no-known-adjacency
case) to touch gets bridged via _try_bridge: real lat/lon (already
carried per-node in frame_poses) picks which node pairs are worth a
real DA3 test, then the actual merge uses the DA3-derived relative
transform between the two segments' own local frames.

Since our chunking guarantees any pair this module is actually asked to
bridge came from a real, previously-confirmed graph edge, two touching
segments should always have SOME node pair within edge_max_dist_m of
each other -- if not, that's treated as a bug upstream (bad chunking,
a lost/corrupted node) and raises loudly.
"""
import time

import numpy as np

from services.geo import haversine_m
from street_builder.reconstruction.walk_graph import rigid_align

# Relaxed keep-rate vs. the main walk's 0.6 -- bridging only needs SOME
# real signal, and any real DA3 estimate beats independent GPS placement
# regardless of how weak. Only decides when a match is confident enough
# to stop searching early; never disqualifies a result from being used.
BRIDGE_KEEP_RATE = 0.5
# An average deviation-among-kept-views this large (not a single outlier
# -- those get filtered out already, see services.da3_ops.bridge_test_edge) means the
# surviving views still don't agree with each other, a real sign the
# pair is worse than usual -- only used to break ties when ranking
# attempts, never to discard a result outright.
BRIDGE_RIDICULOUS_DEV_M = 2.0
# Real DA3 calls spent trying to bridge one pair of pieces, capped
# regardless of how many (Ax, By) node pairs qualify by distance.
BRIDGE_MAX_ATTEMPTS = 10
# Real-world distance a candidate node pair must be within to even
# attempt bridging.
BRIDGE_MAX_DIST_M = 30.0


class NoBridgeCandidatesError(RuntimeError):
    """Raised when two segments that should be adjacent (per the caller's
    own chunking/graph data) have zero real node pairs within
    edge_max_dist_m of each other. Since our chunk boundaries always come
    from a real, previously-confirmed graph edge, this means something
    upstream is wrong (bad chunking, a node that failed to download, a
    corrupted position) -- not a normal outcome to silently work around."""


def _try_bridge(a, b, bridge_test_edge, edge_max_dist_m, deadline, bridge_test_id, expected_adjacent, refetch_path=None):
    """One pair's worth of bridge search: every (Ax, By) node pair within
    edge_max_dist_m, same-date-first then closest-first, up to
    BRIDGE_MAX_ATTEMPTS real tests. ALWAYS merges using whichever attempt
    came out best, however weak -- a clearly confident match (clears
    BRIDGE_KEEP_RATE on both sides, no bad-consensus red flag) stops the
    search early; otherwise every attempt is ranked and the best one
    wins once the attempt budget/deadline is hit.
    Returns (merged_segment, next_bridge_test_id), or (None,
    next_bridge_test_id) if real candidates existed but every DA3 attempt
    on them came back unusable (a genuine per-attempt failure, distinct
    from NoBridgeCandidatesError below) -- or if there were zero
    candidates AND expected_adjacent is False (a normal miss in a blind
    all-pairs scan, not an error).
    expected_adjacent: True when this pair was selected via a DECLARED
    adjacency (known_adjacent_chunk_pairs) -- zero candidates then means
    something upstream is wrong (bad chunking, a lost/corrupted node),
    so this raises NoBridgeCandidatesError instead of returning None.
    False in the blind O(n^2) scan (no known_adjacent_chunk_pairs given),
    where most pairs are legitimately unrelated and zero candidates is
    expected, not a bug.
    refetch_path: optional key -> path (or None on failure) callback.
    frame_poses' own path field points at wherever the image lived in the
    ORIGINAL segment-producing GPU session's ephemeral disk -- a separate
    Join call (a different ZeroGPU worker/container) has no guarantee
    that file still exists. When given, refetch_path re-downloads each
    candidate pano fresh right before testing it instead of trusting the
    stored path; a pair where either side fails to refetch is skipped
    like any other per-attempt failure, not an error. None (default)
    trusts the stored path as-is, for callers that know they're still in
    the same session that produced it (e.g. run_pathfind_and_join_gpu)."""
    a_pts, a_cols, a_edges, a_date, a_reached, a_positions, a_frame_poses = a
    b_pts, b_cols, b_edges, b_date, b_reached, b_positions, b_frame_poses = b

    pairs = []
    closest = None  # (dist, a_key, b_key) -- tracked even when nothing qualifies, for diagnostics
    for a_key, (_, _, _, a_lat, a_lon, _, _) in a_frame_poses.items():
        for b_key, (_, _, _, b_lat, b_lon, _, _) in b_frame_poses.items():
            # REAL geographic distance, not DA3-frame position -- the two
            # pieces' DA3 frames are unrelated coordinate systems
            # (different scale/origin/orientation each), comparing
            # positions across them is meaningless. lat/lon is the only
            # thing both pieces agree on.
            dist = haversine_m(a_lat, a_lon, b_lat, b_lon)
            if closest is None or dist < closest[0]:
                closest = (dist, a_key, b_key)
            if dist <= edge_max_dist_m:
                pairs.append((a_date != b_date, dist, a_key, b_key))
    if not pairs:
        closest_desc = f"closest real pair was {closest[1]} <-> {closest[2]} at {closest[0]:.1f}m" if closest else "no nodes on either side at all"
        if expected_adjacent:
            # ZeroGPU's cross-process exception marshalling can drop the
            # real message, surfacing only the exception class name to
            # the caller -- print the diagnostic here too so it's always
            # visible in the Space's own server logs regardless.
            print(f"[bridge] NoBridgeCandidatesError: {a_date} ({len(a_positions)} node(s)) <-> "
                  f"{b_date} ({len(b_positions)} node(s)): 0 candidate pair(s) within {edge_max_dist_m:.0f}m "
                  f"({closest_desc}) -- declared adjacent, so this points to a bug upstream.")
            raise NoBridgeCandidatesError(
                f"{a_date} ({len(a_positions)} node(s)) <-> {b_date} ({len(b_positions)} node(s)): "
                f"0 candidate pair(s) within {edge_max_dist_m:.0f}m ({closest_desc}) -- "
                f"these were declared adjacent, so this points to a bug upstream, not a normal miss."
            )
        print(f"[bridge] {a_date} ({len(a_positions)} node(s)) <-> {b_date} ({len(b_positions)} node(s)): "
              f"0 candidate pair(s) within {edge_max_dist_m:.0f}m -- skipped ({closest_desc})")
        return None, bridge_test_id
    print(f"[bridge] {a_date} ({len(a_positions)} node(s)) <-> {b_date} ({len(b_positions)} node(s)): "
          f"{len(pairs)} candidate pair(s) within {edge_max_dist_m:.0f}m, trying up to {BRIDGE_MAX_ATTEMPTS}")
    pairs.sort()

    best = None  # (rank_key, result, a_key, b_key)
    attempts = 0
    for _, _, a_key, b_key in pairs:
        if attempts >= BRIDGE_MAX_ATTEMPTS or time.monotonic() >= deadline:
            break
        _, _, a_path, _, _, _, _ = a_frame_poses[a_key]
        _, _, b_path, _, _, _, _ = b_frame_poses[b_key]
        if refetch_path is not None:
            fresh_a, fresh_b = refetch_path(a_key), refetch_path(b_key)
            if fresh_a is None or fresh_b is None:
                print(f"[bridge] {a_key} -> {b_key}: refetch failed, skipping")
                attempts += 1
                continue
            a_path, b_path = fresh_a, fresh_b
        result = bridge_test_edge(a_path, b_path, f"bridge_{bridge_test_id}")
        bridge_test_id += 1
        attempts += 1
        if result is None:
            continue
        ka, ta = result["keep_a"]
        kb, tb = result["keep_b"]
        keep_a_ratio = ka / ta if ta else 0.0
        keep_b_ratio = kb / tb if tb else 0.0
        sane = result["avg_dev_a"] < BRIDGE_RIDICULOUS_DEV_M and result["avg_dev_b"] < BRIDGE_RIDICULOUS_DEV_M
        passed = keep_a_ratio >= BRIDGE_KEEP_RATE and keep_b_ratio >= BRIDGE_KEEP_RATE
        # (confident?, sane?, min keep-rate, -combined avg_dev) -- ranks
        # a genuinely good match first, then prefers a sane result over
        # a flagged one, then the best of what's left by keep-
        # rate/deviation. Never disqualifies outright -- there's always
        # a best available, and it's always used.
        rank_key = (passed and sane, sane, min(keep_a_ratio, keep_b_ratio), -(result["avg_dev_a"] + result["avg_dev_b"]))
        print(f"[bridge] {a_key} -> {b_key}: keep={ka}/{ta},{kb}/{tb} avg_dev={result['avg_dev_a']:.2f}m,{result['avg_dev_b']:.2f}m "
              f"{'OK' if passed and sane else ('weak' if sane else 'poor consensus')}")
        if best is None or rank_key > best[0]:
            best = (rank_key, result, a_key, b_key)
        if passed and sane:
            break

    if best is None:
        return None, bridge_test_id

    _, result, a_key, b_key = best
    a_center, a_rot, _, _, _, _, _ = a_frame_poses[a_key]
    local_R, local_t = rigid_align([result["pose_a"]], [(a_center, a_rot)])
    bridge_pts_in_a = result["pts"] @ local_R.T + local_t
    b_key_center_in_a = local_R @ result["pose_b"][0] + local_t
    b_key_rot_in_a = result["pose_b"][1] @ local_R.T

    b_own_center, b_own_rot, _, _, _, _, _ = b_frame_poses[b_key]
    b_to_a_R, b_to_a_t = rigid_align([(b_own_center, b_own_rot)], [(b_key_center_in_a, b_key_rot_in_a)])

    merged_pts = np.concatenate([a_pts, bridge_pts_in_a, b_pts @ b_to_a_R.T + b_to_a_t], axis=0)
    merged_cols = np.concatenate([a_cols, result["cols"], b_cols], axis=0)
    merged_edges = a_edges + [(a_key, b_key)] + b_edges
    merged_positions = {**a_positions, **{k: b_to_a_R @ p + b_to_a_t for k, p in b_positions.items()}}
    merged_frame_poses = {**a_frame_poses,
                           **{k: (b_to_a_R @ p + b_to_a_t, r @ b_to_a_R.T, path, lat, lon, n_kept, n_total)
                              for k, (p, r, path, lat, lon, n_kept, n_total) in b_frame_poses.items()}}
    print(f"[bridge] {a_date}+{b_date}: merged via {a_key} -> {b_key} (keep={result['keep_a']},{result['keep_b']})")
    merged = (merged_pts, merged_cols, merged_edges, a_date, a_reached, merged_positions, merged_frame_poses)
    return merged, bridge_test_id


def bridge_pieces(segments, bridge_test_edge, edge_max_dist_m=BRIDGE_MAX_DIST_M, deadline=None,
                   chunk_ids=None, known_adjacent_chunk_pairs=None, refetch_path=None, return_ids=False):
    """Try to merge geographically/structurally adjacent segments via real
    DA3-verified transforms. Greedily merges pairs until nothing more
    merges or the deadline hits. Returns a new list of (possibly merged)
    segments, same 7-tuple shape as `segments` (see
    run_pathfind_reconstruction's return docs). Multiple pieces coming
    back is expected whenever parts of the input are genuinely not meant
    to connect (e.g. two unrelated regions with no declared adjacency) --
    not an error; see NoBridgeCandidatesError for the actual error case
    (a declared-adjacent pair with zero real candidates).

    chunk_ids: optional, same length/order as segments -- an identifying
    label per segment (e.g. which chunk of a large-scale corridor it
    came from). Each entry is normally a single id, but MAY itself be an
    iterable of ids -- lets a caller feed back in an already-merged piece
    from a PREVIOUS bridge_pieces call (see return_ids) as one input
    segment carrying its own whole prior chunk-id set, so a later call
    only needs to test that piece against genuinely NEW segments, never
    re-verifying pairs it already merged. known_adjacent_chunk_pairs:
    optional [(id_a, id_b), ...] -- when given (needs chunk_ids too),
    ONLY segment pairs whose chunk id(s) appear together in this list are
    ever attempted, skipping the blind O(n^2) all-pairs scan entirely.
    Use when the caller already knows which pieces are structurally meant
    to connect (e.g. deliberately-chunked corridor segments) -- far
    cheaper once there are many segments, and avoids wrongly bridging two
    segments that just happen to be geographically close but aren't
    actually adjacent (different floor, opposite side of a loop, etc.).
    As pieces merge, a merged piece inherits the union of its
    ingredients' chunk ids, so it stays matchable against anything
    adjacent to either original chunk.

    return_ids: when True, returns (pieces, id_sets) -- id_sets[i] is the
    sorted list of every original chunk id folded into pieces[i], for a
    caller that wants to persist "here's what's already merged" and feed
    it back into a later call as chunk_ids (see above)."""
    if bridge_test_edge is None or len(segments) < 2:
        pieces = list(segments)
        if not return_ids:
            return pieces
        if chunk_ids is not None:
            id_sets = [sorted(cid) if isinstance(cid, (set, frozenset, list, tuple)) else [cid] for cid in chunk_ids]
        else:
            id_sets = [[i] for i in range(len(pieces))]
        return pieces, id_sets
    if deadline is None:
        deadline = time.monotonic() + 200.0

    pieces = list(segments)
    if chunk_ids is not None:
        id_sets = [frozenset(cid) if isinstance(cid, (set, frozenset, list, tuple)) else frozenset({cid}) for cid in chunk_ids]
    else:
        id_sets = [frozenset({i}) for i in range(len(pieces))]
    adjacency_set = ({frozenset(pair) for pair in known_adjacent_chunk_pairs}
                      if known_adjacent_chunk_pairs is not None else None)

    def candidate_pairs():
        for i in range(len(pieces)):
            for j in range(len(pieces)):
                if i == j:
                    continue
                if adjacency_set is not None:
                    if not any(frozenset((x, y)) in adjacency_set for x in id_sets[i] for y in id_sets[j]):
                        continue
                yield i, j

    bridge_test_id = 0
    changed = True
    while changed and len(pieces) > 1 and time.monotonic() < deadline:
        changed = False
        for i, j in candidate_pairs():
            merged, bridge_test_id = _try_bridge(pieces[i], pieces[j], bridge_test_edge, edge_max_dist_m, deadline, bridge_test_id,
                                                  expected_adjacent=adjacency_set is not None, refetch_path=refetch_path)
            if merged is not None:
                merged_ids = id_sets[i] | id_sets[j]
                pieces = [p for k, p in enumerate(pieces) if k not in (i, j)] + [merged]
                id_sets = [s for k, s in enumerate(id_sets) if k not in (i, j)] + [merged_ids]
                changed = True
                break
    if not return_ids:
        return pieces
    return pieces, [sorted(s) for s in id_sets]


def pieces_to_output(pieces, id_sets=None):
    """Converts internal 7-tuple pieces to the (points, colors, metadata)
    shape returned to callers/saved to disk. Includes each node's own
    rotation (needed to bridge further later -- see output_to_piece) --
    only `path` is dropped, since bridging never trusts a stored path
    anyway (every real caller supplies bridge_pieces a refetch_path
    callback that re-downloads each candidate fresh by key; see
    _try_bridge). id_sets: optional, same length/order as pieces -- each
    piece's own list of original chunk ids (see bridge_pieces'
    return_ids=True), tagged onto every node in that piece as
    "chunk_ids" so output_to_piece can recover it later. This makes the
    output round-trippable: see output_to_piece, its exact inverse."""
    results = []
    for idx, (pts, cols, path_edges, date, reached, node_positions, frame_poses) in enumerate(pieces):
        chunk_ids = id_sets[idx] if id_sets is not None else None
        metadata = {k: {"lat": lat, "lon": lon, "date": date, "position": pos.tolist(),
                         "rotation": rot.tolist(), "n_views_kept": n_kept, "n_views_total": n_total,
                         **({"chunk_ids": chunk_ids} if chunk_ids is not None else {})}
                    for k, (pos, rot, path, lat, lon, n_kept, n_total) in frame_poses.items()}
        results.append((pts, cols, metadata))
    return results


def output_to_piece(ply_path, metadata):
    """Inverse of pieces_to_output, for ONE already-saved piece --
    reconstructs a 7-tuple segment good enough to keep bridging further
    (see bridge_pieces/_try_bridge). pts/cols come back from the binary
    PLY file; per-node position/rotation/lat/lon/date/view-counts come
    from metadata. `path` is set to None -- safe ONLY because this is
    meant to be fed to bridge_pieces together with a refetch_path
    callback (every real GPU caller here provides one), which re-fetches
    each candidate fresh by key and never looks at this field; never call
    this for a bridge_pieces call that lacks refetch_path.

    Returns (piece, chunk_ids) -- chunk_ids is whatever pieces_to_output
    tagged this piece's nodes with (all nodes in one piece carry the same
    list), or None for older output that predates that field."""
    import numpy as np
    with open(ply_path, "rb") as f:
        data = f.read()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii")
    n = int(next(l for l in header.splitlines() if l.startswith("element vertex")).split()[-1])
    has_color = "red" in header
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if has_color:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    verts = np.frombuffer(data[header_end:], dtype=np.dtype(fields), count=n)
    pts = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)
    cols = (np.stack([verts["red"], verts["green"], verts["blue"]], axis=1).astype(np.float64) / 255.0) if has_color else None

    node_positions, frame_poses, date, chunk_ids = {}, {}, "merged", None
    for key, m in metadata.items():
        pos, rot = np.array(m["position"]), np.array(m["rotation"])
        node_positions[key] = pos
        frame_poses[key] = (pos, rot, None, m["lat"], m["lon"], m["n_views_kept"], m["n_views_total"])
        date = m["date"]
        if "chunk_ids" in m:
            chunk_ids = m["chunk_ids"]
    piece = (pts, cols, [], date, False, node_positions, frame_poses)
    return piece, chunk_ids


def join_segments(segments, bridge_test_edge, edge_max_dist_m=BRIDGE_MAX_DIST_M,
                   max_time_budget_s: float = 200.0, chunk_ids=None, known_adjacent_chunk_pairs=None,
                   refetch_path=None):
    """Bridges segments together via real DA3 tests (see bridge_pieces),
    then returns each remaining piece as-is -- no GPS fit, no shared
    coordinate frame across pieces that never bridged. Each returned
    piece stays in whichever local DA3 frame its own bridge chain
    anchored to.

    segments: run_pathfind_reconstruction's output -- list of (pts, cols,
    path_edges, date, reached, node_positions, frame_poses). frame_poses
    already carries each node's real lat/lon (see bridge_pieces), so no
    separate node_entries/GPS lookup is needed here.

    chunk_ids/known_adjacent_chunk_pairs/refetch_path: passed straight
    through to bridge_pieces/_try_bridge -- see their own docstrings.

    Returns a list of (points, colors, metadata) -- one per final piece
    still separate after bridging. metadata is {key: {"lat", "lon", "date",
    "position", "n_views_kept", "n_views_total"}} for every node in that
    piece: lat/lon/date/view-counts let a later process know which real
    pano/location produced which piece without storing the images
    themselves (always re-fetchable from source by key); "position" is
    that node's own center in this piece's point cloud (pts/cols), same
    local frame -- lets you locate a node directly within the output."""
    if not segments:
        raise ValueError("No segments to join.")

    deadline = time.monotonic() + max_time_budget_s
    pieces = bridge_pieces(segments, bridge_test_edge, edge_max_dist_m, deadline,
                            chunk_ids=chunk_ids, known_adjacent_chunk_pairs=known_adjacent_chunk_pairs,
                            refetch_path=refetch_path)
    print(f"join: bridge_pieces: {len(segments)} piece(s) in, {len(pieces)} piece(s) out "
          f"({len(segments) - len(pieces)} merge(s))")
    return pieces_to_output(pieces)
