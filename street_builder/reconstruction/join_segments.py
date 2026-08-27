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
    """Raised when a DECLARED-adjacent chunk-id pair never had a single
    real node pair within edge_max_dist_m, across EVERY piece-level
    combination carrying those two ids -- not just the first one tried.
    A chunk can legitimately still be several separate pieces (self-
    bridge/an earlier call didn't fully connect it), so one specific
    piece-pair coming up empty is normal; only the declared id pair as a
    WHOLE coming up empty everywhere points to something wrong upstream
    (bad chunking, a node that failed to download, a corrupted
    position)."""


def _try_bridge(a, b, bridge_test_edge, edge_max_dist_m, deadline, bridge_test_id, refetch_path=None):
    """One pair's worth of bridge search: every (Ax, By) node pair within
    edge_max_dist_m, same-date-first then closest-first, up to
    BRIDGE_MAX_ATTEMPTS real tests. ALWAYS merges using whichever attempt
    came out best, however weak -- a clearly confident match (clears
    BRIDGE_KEEP_RATE on both sides, no bad-consensus red flag) stops the
    search early; otherwise every attempt is ranked and the best one
    wins once the attempt budget/deadline is hit.
    Returns (merged_segment, next_bridge_test_id, had_candidates).
    merged_segment is None if real candidates existed but every DA3
    attempt on them came back unusable (a genuine per-attempt failure),
    or if there were zero candidates at all. had_candidates distinguishes
    those two None cases for the caller (bridge_pieces) -- whether THIS
    declared pair ever needs raising NoBridgeCandidatesError is decided
    there, only once EVERY piece-level pair sharing those two chunk ids
    has been tried, not on this one pair alone.
    refetch_path: optional (key, lat, lon) -> path (or None on failure)
    callback -- lat/lon passed through since Apple's fetch-by-id needs a
    coarse location to know which coverage tile to search (see
    map_selection.candidates.apple_tile_panos), unlike Google's pure
    fetch-by-id.
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
        print(f"[bridge] {a_date} ({len(a_positions)} node(s)) <-> {b_date} ({len(b_positions)} node(s)): "
              f"0 candidate pair(s) within {edge_max_dist_m:.0f}m -- skipped ({closest_desc})")
        return None, bridge_test_id, False
    print(f"[bridge] {a_date} ({len(a_positions)} node(s)) <-> {b_date} ({len(b_positions)} node(s)): "
          f"{len(pairs)} candidate pair(s) within {edge_max_dist_m:.0f}m, trying up to {BRIDGE_MAX_ATTEMPTS}")
    pairs.sort()

    best = None  # (rank_key, result, a_key, b_key)
    attempts = 0
    for _, _, a_key, b_key in pairs:
        if attempts >= BRIDGE_MAX_ATTEMPTS or time.monotonic() >= deadline:
            break
        _, _, a_path, a_lat, a_lon, _, _ = a_frame_poses[a_key]
        _, _, b_path, b_lat, b_lon, _, _ = b_frame_poses[b_key]
        if refetch_path is not None:
            fresh_a, fresh_b = refetch_path(a_key, a_lat, a_lon), refetch_path(b_key, b_lat, b_lon)
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
        return None, bridge_test_id, True

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
    return merged, bridge_test_id, True


def bridge_pieces(segments, bridge_test_edge, edge_max_dist_m=BRIDGE_MAX_DIST_M, deadline=None,
                   chunk_ids=None, known_adjacent_chunk_pairs=None, refetch_path=None, return_ids=False,
                   raise_on_unsatisfied=True):
    """Try to merge geographically/structurally adjacent segments via real
    DA3-verified transforms. Greedily merges pairs until nothing more
    merges or the deadline hits. Returns a new list of (possibly merged)
    segments, same 7-tuple shape as `segments` (see
    run_pathfind_reconstruction's return docs). Multiple pieces coming
    back is expected whenever parts of the input are genuinely not meant
    to connect (e.g. two unrelated regions with no declared adjacency) --
    not an error; see NoBridgeCandidatesError for the actual error case
    (a declared-adjacent pair with zero real candidates, and
    raise_on_unsatisfied=True -- see that param).

    raise_on_unsatisfied: whether a declared-adjacent pair that never
    once had a real candidate raises NoBridgeCandidatesError (True,
    default) or is just left as separate pieces, same as an
    undeclared/blind miss (False). True is right when the caller's own
    adjacency really does guarantee real closeness (e.g. cross-chunk
    bridging, where chunking itself is built around a tight per-dot
    catchment -- see global_dates.split_cover_into_chunks -- so a
    declared pair with zero candidates does point to something wrong
    upstream). False is right for a same-call self-bridge pass over an
    arbitrary corridor (e.g. a hand-picked selection, or ANY chunk that
    isn't specifically engineered for tight spacing): two real graph-
    adjacent dots can legitimately still be a genuine 30m+ apart in
    practice (real gaps in real coverage) -- that's not a bug, it's just
    geography, and should leave those two as separate pieces like any
    other normal miss, not hard-fail the whole call.

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

    # Declared pairs (from known_adjacent_chunk_pairs) that have had at
    # least one real candidate node pair in SOME piece-level attempt --
    # tracked across the whole run, not per-attempt, since a chunk can
    # legitimately still be several separate pieces (self-bridge/an
    # earlier call didn't fully connect it) and only one of them needs
    # to actually have candidates for the declared pair to be "satisfied".
    satisfied_declared_pairs = set()

    bridge_test_id = 0
    changed = True
    while changed and len(pieces) > 1 and time.monotonic() < deadline:
        changed = False
        for i, j in candidate_pairs():
            merged, bridge_test_id, had_candidates = _try_bridge(pieces[i], pieces[j], bridge_test_edge, edge_max_dist_m, deadline, bridge_test_id,
                                                                   refetch_path=refetch_path)
            if had_candidates and adjacency_set is not None:
                satisfied_declared_pairs |= {frozenset((x, y)) for x in id_sets[i] for y in id_sets[j]
                                              if frozenset((x, y)) in adjacency_set}
            if merged is not None:
                merged_ids = id_sets[i] | id_sets[j]
                pieces = [p for k, p in enumerate(pieces) if k not in (i, j)] + [merged]
                id_sets = [s for k, s in enumerate(id_sets) if k not in (i, j)] + [merged_ids]
                changed = True
                break

    if adjacency_set is not None:
        unsatisfied = adjacency_set - satisfied_declared_pairs
        if unsatisfied and not raise_on_unsatisfied:
            desc = ", ".join(f"{sorted(pair)[0]} <-> {sorted(pair)[1]}" for pair in sorted(unsatisfied, key=sorted))
            print(f"[bridge] {len(unsatisfied)} declared-adjacent pair(s) never had a single real candidate within "
                  f"{edge_max_dist_m:.0f}m, across every piece-level attempt: {desc} -- left as separate pieces "
                  f"(raise_on_unsatisfied=False, a normal real-world gap, not treated as a bug here).")
        elif unsatisfied:
            desc = ", ".join(f"{sorted(pair)[0]} <-> {sorted(pair)[1]}" for pair in sorted(unsatisfied, key=sorted))
            # ZeroGPU's cross-process exception marshalling can drop the
            # real message, surfacing only the exception class name to
            # the caller -- print the diagnostic here too so it's always
            # visible in the Space's own server logs regardless.
            print(f"[bridge] NoBridgeCandidatesError: {len(unsatisfied)} declared-adjacent pair(s) never had a single "
                  f"real candidate within {edge_max_dist_m:.0f}m, across every piece-level attempt: {desc} -- "
                  f"declared adjacent, so this points to a bug upstream.")
            raise NoBridgeCandidatesError(
                f"{len(unsatisfied)} declared-adjacent pair(s) never had a single real candidate within "
                f"{edge_max_dist_m:.0f}m, across every piece-level attempt: {desc} -- these were declared "
                f"adjacent, so this points to a bug upstream, not a normal miss."
            )

    if not return_ids:
        return pieces
    return pieces, [sorted(s) for s in id_sets]


# ---- Metadata-only merging (for large-scale, many-chunk trees) --------
#
# bridge_pieces/_try_bridge above concatenate real pts/cols on every
# merge -- fine for a handful of segments, but a real chunk's own raw
# output routinely runs 50-100+MB, and a large-scale reconstruction
# (many chunks, merged as a tree -- see tests/staged_corridor_test.py)
# would otherwise carry an ever-growing point cloud through every
# intermediate merge step, right up to the root. The functions below
# do the exact same real DA3-verified bridging (same candidate search,
# same rigid_align, same NoBridgeCandidatesError semantics) but track
# WHERE each original leaf piece's own points would need to go
# (chunk_id, piece_index, and the rigid transform into the current
# group's shared frame) instead of actually moving any point-cloud
# bytes. The real points only ever get touched once, at the very end,
# by assemble_metadata_piece -- reading each leaf's own already-
# durably-saved raw .ply (see tab.py's cli_run_chunk) and applying its
# final composed transform.


def segments_to_meta_pieces(chunk_id, segments):
    """Converts one chunk's own real segments (run_pathfind_reconstruction's
    output, already self-bridged -- see pipeline_runner.py) into
    metadata-only pieces ready for tree merging (see bridge_metadata).
    Each piece starts as its own single leaf reference with an identity
    transform -- its own raw frame IS the group's frame, until a real
    merge composes something else in front of it. piece_index is this
    segment's own position in `segments`, matching exactly how
    pieces_to_output/_save_joined_pieces numbers a chunk's own raw
    output files (pathfind_joined_0.ply, _1.ply, ...) -- what
    assemble_metadata_piece needs to find the right file later.

    Returns a list of meta pieces: (leaf_refs, path_edges, date,
    reached, frame_poses) -- leaf_refs: [(chunk_id, piece_index, R, t),
    ...], normally just one entry until real merges start accumulating
    more."""
    meta_pieces = []
    for i, (pts, cols, path_edges, date, reached, node_positions, frame_poses) in enumerate(segments):
        leaf_refs = [(chunk_id, i, np.eye(3), np.zeros(3))]
        meta_pieces.append((leaf_refs, path_edges, date, reached, frame_poses))
    return meta_pieces


def _try_bridge_meta(a, b, bridge_test_edge, edge_max_dist_m, deadline, bridge_test_id, refetch_path=None):
    """Metadata-only sibling of _try_bridge -- see its own docstring for
    the shared candidate search/ranking/refetch logic, identical here.
    The only real difference: _try_bridge concatenates a_pts/bridge_pts/
    b_pts into a new array; this composes the same b_to_a_R/b_to_a_t
    transform into each of b's own leaf_refs instead (point p in a
    leaf's own raw frame -> R_leaf @ p + t_leaf gives p in b's frame ->
    b_to_a_R @ (that) + b_to_a_t gives p in a's frame -- so composing is
    just b_to_a_R @ R_leaf, b_to_a_R @ t_leaf + b_to_a_t), leaving the
    actual point-cloud bytes untouched anywhere.

    Returns (merged_meta_piece_or_None, next_bridge_test_id,
    had_candidates) -- same three-part contract as _try_bridge."""
    leaf_refs_a, a_edges, a_date, a_reached, a_frame_poses = a
    leaf_refs_b, b_edges, b_date, b_reached, b_frame_poses = b

    pairs = []
    closest = None
    for a_key, (_, _, _, a_lat, a_lon, _, _) in a_frame_poses.items():
        for b_key, (_, _, _, b_lat, b_lon, _, _) in b_frame_poses.items():
            dist = haversine_m(a_lat, a_lon, b_lat, b_lon)
            if closest is None or dist < closest[0]:
                closest = (dist, a_key, b_key)
            if dist <= edge_max_dist_m:
                pairs.append((a_date != b_date, dist, a_key, b_key))
    if not pairs:
        closest_desc = f"closest real pair was {closest[1]} <-> {closest[2]} at {closest[0]:.1f}m" if closest else "no nodes on either side at all"
        print(f"[bridge-meta] {a_date} ({len(a_frame_poses)} node(s)) <-> {b_date} ({len(b_frame_poses)} node(s)): "
              f"0 candidate pair(s) within {edge_max_dist_m:.0f}m -- skipped ({closest_desc})")
        return None, bridge_test_id, False
    print(f"[bridge-meta] {a_date} ({len(a_frame_poses)} node(s)) <-> {b_date} ({len(b_frame_poses)} node(s)): "
          f"{len(pairs)} candidate pair(s) within {edge_max_dist_m:.0f}m, trying up to {BRIDGE_MAX_ATTEMPTS}")
    pairs.sort()

    best = None
    attempts = 0
    for _, _, a_key, b_key in pairs:
        if attempts >= BRIDGE_MAX_ATTEMPTS or time.monotonic() >= deadline:
            break
        _, _, a_path, a_lat, a_lon, _, _ = a_frame_poses[a_key]
        _, _, b_path, b_lat, b_lon, _, _ = b_frame_poses[b_key]
        if refetch_path is not None:
            fresh_a, fresh_b = refetch_path(a_key, a_lat, a_lon), refetch_path(b_key, b_lat, b_lon)
            if fresh_a is None or fresh_b is None:
                print(f"[bridge-meta] {a_key} -> {b_key}: refetch failed, skipping")
                attempts += 1
                continue
            a_path, b_path = fresh_a, fresh_b
        result = bridge_test_edge(a_path, b_path, f"bridgemeta_{bridge_test_id}")
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
        rank_key = (passed and sane, sane, min(keep_a_ratio, keep_b_ratio), -(result["avg_dev_a"] + result["avg_dev_b"]))
        print(f"[bridge-meta] {a_key} -> {b_key}: keep={ka}/{ta},{kb}/{tb} avg_dev={result['avg_dev_a']:.2f}m,{result['avg_dev_b']:.2f}m "
              f"{'OK' if passed and sane else ('weak' if sane else 'poor consensus')}")
        if best is None or rank_key > best[0]:
            best = (rank_key, result, a_key, b_key)
        if passed and sane:
            break

    if best is None:
        return None, bridge_test_id, True

    _, result, a_key, b_key = best
    a_center, a_rot, _, _, _, _, _ = a_frame_poses[a_key]
    local_R, local_t = rigid_align([result["pose_a"]], [(a_center, a_rot)])
    b_key_center_in_a = local_R @ result["pose_b"][0] + local_t
    b_key_rot_in_a = result["pose_b"][1] @ local_R.T

    b_own_center, b_own_rot, _, _, _, _, _ = b_frame_poses[b_key]
    b_to_a_R, b_to_a_t = rigid_align([(b_own_center, b_own_rot)], [(b_key_center_in_a, b_key_rot_in_a)])

    merged_leaf_refs = list(leaf_refs_a) + [
        (chunk_id, piece_idx, b_to_a_R @ R_leaf, b_to_a_R @ t_leaf + b_to_a_t)
        for chunk_id, piece_idx, R_leaf, t_leaf in leaf_refs_b
    ]
    merged_frame_poses = {**a_frame_poses,
                           **{k: (b_to_a_R @ p + b_to_a_t, r @ b_to_a_R.T, path, lat, lon, n_kept, n_total)
                              for k, (p, r, path, lat, lon, n_kept, n_total) in b_frame_poses.items()}}
    print(f"[bridge-meta] {a_date}+{b_date}: merged via {a_key} -> {b_key} (keep={result['keep_a']},{result['keep_b']})")
    merged = (merged_leaf_refs, a_edges + [(a_key, b_key)] + b_edges, a_date, a_reached, merged_frame_poses)
    return merged, bridge_test_id, True


def bridge_metadata(meta_pieces, bridge_test_edge, edge_max_dist_m=BRIDGE_MAX_DIST_M, deadline=None,
                     chunk_ids=None, known_adjacent_chunk_pairs=None, refetch_path=None, return_ids=False,
                     raise_on_unsatisfied=True):
    """Metadata-only sibling of bridge_pieces -- see its own docstring
    for the shared merge-order/declared-adjacency-restriction/
    NoBridgeCandidatesError logic, identical here except it calls
    _try_bridge_meta instead of _try_bridge and never touches point-
    cloud-sized data. meta_pieces: from segments_to_meta_pieces, or a
    previous bridge_metadata call's own output fed back in."""
    if bridge_test_edge is None or len(meta_pieces) < 2:
        pieces = list(meta_pieces)
        if not return_ids:
            return pieces
        if chunk_ids is not None:
            id_sets = [sorted(cid) if isinstance(cid, (set, frozenset, list, tuple)) else [cid] for cid in chunk_ids]
        else:
            id_sets = [[i] for i in range(len(pieces))]
        return pieces, id_sets
    if deadline is None:
        deadline = time.monotonic() + 200.0

    pieces = list(meta_pieces)
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

    satisfied_declared_pairs = set()
    bridge_test_id = 0
    changed = True
    while changed and len(pieces) > 1 and time.monotonic() < deadline:
        changed = False
        for i, j in candidate_pairs():
            merged, bridge_test_id, had_candidates = _try_bridge_meta(pieces[i], pieces[j], bridge_test_edge, edge_max_dist_m, deadline, bridge_test_id,
                                                                        refetch_path=refetch_path)
            if had_candidates and adjacency_set is not None:
                satisfied_declared_pairs |= {frozenset((x, y)) for x in id_sets[i] for y in id_sets[j]
                                              if frozenset((x, y)) in adjacency_set}
            if merged is not None:
                merged_ids = id_sets[i] | id_sets[j]
                pieces = [p for k, p in enumerate(pieces) if k not in (i, j)] + [merged]
                id_sets = [s for k, s in enumerate(id_sets) if k not in (i, j)] + [merged_ids]
                changed = True
                break

    if adjacency_set is not None:
        unsatisfied = adjacency_set - satisfied_declared_pairs
        if unsatisfied and not raise_on_unsatisfied:
            desc = ", ".join(f"{sorted(pair)[0]} <-> {sorted(pair)[1]}" for pair in sorted(unsatisfied, key=sorted))
            print(f"[bridge-meta] {len(unsatisfied)} declared-adjacent pair(s) never had a single real candidate within "
                  f"{edge_max_dist_m:.0f}m, across every piece-level attempt: {desc} -- left as separate pieces "
                  f"(raise_on_unsatisfied=False, a normal real-world gap, not treated as a bug here).")
        elif unsatisfied:
            desc = ", ".join(f"{sorted(pair)[0]} <-> {sorted(pair)[1]}" for pair in sorted(unsatisfied, key=sorted))
            print(f"[bridge-meta] NoBridgeCandidatesError: {len(unsatisfied)} declared-adjacent pair(s) never had a single "
                  f"real candidate within {edge_max_dist_m:.0f}m, across every piece-level attempt: {desc} -- "
                  f"declared adjacent, so this points to a bug upstream.")
            raise NoBridgeCandidatesError(
                f"{len(unsatisfied)} declared-adjacent pair(s) never had a single real candidate within "
                f"{edge_max_dist_m:.0f}m, across every piece-level attempt: {desc} -- these were declared "
                f"adjacent, so this points to a bug upstream, not a normal miss."
            )

    if not return_ids:
        return pieces
    return pieces, [sorted(s) for s in id_sets]


def assemble_metadata_piece(meta_piece, fetch_leaf_ply):
    """Resolves ONE metadata-only piece's leaf_refs into an actual point
    cloud -- the ONLY place in the whole tree-merge flow that ever
    touches point-cloud-sized data, and it only ever needs to run once,
    on the final root (or whenever a human wants to see an actual
    viewable result from an in-progress tree).

    fetch_leaf_ply(chunk_id, piece_index) -> local .ply path, already
    resolved to a plain (decompressed, dequantized) file -- this
    module has no idea how or where leaves are actually stored (see
    tab.py), it only knows how to read one once it has a path (see
    _read_ply_points).

    Returns (pts, cols, path_edges, date, reached, node_positions,
    frame_poses) -- the SAME real-segment 7-tuple shape bridge_pieces/
    pieces_to_output already expect, so nothing downstream needs to
    know metadata-only merging happened at all."""
    leaf_refs, path_edges, date, reached, frame_poses = meta_piece
    all_pts, all_cols = [], []
    any_cols = False
    for chunk_id, piece_idx, R, t in leaf_refs:
        pts, cols = _read_ply_points(fetch_leaf_ply(chunk_id, piece_idx))
        all_pts.append(pts @ R.T + t)
        all_cols.append(cols)
        any_cols = any_cols or cols is not None
    merged_pts = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3))
    merged_cols = (np.concatenate([c if c is not None else np.zeros((len(p), 3)) for p, c in zip(all_pts, all_cols)], axis=0)
                   if any_cols else None)
    node_positions = {k: pos for k, (pos, rot, path, lat, lon, n_kept, n_total) in frame_poses.items()}
    return merged_pts, merged_cols, path_edges, date, reached, node_positions, frame_poses


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


def _read_ply_points(ply_path):
    """Reads a plain (uncompressed, unquantized -- see tab.py's own
    _dequantize_ply for the CLI storage format that resolves to this
    before it ever reaches here) binary PLY back into (pts, cols) --
    the shared core of output_to_piece and assemble_metadata_piece, so
    there's exactly one place that understands this file format."""
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
    return pts, cols


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
    pts, cols = _read_ply_points(ply_path)

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
                   refetch_path=None, raise_on_unsatisfied=True):
    """Bridges segments together via real DA3 tests (see bridge_pieces),
    then returns each remaining piece as-is -- no GPS fit, no shared
    coordinate frame across pieces that never bridged. Each returned
    piece stays in whichever local DA3 frame its own bridge chain
    anchored to.

    segments: run_pathfind_reconstruction's output -- list of (pts, cols,
    path_edges, date, reached, node_positions, frame_poses). frame_poses
    already carries each node's real lat/lon (see bridge_pieces), so no
    separate node_entries/GPS lookup is needed here.

    chunk_ids/known_adjacent_chunk_pairs/refetch_path/raise_on_unsatisfied:
    passed straight through to bridge_pieces/_try_bridge -- see their
    own docstrings.

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
                            refetch_path=refetch_path, raise_on_unsatisfied=raise_on_unsatisfied)
    print(f"join: bridge_pieces: {len(segments)} piece(s) in, {len(pieces)} piece(s) out "
          f"({len(segments) - len(pieces)} merge(s))")
    return pieces_to_output(pieces)
