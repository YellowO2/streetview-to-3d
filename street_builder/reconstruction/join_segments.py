"""Combine independently-reconstructed pathfind segments into one merged
point cloud.

Two stages:
1. bridge_pieces (optional, needs a GPU-touching bridge_test_edge
   callback -- see services/pipeline_runner.py's join_segments_gpu):
   tries to replace independent GPS placement between geographically
   close pieces with a real DA3-verified transform. ANY real DA3
   estimate, however weak, is trusted over independent GPS placement --
   GPS is never used to reconcile two pieces against each other, only
   ever (in stage 2) to anchor the final result to real-world
   coordinates. Runs its OWN GPU session, separate from the corridor
   search's (run_pathfind_reconstruction_gpu) -- bridging only needs
   each piece's own already-confirmed nodes (no candidate pool, no
   date/corridor data), so it doesn't need to share that call at all,
   and keeping it separate means bridging behavior can be iterated on
   without re-paying for the whole (much more expensive) corridor search
   each time.
2. join_segments: whatever's still separate after bridging (or all of
   it, if bridge_test_edge wasn't given) gets independently fit against
   real GPS using its own confirmed nodes, landing every piece in one
   shared real-world-meters frame to concatenate. No GPU needed -- plain
   linear algebra (2D Kabsch/Procrustes).
"""
import time

import numpy as np

from services.geo import haversine_m, latlon_to_local_m
from street_builder.reconstruction.walk_graph import rigid_align

# Relaxed keep-rate vs. the main walk's 0.6 -- bridging only needs SOME
# real signal, and any real DA3 estimate beats independent GPS placement
# regardless of how weak. Only decides when a match is confident enough
# to stop searching early; never disqualifies a result from being used.
BRIDGE_KEEP_RATE = 0.5
# An average deviation-among-kept-views this large (not a single outlier
# -- those get filtered out already, see test_edge_da3_bridge) means the
# surviving views still don't agree with each other, a real sign the
# pair is worse than usual -- only used to break ties when ranking
# attempts, never to discard a result outright.
BRIDGE_RIDICULOUS_DEV_M = 2.0
# Real DA3 calls spent trying to bridge one pair of pieces, capped
# regardless of how many (Ax, By) node pairs qualify by distance.
BRIDGE_MAX_ATTEMPTS = 10
# Real-world distance a candidate node pair must be within to even
# attempt bridging. Temporarily bumped to 30m (from 18m, the main walk's
# own edge_max_dist_m default) to diagnose a real run where bridging
# reported "0 candidate pairs within 18m" between pieces the user
# expected to be close -- see the closest-real-pair distance now logged
# alongside that message for the actual number once this runs again.
BRIDGE_MAX_DIST_M = 30.0


def _try_bridge(a, b, bridge_test_edge, edge_max_dist_m, deadline, bridge_test_id):
    """One pair's worth of bridge search: every (Ax, By) node pair within
    edge_max_dist_m, same-date-first then closest-first, up to
    BRIDGE_MAX_ATTEMPTS real tests. ALWAYS merges using whichever attempt
    came out best, however weak -- a clearly confident match (clears
    BRIDGE_KEEP_RATE on both sides, no bad-consensus red flag) stops the
    search early; otherwise every attempt is ranked and the best one
    wins once the attempt budget/deadline is hit.
    Returns (merged_segment, next_bridge_test_id) or (None,
    next_bridge_test_id) only if there were no (Ax, By) pairs within
    range to try at all -- the one remaining case the GPS fit still has
    to cover, since there's no real signal to use in the first place."""
    a_pts, a_cols, a_edges, a_date, a_reached, a_positions, a_frame_poses = a
    b_pts, b_cols, b_edges, b_date, b_reached, b_positions, b_frame_poses = b

    pairs = []
    closest = None  # (dist, a_key, b_key) -- tracked even when nothing qualifies, for diagnostics
    for a_key, (_, _, _, a_lat, a_lon) in a_frame_poses.items():
        for b_key, (_, _, _, b_lat, b_lon) in b_frame_poses.items():
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
        return None, bridge_test_id
    print(f"[bridge] {a_date} ({len(a_positions)} node(s)) <-> {b_date} ({len(b_positions)} node(s)): "
          f"{len(pairs)} candidate pair(s) within {edge_max_dist_m:.0f}m, trying up to {BRIDGE_MAX_ATTEMPTS}")
    pairs.sort()

    best = None  # (rank_key, result, a_key, b_key)
    attempts = 0
    for _, _, a_key, b_key in pairs:
        if attempts >= BRIDGE_MAX_ATTEMPTS or time.monotonic() >= deadline:
            break
        _, _, a_path, _, _ = a_frame_poses[a_key]
        _, _, b_path, _, _ = b_frame_poses[b_key]
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
    a_center, a_rot, _, _, _ = a_frame_poses[a_key]
    local_R, local_t = rigid_align([result["pose_a"]], [(a_center, a_rot)])
    bridge_pts_in_a = result["pts"] @ local_R.T + local_t
    b_key_center_in_a = local_R @ result["pose_b"][0] + local_t
    b_key_rot_in_a = result["pose_b"][1] @ local_R.T

    b_own_center, b_own_rot, _, _, _ = b_frame_poses[b_key]
    b_to_a_R, b_to_a_t = rigid_align([(b_own_center, b_own_rot)], [(b_key_center_in_a, b_key_rot_in_a)])

    merged_pts = np.concatenate([a_pts, bridge_pts_in_a, b_pts @ b_to_a_R.T + b_to_a_t], axis=0)
    merged_cols = np.concatenate([a_cols, result["cols"], b_cols], axis=0)
    merged_edges = a_edges + [(a_key, b_key)] + b_edges
    merged_positions = {**a_positions, **{k: b_to_a_R @ p + b_to_a_t for k, p in b_positions.items()}}
    merged_frame_poses = {**a_frame_poses,
                           **{k: (b_to_a_R @ p + b_to_a_t, r @ b_to_a_R.T, path, lat, lon)
                              for k, (p, r, path, lat, lon) in b_frame_poses.items()}}
    print(f"[bridge] {a_date}+{b_date}: merged via {a_key} -> {b_key} (keep={result['keep_a']},{result['keep_b']})")
    merged = (merged_pts, merged_cols, merged_edges, a_date, a_reached, merged_positions, merged_frame_poses)
    return merged, bridge_test_id


def bridge_pieces(segments, bridge_test_edge, edge_max_dist_m=BRIDGE_MAX_DIST_M, deadline=None,
                   chunk_ids=None, known_adjacent_chunk_pairs=None):
    """Try to replace independent GPS placement between geographically
    close segments with a real DA3-verified transform (see this module's
    own docstring for the full design). Greedily merges pairs until
    nothing more merges or the deadline hits. Returns a new list of
    (possibly merged) segments, same 7-tuple shape as `segments` (see
    run_pathfind_reconstruction's return docs).

    chunk_ids: optional, same length/order as segments -- an identifying
    label per segment (e.g. which chunk of a large-scale corridor it
    came from). known_adjacent_chunk_pairs: optional [(id_a, id_b), ...]
    -- when given (needs chunk_ids too), ONLY segment pairs whose chunk
    id(s) appear together in this list are ever attempted, skipping the
    blind O(n^2) all-pairs scan entirely. Use when the caller already
    knows which pieces are structurally meant to connect (e.g.
    deliberately-chunked corridor segments) -- far cheaper once there
    are many segments, and avoids wrongly bridging two segments that
    just happen to be geographically close but aren't actually adjacent
    (different floor, opposite side of a loop, etc.). As pieces merge, a
    merged piece inherits the union of its ingredients' chunk ids, so it
    stays matchable against anything adjacent to either original chunk."""
    if bridge_test_edge is None or len(segments) < 2:
        return segments
    if deadline is None:
        deadline = time.monotonic() + 200.0

    pieces = list(segments)
    if chunk_ids is not None:
        id_sets = [frozenset({cid}) for cid in chunk_ids]
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
            merged, bridge_test_id = _try_bridge(pieces[i], pieces[j], bridge_test_edge, edge_max_dist_m, deadline, bridge_test_id)
            if merged is not None:
                merged_ids = id_sets[i] | id_sets[j]
                pieces = [p for k, p in enumerate(pieces) if k not in (i, j)] + [merged]
                id_sets = [s for k, s in enumerate(id_sets) if k not in (i, j)] + [merged_ids]
                changed = True
                break
    return pieces


def _fit_rigid_2d(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rigid transform (2x2 rotation + 2D translation) mapping
    src onto dst, given Nx2 arrays matched row-by-row (N >= 2). Standard
    Kabsch algorithm restricted to 2D -- no scale, since DA3's own metric
    scale isn't being second-guessed here, only its horizontal position and
    heading against real GPS."""
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T
    t = dst_mean - R @ src_mean
    return R, t


def join_segments(segments, node_entries, bridge_test_edge=None, edge_max_dist_m=BRIDGE_MAX_DIST_M,
                   max_time_budget_s: float = 200.0, chunk_ids=None, known_adjacent_chunk_pairs=None):
    """segments: run_pathfind_reconstruction's output -- list of (pts,
    cols, path_edges, date, reached, node_positions, frame_poses),
    node_positions being {key: np.ndarray(3,)} for that segment's own
    confirmed nodes. node_entries: prep['node_entries'] -- (key, path,
    lat, lon, date) tuples, giving real lat/lon for every node key
    referenced anywhere.

    bridge_test_edge: optional, see this module's own docstring --
    when given, segments are bridged (real DA3 connections between
    geographically close pieces) before the GPS fit below runs, so only
    whatever's genuinely disconnected still needs it. chunk_ids/
    known_adjacent_chunk_pairs: passed straight through to bridge_pieces
    -- see its own docstring.

    For each (post-bridging) segment: fit rotation (about the vertical
    axis only -- GPS has no elevation data to fit against, so the
    vertical axis is left alone rather than risk an unconstrained tilt)
    + horizontal translation from that segment's own DA3-frame node
    positions onto their real GPS positions (converted to local meters,
    one shared origin for every segment). Vertical placement uses a
    simple heuristic instead -- shifting each segment's own average node
    height to a shared baseline, since GPS can't fit that axis and
    street-level camera heights should be roughly comparable across
    segments anyway.

    Returns (points, colors, metadata): points/colors are one merged
    point cloud, all segments in the same real-world-meters frame.
    metadata is {key: {"lat", "lon", "date", "world_position": [x, y,
    z]}} for every node that contributed -- world_position is that
    node's own placement in the SAME joined frame as points/colors
    (using the same fitted transform, not its raw DA3-local center),
    letting a later process know which real pano/location produced
    which region of the cloud without storing the images themselves
    (always re-fetchable from source; key/lat/lon/date is enough) --
    e.g. re-running with cleaned-up images later, or roughly re-
    splitting an already-finished reconstruction by node position.
    """
    if not segments:
        raise ValueError("No segments to join.")

    if bridge_test_edge is not None:
        n_before = len(segments)
        deadline = time.monotonic() + max_time_budget_s
        segments = bridge_pieces(segments, bridge_test_edge, edge_max_dist_m, deadline,
                                  chunk_ids=chunk_ids, known_adjacent_chunk_pairs=known_adjacent_chunk_pairs)
        print(f"join: bridge_pieces: {n_before} piece(s) in, {len(segments)} piece(s) out "
              f"({n_before - len(segments)} merge(s))")

    by_key = {e[0]: e for e in node_entries}

    # Shared origin for every segment's lat/lon -> local-meters conversion --
    # arbitrary choice (first segment's first confirmed node), just needs
    # to be the SAME point for all of them so they land in one frame.
    first_key = next(iter(segments[0][5]))
    _, _, origin_lat, origin_lon, _ = by_key[first_key]

    all_pts, all_cols = [], []
    metadata = {}
    for seg_i, (pts, cols, path_edges, date, reached, node_positions, frame_poses) in enumerate(segments):
        keys = list(node_positions.keys())
        if len(keys) < 2:
            print(f"join: segment {seg_i} ({date}) has <2 confirmed nodes -- skipping, can't fit a rotation")
            continue

        da3_xz = np.array([[node_positions[k][0], node_positions[k][2]] for k in keys])
        real_en = []
        for k in keys:
            _, _, lat, lon, _ = by_key[k]
            e, n = latlon_to_local_m(lat, lon, origin_lat, origin_lon)
            real_en.append([e, n])
        real_en = np.array(real_en)

        R, t = _fit_rigid_2d(da3_xz, real_en)
        avg_y = float(np.mean([node_positions[k][1] for k in keys]))

        xz = pts[:, [0, 2]] @ R.T + t
        y = pts[:, 1] - avg_y
        transformed = np.column_stack([xz[:, 0], y, xz[:, 1]])

        heading = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        print(f"join: segment {seg_i} ({date}) fit against {len(keys)} node(s), rotation={heading:.1f}deg")

        all_pts.append(transformed)
        all_cols.append(cols)

        node_xz = da3_xz @ R.T + t
        node_y = np.array([node_positions[k][1] for k in keys]) - avg_y
        for idx, k in enumerate(keys):
            _, _, lat, lon, node_date = by_key[k]
            metadata[k] = {
                "lat": lat, "lon": lon, "date": node_date,
                "world_position": [float(node_xz[idx, 0]), float(node_y[idx]), float(node_xz[idx, 1])],
            }

    if not all_pts:
        raise ValueError("No segment had enough confirmed nodes to fit.")

    return np.concatenate(all_pts, axis=0), np.concatenate(all_cols, axis=0), metadata
