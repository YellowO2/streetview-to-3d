"""Combine independently-reconstructed pathfind segments using real DA3
tests -- no GPS placement here; that happens later, in postprocess/.

Every pair of segments is tried via _try_bridge: real lat/lon (carried
per node in frame_poses) picks which node pairs are worth a real DA3 test,
then the merge uses the DA3-derived relative transform between the two
segments' own local frames. A pair with no nodes in range is skipped.
"""
import time

from streetview_to_3d.services.da3_ops import MIN_KEEP_RATE
from streetview_to_3d.services.geo import haversine_m
from streetview_to_3d.reconstruct.walk_graph import rigid_align

# Relaxed keep-rate vs. the main walk's 0.6 -- bridging only needs SOME
# real signal. Decides when a match is confident enough to stop
# searching early, and is part of ranking attempts against each other;
# NOT the reject floor itself -- see BRIDGE_MIN_KEEP_RATE for that.
BRIDGE_KEEP_RATE = 0.5
# An average deviation-among-kept-views this large (not a single outlier
# -- those get filtered out already, see services.da3_ops.bridge_test_edge) means the
# surviving views still don't agree with each other -- a real sign the
# pair is worse than usual. Part of the reject floor: even the best-
# ranked attempt across a pair of pieces must clear this (both sides),
# or the merge is rejected and the pieces stay separate.
BRIDGE_RIDICULOUS_DEV_M = 2.0
# Second half of the reject floor, alongside BRIDGE_RIDICULOUS_DEV_M --
# deliberately much more lenient than BRIDGE_KEEP_RATE's 0.5 "confident"
# bar: this only exists to catch a genuinely bad match (most of a
# pano's views failing DA3's own consensus filter), not to demand a
# strong one. Even the best-ranked attempt must clear BOTH this and the
# deviation floor, or the merge is rejected.
BRIDGE_MIN_KEEP_RATE = MIN_KEEP_RATE
# Real DA3 calls spent trying to bridge one pair of pieces, capped
# regardless of how many (Ax, By) node pairs qualify by distance.
BRIDGE_MAX_ATTEMPTS = 10
# Real-world distance a candidate node pair must be within to even
# attempt bridging.
BRIDGE_MAX_DIST_M = 30.0


def _try_bridge(a, b, bridge_test_edge, edge_max_dist_m, deadline, bridge_test_id):
    """One pair's worth of bridge search: every (Ax, By) node pair within
    edge_max_dist_m, same-date-first then closest-first, up to
    BRIDGE_MAX_ATTEMPTS real tests. Merges using whichever attempt came
    out best -- a clearly confident match (clears BRIDGE_KEEP_RATE on
    both sides, no bad-consensus red flag) stops the search early;
    otherwise every attempt is ranked and the best one wins once the
    attempt budget/deadline is hit -- UNLESS even the best available
    attempt fails the reject floor (avg_dev_a/avg_dev_b both under
    BRIDGE_RIDICULOUS_DEV_M, AND both sides' keep-rate at or above
    BRIDGE_MIN_KEEP_RATE), in which case nothing merges: a genuinely
    bad match doesn't get forced through just for being the least-bad of
    a weak field, matching the Run step's own walk (which already never
    accepts a genuinely failed edge test either).
    Returns (merged_segment, next_bridge_test_id). merged_segment is None
    if there were no candidates in range, or every DA3 attempt on them came
    back unusable or failed the reject floor."""
    a_clouds, a_edges, a_date, a_reached, a_positions, a_frame_poses = a
    b_clouds, b_edges, b_date, b_reached, b_positions, b_frame_poses = b

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
        return None, bridge_test_id
    print(f"[bridge] {a_date} ({len(a_positions)} node(s)) <-> {b_date} ({len(b_positions)} node(s)): "
          f"{len(pairs)} candidate pair(s) within {edge_max_dist_m:.0f}m, trying up to {BRIDGE_MAX_ATTEMPTS}")
    pairs.sort()

    best = None  # (rank_key, result, a_key, b_key)
    attempts = 0
    for _, _, a_key, b_key in pairs:
        if attempts >= BRIDGE_MAX_ATTEMPTS or time.monotonic() >= deadline:
            break
        a_path = a_frame_poses[a_key][2]
        b_path = b_frame_poses[b_key][2]
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
        # rate/deviation. The best-ranked attempt still has to clear the
        # reject floor on its own (checked below) -- being the least-bad
        # of a weak field isn't good enough by itself.
        rank_key = (passed and sane, sane, min(keep_a_ratio, keep_b_ratio), -(result["avg_dev_a"] + result["avg_dev_b"]))
        print(f"[bridge] {a_key} -> {b_key}: keep={ka}/{ta},{kb}/{tb} avg_dev={result['avg_dev_a']:.2f}m,{result['avg_dev_b']:.2f}m "
              f"{'OK' if passed and sane else ('weak' if sane else 'poor consensus')}")
        if best is None or rank_key > best[0]:
            best = (rank_key, result, a_key, b_key)
        if passed and sane:
            break

    if best is not None:
        _, best_sane, best_min_keep, _ = best[0]
    if best is None or not best_sane or best_min_keep < BRIDGE_MIN_KEEP_RATE:
        if best is not None:
            print(f"[bridge] {a_date}+{b_date}: best available still failed the reject floor "
                  f"(avg_dev >= {BRIDGE_RIDICULOUS_DEV_M:.1f}m or keep-rate < {BRIDGE_MIN_KEEP_RATE:.2f}) -- leaving separate")
        return None, bridge_test_id

    _, result, a_key, b_key = best
    a_center, a_rot, _, _, _, _, _ = a_frame_poses[a_key]
    local_R, local_t = rigid_align([result["pose_a"]], [(a_center, a_rot)])
    b_key_center_in_a = local_R @ result["pose_b"][0] + local_t
    b_key_rot_in_a = result["pose_b"][1] @ local_R.T

    b_own_center, b_own_rot, _, _, _, _, _ = b_frame_poses[b_key]
    b_to_a_R, b_to_a_t = rigid_align([(b_own_center, b_own_rot)], [(b_key_center_in_a, b_key_rot_in_a)])

    # The bridge test's own cloud is a second copy of these two panoramas,
    # which both pieces already carry. Only its transform is new.
    merged_clouds = {**a_clouds,
                     **{k: (pts @ b_to_a_R.T + b_to_a_t, cols)
                        for k, (pts, cols) in b_clouds.items()}}
    # carries its own keep counts like any other edge -- a bridge is a real
    # DA3 test between two panoramas, and how well they agreed is exactly
    # what decides whether the link should later be trusted
    merged_edges = a_edges + [(a_key, b_key, list(result["keep_a"]),
                               list(result["keep_b"]))] + b_edges
    merged_positions = {**a_positions, **{k: b_to_a_R @ p + b_to_a_t for k, p in b_positions.items()}}
    merged_frame_poses = {**a_frame_poses,
                           **{k: (b_to_a_R @ p + b_to_a_t, r @ b_to_a_R.T, path, lat, lon, n_kept, n_total)
                              for k, (p, r, path, lat, lon, n_kept, n_total) in b_frame_poses.items()}}
    print(f"[bridge] {a_date}+{b_date}: merged via {a_key} -> {b_key} (keep={result['keep_a']},{result['keep_b']})")
    merged = (merged_clouds, merged_edges, a_date, a_reached, merged_positions, merged_frame_poses)
    return merged, bridge_test_id


def bridge_pieces(segments, bridge_test_edge, edge_max_dist_m=BRIDGE_MAX_DIST_M,
                   deadline=None):
    """Merge segments that a real DA3 test says belong together.

    Greedily tries every pair until nothing more merges or the deadline
    hits, and returns the segments that remain. More than one coming back
    is expected, not an error: two parts of a corridor can be genuinely
    unconnectable because the imagery between them has a real gap.
    """
    if bridge_test_edge is None or len(segments) < 2:
        return list(segments)
    if deadline is None:
        deadline = time.monotonic() + 200.0

    pieces = list(segments)
    bridge_test_id = 0
    changed = True
    while changed and len(pieces) > 1 and time.monotonic() < deadline:
        changed = False
        for i in range(len(pieces)):
            for j in range(len(pieces)):
                if i == j:
                    continue
                merged, bridge_test_id = _try_bridge(
                    pieces[i], pieces[j], bridge_test_edge, edge_max_dist_m,
                    deadline, bridge_test_id)
                if merged is not None:
                    pieces = [p for k, p in enumerate(pieces) if k not in (i, j)] + [merged]
                    changed = True
                    break
            if changed:
                break
    return pieces


def _links_by_node(path_edges):
    """{node key: {neighbour key: [views kept, views total]}}.

    Stored per node so it survives the per-node metadata JSON the Hub
    already holds.
    """
    links = {}
    for a, b, keep_a, keep_b in path_edges:
        links.setdefault(a, {})[b] = keep_a
        links.setdefault(b, {})[a] = keep_b
    return links


def _piece_edges(metadata):
    """[(a, b, keep_a, keep_b)] rebuilt from the links in `metadata`."""
    edges, seen = [], set()
    for a, m in metadata.items():
        for b, keep_a in (m.get("links") or {}).items():
            if b not in metadata or (b, a) in seen:
                continue
            seen.add((a, b))
            edges.append((a, b, keep_a, (metadata[b].get("links") or {}).get(a)))
    return edges


def pieces_to_output(pieces):
    """[(clouds, metadata), ...] -- one entry per still-separate piece.

    clouds is {node key: (points, colors)}: DA3 only ever reconstructs one
    or two panoramas at a time and a node's points enter exactly once, so
    every point belongs to a known node and nothing needs re-deriving it.

    metadata carries, per node, its real lat/lon/date, its position and
    rotation in this piece's frame, the view counts behind DA3's own
    confidence in it, and its links to the nodes it was reconstructed with.
    """
    results = []
    for clouds, path_edges, date, reached, node_positions, frame_poses in pieces:
        links = _links_by_node(path_edges)
        metadata = {k: {"lat": lat, "lon": lon, "date": date, "position": pos.tolist(),
                        "rotation": rot.tolist(), "n_views_kept": n_kept,
                        "n_views_total": n_total,
                        **({"links": links[k]} if k in links else {})}
                    for k, (pos, rot, path, lat, lon, n_kept, n_total) in frame_poses.items()}
        results.append(({k: v for k, v in clouds.items() if k in metadata}, metadata))
    return results


def join_segments(segments, bridge_test_edge, edge_max_dist_m=BRIDGE_MAX_DIST_M,
                   max_time_budget_s: float = 200.0):
    """Bridge segments together, then hand back what is left.

    No GPS fit and no shared frame across pieces that never bridged: each
    stays in whichever DA3 frame its own bridge chain anchored to, which
    is exactly what makes a piece a connected component later.
    """
    if not segments:
        raise ValueError("No segments to join.")
    deadline = time.monotonic() + max_time_budget_s
    pieces = bridge_pieces(segments, bridge_test_edge, edge_max_dist_m, deadline)
    print(f"join: bridge_pieces: {len(segments)} piece(s) in, {len(pieces)} piece(s) out "
          f"({len(segments) - len(pieces)} merge(s))")
    return pieces_to_output(pieces)
