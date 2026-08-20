"""Given N date graphs, reconstruct a graph such that it is the most
complete version while using the least segments.

This module owns ONLY that algorithm -- no GPU, no dates/download
orchestration, no candidate-gathering. It calls a test_edge(path_a,
path_b, test_id) -> result-or-None callback for each candidate edge; the
caller (services/pipeline_runner.py's @spaces.GPU-decorated function)
owns the loaded DA3Model and builds that callback around
panoramic_to_3dgs.test_edge_da3. This split exists because of ZeroGPU,
not for its own sake: GPU access is only granted for the duration of one
@spaces.GPU call, so the whole decision loop (which edge to try next,
based on the previous edge's real result) has to run inside that one
call -- but nothing about WHERE that decision code is defined matters to
ZeroGPU, so it lives here, next to the rest of the corridor/date logic it
actually reasons about, rather than inside the GPU package which has no
business knowing what a "corridor" or "date" is.
"""
import time
from collections import deque

import numpy as np

from services.geo import haversine_m


def rigid_align(shared_from: list[tuple[np.ndarray, np.ndarray]], shared_to: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Average rigid transform (R, t) mapping the 'from' frame onto the
    'to' frame, given 1+ shared anchor poses (center, rotation) expressed
    in both. Rotation averaged via quaternion mean, translation directly.
    Duplicated from panoramic_to_3dgs.rigid_align rather than imported --
    that package's __init__ pulls in SplatGenerator/DA3Model/sharp at
    import time (real GPU deps), which this module has no business paying
    for just to reuse ~15 lines of pure numpy/scipy math.

    r_from/r_to are world-to-pano rotations for the SAME physical anchor,
    expressed in each call's own arbitrary world frame (v_pano = r @ v_world).
    For a direction to agree either way it's expressed: r_from @ v_from ==
    r_to @ v_to, and v_to = R @ v_from, so r_from = r_to @ R, i.e.
    R = r_to^-1 @ r_from = r_to.T @ r_from (rotations are orthogonal)."""
    from scipy.spatial.transform import Rotation

    Rs, ts = [], []
    for (c_from, r_from), (c_to, r_to) in zip(shared_from, shared_to):
        R = r_to.T @ r_from
        Rs.append(R)
        ts.append(c_to - R @ c_from)
    quats = np.array([Rotation.from_matrix(R).as_quat() for R in Rs])
    quats *= np.sign(quats @ quats[0])[:, None]
    return Rotation.from_quat(quats.mean(axis=0)).as_matrix(), np.mean(ts, axis=0)


# Rough real DA3 pairwise-test cost x ~2 tests/dot (most dots succeed on
# the first candidate; occasional floods/restarts need more) -- scales
# the search's own time budget to corridor size (see
# run_pathfind_reconstruction's deadline calc) instead of every corridor
# getting the same flat allowance regardless of how much there is to walk.
#
# Calibrated from two real measurements (see tests/debug_solo_score_experiment.py
# for the full methodology/results): a real production pathfind run
# averaged ~3.2s/pairwise-test (27 tests in 86.5s, includes rigid_align +
# point-cloud merge overhead), while an isolated experiment doing only
# raw DA3 pairwise calls averaged ~2.0s/test (no merge overhead). ~3.0s
# padded per test x ~2 tests/dot -> 6.0s/dot.
SECONDS_PER_DOT_ESTIMATE = 6.0

# Phase 3 (bridge_pieces) constants. Bridging never falls back to GPS
# between two pieces -- ANY real DA3 estimate, however weak, is trusted
# over independent GPS placement (GPS only ever anchors the final
# combined result to real-world coordinates once, at the very end, not
# used to reconcile pieces against each other). These two constants only
# decide when a match is confident enough to stop searching early vs.
# needing to rank every attempt tried and take the best -- they never
# disqualify a result from being used at all.
BRIDGE_KEEP_RATE = 0.5
# An average deviation-among-kept-views this large (not a single outlier
# -- those get filtered out already, see test_edge_da3_bridge) means the
# surviving views still don't agree with each other, a real sign the pair
# is worse than usual -- only used to break ties when ranking attempts,
# never to discard a result outright.
BRIDGE_RIDICULOUS_DEV_M = 2.0
# Real DA3 calls spent trying to bridge one pair of pieces, capped
# regardless of how many (Ax, By) node pairs qualify by distance.
BRIDGE_MAX_ATTEMPTS = 10


def run_pathfind_reconstruction(
    date_graphs: list[dict],
    points: list[tuple[float, float]],
    adjacency: dict[int, list[int]],
    start_lat: float,
    start_lon: float,
    test_edge,
    score_pano=None,
    bridge_test_edge=None,
    start_zone_m: float = 5.0,
    point_cover_tolerance_m: float = 15.0,
    edge_max_dist_m: float = 18.0,
    max_time_budget_s: float = 220.0,
    early_exit_segments: int = 4,
) -> list[tuple]:
    """Two-phase pathfind.

    - Phase 1 (map_date): per date graph, walk dot-by-dot over the shared
      corridor adjacency (see street_builder/build_graph/fetch_nodes.py --
      dot i's structural neighbors, independent of which real panos end
      up at either dot). From a confirmed dot, try each structural
      neighbor's own top candidates against the confirmed pano; if that
      neighbor is empty or every candidate fails, try skipping past it to
      ITS neighbors instead, but only within real edge_max_dist_m (the
      corridor's own capture density can't be trusted to have SOMETHING
      at every single dot). Every dot is given exactly one chance, ever,
      to connect in from wherever first reaches it -- no retries, no
      re-scored frontier, no dead_edges bookkeeping needed. On dead end
      (BFS queue drains before the whole corridor is covered), restart a
      fresh piece from whichever untried non-empty dot is closest to the
      nearest still-uncovered corridor point. Produces N disconnected
      pieces per date. Early-exits date exploration once pieces so far
      already need < early_exit_segments to fully cover the corridor.

      Bounded by ONE shared wall-clock deadline across ALL dates combined,
      not a per-date call count -- this call runs inside a single
      @spaces.GPU window with a real, fixed wall-clock duration (ZeroGPU
      kills the call outright once it's up, regardless of what's
      mid-flight), so the real constraint was always time, not "how many
      tests." A call-count budget was only ever an approximation of that,
      and a bad one once calls stop being uniform cost (e.g. a future
      solo-pano scoring pass alongside the pairwise tests). The deadline
      itself scales with corridor size (len(points) * SECONDS_PER_DOT_ESTIMATE)
      so a short street doesn't wait around for a budget sized for a long
      one, capped at max_time_budget_s -- the caller's own real GPU window,
      margined for model load/teardown (see pipeline_runner.py).
    - Phase 2 (set_cover): greedy set cover over every piece from every
      date mapped -- picks fewest pieces covering the most corridor.

    Why not search toward goal points directly (earlier v1 design): one
    walk only chases one branch at a time, so N disconnected branches
    force N segments -- even when a single other date's own pieces
    could've covered several branches at once. Not discoverable without
    seeing the whole per-date picture first.

    Inputs (pre-downloaded by caller, no network here):
    - date_graphs: [{"date": str, "dot_candidates": {dot_index: [(key,
      path, lat, lon), ...]}}, ...], ranked best first, already
      capped/isolated per date (see build_corridor_graphs) -- this
      function tries them in the given order and stops once
      early_exit_segments is satisfied, it doesn't re-rank them.
    - points/adjacency: the corridor's shared spine and dot-to-dot
      structural graph (see fetch_nodes.interpolate_points) -- dates
      never share real panos, but they all walk the same structure.
    - test_edge(path_a, path_b, test_id) -> (pose_a, pose_b, pts, cols)
      or None on failure. The only GPU-touching thing this function calls
      for real connectivity.
    - score_pano(path) -> int, optional. A candidate's solo DA3 self-
      consistency score (higher = more internally coherent, correlates
      with real pairwise success -- see tests/debug_solo_score_experiment.py
      for the real-data validation: 33% success at score 6 up to 100% at
      score 13+). When given, a dot's own candidates get scored lazily --
      only the first time the walk actually reaches that dot, never
      upfront for dots that end up skipped entirely -- and tried
      best-scored-first instead of in whatever order dot_candidates gave
      them. None (default) skips scoring, preserving the given order.
    - bridge_test_edge(path_a, path_b, test_id) -> dict | None, optional.
      Diagnostic pairwise test for Phase 3 (bridge_pieces) -- see
      panoramic_to_3dgs.test_edge_da3_bridge. Unlike test_edge, never
      gates pass/fail itself; returns raw keep-rate/deviation data so
      bridge_pieces can rank several attempts and ALWAYS use the best
      one found, however weak -- a real DA3 estimate is trusted over
      independent GPS placement between two pieces, full stop, no
      fallback. None (default) skips Phase 3 entirely -- pieces are
      returned exactly as Phase 2's set_cover chose them, independent
      GPS-only placement is the caller's (join_segments.py's) job,
      unchanged from before.

    Segments are NOT fully stitched together even with bridging enabled
    -- only pieces bridge_pieces actually managed to connect share a
    frame; joining whatever's left (GPS + ICP) is still the caller's job.

    Returns [(pts, cols, path_edges, date, reached_all, node_positions), ...],
    phase 3's (post-bridging) pieces. reached_all: whole corridor covered.
    node_positions: {key: np.ndarray(3,)}, DA3's placement in that
    piece's own frame, for the caller's join step.
    """
    if not date_graphs or not points:
        return []

    def pdist(lat, lon, pi):
        return haversine_m(lat, lon, points[pi][0], points[pi][1])

    def map_date(date, dot_candidates, test_offset, deadline):
        """Phase 1 for ONE date's own dot_candidates. Returns (pieces,
        tests_used); pieces: list of (pts, cols, path_edges,
        node_positions, covered_point_indices). deadline: shared
        time.monotonic() cutoff across every date in this call, not a
        per-date allowance."""
        confirmed = {}  # dot_index -> {key, path, lat, lon, seg_R, seg_t, pose, piece_id}
        piece_data = {}  # piece_id -> {pts, cols, path_edges}
        next_piece_id = [0]
        visited = set()  # dot indices already given their one chance (confirmed or genuinely failed)
        tests_used = [0]
        score_cache = {}  # pano key -> solo score, so a candidate never gets re-scored twice

        def score_sorted(candidates):
            """Best-solo-score-first ordering of a dot's own candidates,
            computed lazily right here (only for a dot the walk actually
            reached) -- never upfront for the whole corridor. No-op
            (original order) with no scorer configured, nothing to
            reorder, or the deadline's already passed (graceful degrade,
            not a wasted call)."""
            if score_pano is None or len(candidates) <= 1 or time.monotonic() >= deadline:
                return candidates
            scored = []
            for c in candidates:
                key = c[0]
                if key not in score_cache:
                    score_cache[key] = score_pano(c[1])
                scored.append((score_cache[key], c))
            scored.sort(key=lambda sc: sc[0], reverse=True)
            return [c for _, c in scored]

        def covered_points(dots):
            covered = set()
            for d in dots:
                lat, lon = confirmed[d]["lat"], confirmed[d]["lon"]
                for pi in range(len(points)):
                    if pi not in covered and pdist(lat, lon, pi) <= point_cover_tolerance_m:
                        covered.add(pi)
            return covered

        def test_and_confirm(from_dot, from_key, from_path, from_lat, from_lon, to_dot, to_key, to_path, to_lat, to_lon):
            """One real DA3 test. On success, confirms to_dot -- and
            from_dot too if it wasn't already (the bootstrap case: a
            single test_edge call already returns both poses in one
            shared frame, so the founding edge of a piece needs no
            rigid_align at all, only later extensions do)."""
            if time.monotonic() >= deadline:
                return False
            result = test_edge(from_path, to_path, f"{date}_{test_offset + tests_used[0]}")
            tests_used[0] += 1
            if result is None:
                print(f"[{date}] {from_key} -> {to_key}: FAIL")
                return False
            pose_a, pose_b, pts, cols = result

            if from_dot not in confirmed:
                pid = next_piece_id[0]
                next_piece_id[0] += 1
                confirmed[from_dot] = {"key": from_key, "path": from_path, "lat": from_lat, "lon": from_lon,
                                        "seg_R": np.eye(3), "seg_t": np.zeros(3), "pose": pose_a, "piece_id": pid}
                piece_data[pid] = {"pts": pts, "cols": cols, "path_edges": []}
            else:
                pf = confirmed[from_dot]
                pid = pf["piece_id"]
                local_R, local_t = rigid_align([pose_a], [pf["pose"]])
                seg_R = pf["seg_R"] @ local_R
                seg_t = pf["seg_R"] @ local_t + pf["seg_t"]
                pd = piece_data[pid]
                pd["pts"] = np.concatenate([pd["pts"], pts @ seg_R.T + seg_t], axis=0)
                pd["cols"] = np.concatenate([pd["cols"], cols], axis=0)
                confirmed[to_dot] = {"key": to_key, "path": to_path, "lat": to_lat, "lon": to_lon,
                                      "seg_R": seg_R, "seg_t": seg_t, "pose": pose_b, "piece_id": pid}
                piece_data[pid]["path_edges"].append((from_key, to_key))

            if to_dot not in confirmed:
                confirmed[to_dot] = {"key": to_key, "path": to_path, "lat": to_lat, "lon": to_lon,
                                      "seg_R": np.eye(3), "seg_t": np.zeros(3), "pose": pose_b, "piece_id": pid}
                piece_data[pid]["path_edges"].append((from_key, to_key))

            print(f"[{date}] {from_key} -> {to_key}: OK")
            return True

        def try_target(from_dot, to_dot, to_candidates):
            """Try to_candidates (best solo-score first) against from_dot's
            one confirmed pano; if from_dot itself isn't confirmed yet
            (bootstrap), also score-sort ITS OWN candidates and try
            best-from x best-to first, falling back down each list.
            First success wins."""
            if from_dot in confirmed:
                c = confirmed[from_dot]
                for key, path, lat, lon in score_sorted(to_candidates):
                    if test_and_confirm(from_dot, c["key"], c["path"], c["lat"], c["lon"], to_dot, key, path, lat, lon):
                        return True
                return False
            from_candidates = score_sorted(dot_candidates.get(from_dot, []))
            to_candidates_sorted = score_sorted(to_candidates)
            for fk, fpath, flat, flon in from_candidates:
                for key, path, lat, lon in to_candidates_sorted:
                    if test_and_confirm(from_dot, fk, fpath, flat, flon, to_dot, key, path, lat, lon):
                        return True
            return False

        queue = deque()

        def visit(dot):
            """Give `dot` its one chance to reach every structural
            direction out of it. The immediate structural neighbor is
            always tried directly, no distance check -- that's what
            "structurally adjacent" means. If that neighbor is empty or
            every candidate there fails, flood PAST it through further
            empty (or already-failed) dots -- as far as real distance
            allows, capped at edge_max_dist_m from `dot` -- and try every
            candidate-bearing dot found that way, closest real-distance
            first, until one succeeds. This bridges a gap of ANY width,
            not a fixed hop count: the real constraint was always "close
            enough to plausibly connect," never "exactly one empty dot in
            between." Each dot is only ever tested once, ever, across
            this whole date -- nothing is ever retried, so no dead-edge
            tracking is needed. Works uniformly for a brand-new bootstrap
            dot (not yet confirmed) and a normal already-confirmed
            frontier dot, since try_target itself handles both cases."""
            was_confirmed = dot in confirmed
            lat0, lon0 = points[dot]

            def confirm_dot():
                nonlocal was_confirmed
                if not was_confirmed:
                    queue.append(dot)
                    was_confirmed = True
                    visited.add(dot)

            for nb in adjacency.get(dot, []):
                if nb in visited or nb in confirmed:
                    continue
                if try_target(dot, nb, dot_candidates.get(nb, [])):
                    visited.add(nb)
                    confirm_dot()
                    queue.append(nb)
                    continue
                visited.add(nb)

                seen = {dot, nb}
                frontier = [k for k in adjacency.get(nb, []) if k not in seen and k not in confirmed]
                seen.update(frontier)
                reachable = []  # (dist_m, dot_index)
                while frontier:
                    d = frontier.pop()
                    d_lat, d_lon = points[d]
                    dist = haversine_m(lat0, lon0, d_lat, d_lon)
                    if dist > edge_max_dist_m:
                        continue
                    if dot_candidates.get(d) and d not in visited:
                        reachable.append((dist, d))
                    for nxt in adjacency.get(d, []):
                        if nxt not in seen and nxt not in confirmed:
                            seen.add(nxt)
                            frontier.append(nxt)
                reachable.sort()

                for _, target in reachable:
                    if target in visited:
                        continue  # reached via more than one branch this call
                    ok = try_target(dot, target, dot_candidates.get(target, []))
                    visited.add(target)
                    if ok:
                        confirm_dot()
                        queue.append(target)
                        break

        def pick_seed(uncovered):
            """Nearest untried non-empty dot to the real start (very
            first seed of this date) or to the nearest still-uncovered
            corridor point (later restarts, once a piece's own growth
            has fully drained but the corridor isn't covered yet)."""
            candidates = [d for d in dot_candidates if d not in visited]
            if not candidates:
                return None
            if not confirmed:
                in_zone = [d for d in candidates
                           if haversine_m(points[d][0], points[d][1], start_lat, start_lon) <= start_zone_m]
                pool = in_zone or candidates
                return min(pool, key=lambda d: haversine_m(points[d][0], points[d][1], start_lat, start_lon))
            return min(candidates, key=lambda d: min(pdist(points[d][0], points[d][1], pi) for pi in uncovered))

        while time.monotonic() < deadline:
            uncovered = set(range(len(points))) - covered_points(confirmed.keys())
            if not uncovered:
                break
            if queue:
                visit(queue.popleft())
                continue
            seed = pick_seed(uncovered)
            if seed is None:
                break
            visit(seed)
            if seed not in confirmed:
                visited.add(seed)

        pieces = []
        for pid, pd in piece_data.items():
            dots = [d for d, c in confirmed.items() if c["piece_id"] == pid]
            node_positions = {confirmed[d]["key"]: confirmed[d]["seg_R"] @ confirmed[d]["pose"][0] + confirmed[d]["seg_t"] for d in dots}
            # Each node's own (center, rotation, path, real lat, real lon)
            # re-expressed in the piece's shared frame (path/lat/lon are
            # unchanged, just carried along) -- node_positions alone is
            # enough for GPS-fitting (join_segments.py), but bridging two
            # pieces together (see bridge_pieces) needs the full pose +
            # path to run a NEW real test and chain rigid_align onto this
            # piece's existing frame, and REAL lat/lon (not the DA3-frame
            # position, which is meaningless to compare across two
            # different pieces' unrelated local coordinate frames) to
            # decide which node pairs are even worth attempting.
            # Internal-only: never leaves run_pathfind_reconstruction.
            frame_poses = {confirmed[d]["key"]: (node_positions[confirmed[d]["key"]], confirmed[d]["pose"][1] @ confirmed[d]["seg_R"].T,
                                                   confirmed[d]["path"], confirmed[d]["lat"], confirmed[d]["lon"]) for d in dots}
            pieces.append((pd["pts"], pd["cols"], pd["path_edges"], node_positions, covered_points(dots), frame_poses))
        return pieces, tests_used[0]

    def set_cover(pieces, total_points):
        """Phase 2: greedy set cover. Repeatedly take whichever piece
        (from any date) covers the most still-uncovered corridor
        points, until covered or nothing left adds anything new.
        Returns (chosen, leftover_uncovered)."""
        uncovered = set(range(total_points))
        chosen = []
        pool = list(pieces)
        while uncovered and pool:
            pool.sort(key=lambda p: len(p[4] & uncovered), reverse=True)
            top = pool[0]
            if not (top[4] & uncovered):
                break
            chosen.append(top)
            uncovered -= top[4]
            pool.pop(0)
        return chosen, uncovered

    def try_bridge(a, b, deadline, bridge_test_id):
        """One pair's worth of Phase 3 search: every (Ax, By) node pair
        within edge_max_dist_m, same-date-first then closest-first, up to
        BRIDGE_MAX_ATTEMPTS real tests. ALWAYS merges using whichever
        attempt came out best, however weak -- even a poor DA3 estimate
        beats independent GPS placement between two pieces, and GPS is
        never used for that (see run_pathfind_reconstruction's own
        docstring). A clearly confident match (clears BRIDGE_KEEP_RATE on
        both sides and no bad-consensus red flag) stops the search early;
        otherwise every attempt is ranked and the best one wins once the
        attempt budget/deadline is hit.
        Returns (merged_piece, next_bridge_test_id) or (None, next_bridge_test_id)
        only if there were no (Ax, By) pairs within range to try at all --
        the one remaining case join_segments.py's GPS fit still has to
        cover, since there's no real signal to use in the first place."""
        a_pts, a_cols, a_edges, a_positions, a_covered, a_frame_poses, a_date = a
        b_pts, b_cols, b_edges, b_positions, b_covered, b_frame_poses, b_date = b

        pairs = []
        for a_key, (_, _, _, a_lat, a_lon) in a_frame_poses.items():
            for b_key, (_, _, _, b_lat, b_lon) in b_frame_poses.items():
                # REAL geographic distance, not DA3-frame position -- the
                # two pieces' DA3 frames are unrelated coordinate systems
                # (different scale/origin/orientation each), comparing
                # positions across them is meaningless. lat/lon is the
                # only thing both pieces agree on.
                dist = haversine_m(a_lat, a_lon, b_lat, b_lon)
                if dist <= edge_max_dist_m:
                    pairs.append((a_date != b_date, dist, a_key, b_key))
        if not pairs:
            return None, bridge_test_id
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
            # (confident?, sane?, min keep-rate, -combined avg_dev) --
            # ranks a genuinely good match first, then prefers a sane
            # result over a flagged one, then the best of what's left by
            # keep-rate/deviation. Never disqualifies outright -- there's
            # always a best available, and it's always used.
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
        merged_covered = a_covered | b_covered
        print(f"[bridge] {a_date}+{b_date}: merged via {a_key} -> {b_key} (keep={result['keep_a']},{result['keep_b']})")
        return (merged_pts, merged_cols, merged_edges, merged_positions, merged_covered, merged_frame_poses, a_date), bridge_test_id

    def bridge_pieces(chosen, deadline):
        """Phase 3: try to replace independent GPS placement between
        geographically close pieces with a real DA3-verified transform
        (see run_pathfind_reconstruction's own docstring for the full
        design). Greedily merges pairs until nothing more merges or the
        deadline hits. Returns a new list of (possibly merged) pieces,
        same 7-tuple shape as `chosen`."""
        if bridge_test_edge is None or len(chosen) < 2:
            return chosen

        pieces = list(chosen)
        bridge_test_id = 0
        changed = True
        while changed and len(pieces) > 1 and time.monotonic() < deadline:
            changed = False
            for i in range(len(pieces)):
                if changed:
                    break
                for j in range(len(pieces)):
                    if i == j:
                        continue
                    merged, bridge_test_id = try_bridge(pieces[i], pieces[j], deadline, bridge_test_id)
                    if merged is not None:
                        pieces = [p for k, p in enumerate(pieces) if k not in (i, j)] + [merged]
                        changed = True
                        break
        return pieces

    all_pieces = []  # (pts, cols, path_edges, node_positions, covered, frame_poses, date)
    total_tests = 0
    time_budget_s = min(len(points) * SECONDS_PER_DOT_ESTIMATE, max_time_budget_s)
    deadline = time.monotonic() + time_budget_s
    print(f"pathfind: time budget {time_budget_s:.0f}s ({len(points)} dot(s) x {SECONDS_PER_DOT_ESTIMATE}s, capped at {max_time_budget_s:.0f}s)")

    for date_graph in date_graphs:
        if time.monotonic() >= deadline:
            print("pathfind: time budget exhausted -- stopping date exploration")
            break

        date, dot_candidates = date_graph["date"], date_graph["dot_candidates"]
        pieces, tests_used = map_date(date, dot_candidates, total_tests, deadline)
        total_tests += tests_used
        for p in pieces:
            all_pieces.append(p + (date,))
        print(f"pathfind: date {date} mapped into {len(pieces)} piece(s), {total_tests} attempts so far")

        chosen_so_far, uncovered_so_far = set_cover(all_pieces, len(points))
        if chosen_so_far and not uncovered_so_far and len(chosen_so_far) < early_exit_segments:
            print(f"pathfind: {len(chosen_so_far)} segment(s) already cover everything -- stopping date exploration")
            break

    chosen, leftover_uncovered = set_cover(all_pieces, len(points))

    n_before_bridge = len(chosen)
    chosen = bridge_pieces(chosen, deadline)
    if len(chosen) != n_before_bridge:
        print(f"pathfind: bridge_pieces merged {n_before_bridge} piece(s) into {len(chosen)}")

    reached_all = not leftover_uncovered
    segments = [
        (pts, cols, path_edges, date, reached_all, node_positions)
        for pts, cols, path_edges, node_positions, covered, frame_poses, date in chosen
    ]
    print(f"pathfind: {total_tests} attempts total, {len(date_graphs)} date(s) considered, {len(all_pieces)} piece(s) found, {len(segments)} segment(s) chosen, corridor {'fully' if reached_all else 'partially'} covered ({len(leftover_uncovered)}/{len(points)} point(s) never covered)")
    return segments
