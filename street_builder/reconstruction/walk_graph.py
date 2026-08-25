"""Given N date graphs, reconstruct a graph such that it is the most
complete version while using the least segments.

This module owns ONLY that algorithm -- no GPU, no dates/download
orchestration, no candidate-gathering. It calls a test_edge(path_a,
path_b, test_id) -> result-or-None callback for each candidate edge; the
caller (services/pipeline_runner.py's @spaces.GPU-decorated function)
owns the loaded DA3Model and builds that callback around
services.da3_ops.test_edge. This split exists because of ZeroGPU,
not for its own sake: GPU access is only granted for the duration of one
@spaces.GPU call, so the whole decision loop (which edge to try next,
based on the previous edge's real result) has to run inside that one
call -- but nothing about WHERE that decision code is defined matters to
ZeroGPU, so it lives here, next to the rest of the corridor/date logic it
actually reasons about, rather than inside the GPU package which has no
business knowing what a "corridor" or "date" is.
"""
import os
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
# the first candidate; occasional restarts need more) -- scales
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


PROTECTED_POSITION_MATCH_M = 3.0


def _rescue_protected_pieces(chosen, all_pieces, leftover_uncovered, protected_positions):
    """After set_cover has already picked its coverage-optimal pieces,
    force back in any piece covering a `protected_positions` entry that
    got dropped as geographically redundant -- see
    run_pathfind_reconstruction's own docstring for why. Pure
    bookkeeping, no GPU/network -- factored out from
    run_pathfind_reconstruction so it's directly unit-testable without
    needing a real walk to exercise it (see tests/test_pathfind_scenarios.py).

    Matched by REAL DISTANCE (within PROTECTED_POSITION_MATCH_M), not by
    node key -- confirmed empirically (a real chunk-boundary bridge
    failure) that a node key is date-specific, not location-stable: the
    exact same real spot gets a totally different pano id on every
    historical date, so exact-key matching can never rescue a location
    whose winning date differs from whichever date the caller's own
    boundary-node snapshot came from. protected_positions: set of (lat,
    lon) real-world coordinates that must end up in the result if
    reconstructed at all, in ANY date.

    chosen/all_pieces: (pts, cols, path_edges, node_positions, covered,
    frame_poses, date) tuples -- id()-based membership check throughout,
    NOT ==, since these tuples hold numpy arrays (pts/cols) that make `==`
    ambiguous/raise. Returns (chosen, leftover_uncovered), both possibly
    updated in place... actually returned fresh, not mutated -- chosen is
    the same list object appended to, leftover_uncovered is a new set."""
    if not protected_positions:
        return chosen, leftover_uncovered

    def piece_has_position(piece, target_lat, target_lon):
        for (_pos, _rot, _path, lat, lon, _nk, _nt) in piece[5].values():  # piece[5] == frame_poses
            if haversine_m(lat, lon, target_lat, target_lon) <= PROTECTED_POSITION_MATCH_M:
                return True
        return False

    missing = [pos for pos in protected_positions if not any(piece_has_position(c, *pos) for c in chosen)]
    if not missing:
        return chosen, leftover_uncovered

    rescued = 0
    chosen_ids = {id(c) for c in chosen}
    still_missing = []
    for target_lat, target_lon in missing:
        for p in all_pieces:
            if id(p) in chosen_ids:
                continue
            if piece_has_position(p, target_lat, target_lon):
                chosen.append(p)
                chosen_ids.add(id(p))
                leftover_uncovered = leftover_uncovered - p[4]  # p[4] == covered
                rescued += 1
                break
        else:
            still_missing.append((target_lat, target_lon))
    if rescued:
        print(f"pathfind: rescued {rescued} piece(s) covering protected location(s) set_cover had dropped as redundant")
    if still_missing:
        print(f"pathfind: {len(still_missing)} protected location(s) never reconstructed in any date, nothing to rescue: {still_missing}")
    return chosen, leftover_uncovered


def run_pathfind_reconstruction(
    date_graphs: list[dict],
    points: list[tuple[float, float]],
    adjacency: dict[int, list[int]],
    start_lat: float,
    start_lon: float,
    test_edge,
    rate_pano=None,
    point_cover_tolerance_m: float = 15.0,
    max_time_budget_s: float = 220.0,
    early_exit_segments: int = 4,
    protected_positions: set = None,
) -> list[tuple]:
    """Two-phase pathfind.

    - Phase 1 (map_date): per date graph, walk dot-by-dot over the shared
      corridor adjacency (see street_builder/build_graph/fetch_nodes.py --
      dot i's structural neighbors, independent of which real panos end
      up at either dot). The FIRST time a dot is ever looked at (as a walk
      target OR a seed), it's rated (see rate_pano below) and keeps its
      own best-scoring candidate's REAL solo point cloud as a one-node
      piece -- so every dot the walk ever touches ends up in the output,
      even if it never successfully pairs with anything. From there, try
      each structural neighbor's own top candidates against the current
      dot's established candidate; a dot is now a real selection-graph
      node (not an interpolated sample point), so an empty/failed
      neighbor is a genuine dead end for that date, not skipped past --
      no flood-past-empty-dot fallback. A successful pairwise test MERGES
      the two dots' pieces (discarding the
      newly-reached dot's own solo piece in favor of this edge's own
      jointly-reconstructed, higher-quality points for it -- the
      already-established side is never re-added, so its points never get
      duplicated across however many further edges touch it). Every dot is
      given exactly one chance, ever, to connect in from wherever first
      reaches it -- no retries, no re-scored frontier, no dead_edges
      bookkeeping needed. On dead end (BFS queue drains before the whole
      corridor is covered), restart a fresh piece from whichever untried
      non-empty dot is closest to the nearest still-uncovered corridor
      point. Produces N disconnected pieces per date (each already
      guaranteed non-empty by the per-dot rating above). Early-exits date
      exploration once pieces so far already need < early_exit_segments to
      fully cover the corridor.

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
      structural graph (see fetch_nodes.corridor_points) -- dates
      never share real panos, but they all walk the same structure.
    - test_edge(path_a, path_b, test_id) -> (pose_a, pose_b, pts, cols,
      per_pano_pts, per_pano_cols) or None on failure. per_pano_pts/cols:
      {os.path.basename(path): points/colors} -- used to add only the
      newly-reached dot's own slice of a successful pairwise result onto
      an existing piece (see test_and_confirm), not the whole pairwise
      result. The only GPU-touching thing this function calls for real
      connectivity.
    - rate_pano(path) -> (score, pose, pts, cols), optional. A candidate's
      solo DA3 self-consistency score (higher = more internally coherent,
      correlates with real pairwise success -- see
      tests/debug_solo_score_experiment.py for the real-data validation:
      33% success at score 6 up to 100% at score 13+) PLUS that pano's own
      real solo point cloud (pose/pts/cols, same shape/frame convention as
      test_edge's pose_a/pose_b/pts). When given, a dot's own candidates
      get rated lazily -- only the first time the walk actually reaches
      that dot, never upfront for dots that end up skipped entirely -- the
      best-scored one is tried first for pairwise tests AND becomes that
      dot's guaranteed fallback one-node piece (see ensure_piece). None
      (default) skips rating entirely, preserving the given candidate
      order -- but then a dot that never pairs with anything is dropped
      instead of falling back to a solo piece (old behavior).

    Segments are NOT stitched together -- each is DA3's own arbitrary
    frame; joining/bridging is entirely the caller's job (see
    street_builder/reconstruction/join_segments.py's bridge_pieces,
    which reconciles pieces using real DA3 tests, and join_segments,
    which GPS-fits whatever's left over -- both run in their own later
    GPU call, not this one, since bridging needs no data this function
    doesn't already expose).

    protected_positions: optional set of (lat, lon) real-world coordinates
    that MUST end up in the returned segments if reconstructed at all (in
    ANY date), even if set_cover would otherwise drop their piece as
    geographically redundant. For a chunked large-area reconstruction,
    these are a chunk's own real boundary node COORDINATES (real edges to
    a neighboring chunk, known from the chunking step itself) -- set_cover
    only optimizes for covering THIS chunk's own corridor, so a boundary
    location already covered by a different date's piece looks
    "redundant" and gets discarded, even though it's exactly what a later
    cross-chunk bridge attempt needs. Matched by real distance, not node
    key -- a node key is date-specific (the same real spot gets a
    different pano id on every historical date), so exact-key matching
    can't rescue a location whose winning date differs from whichever
    date the coordinate itself came from (see _rescue_protected_pieces).
    A protected position with zero real candidates anywhere just stays
    absent -- this only rescues a location that WAS reconstructed
    somewhere but lost the coverage competition.

    Returns [(pts, cols, path_edges, date, reached_all, node_positions,
    frame_poses), ...], phase 2's (set_cover's) chosen pieces.
    reached_all: whole corridor covered. node_positions: {key:
    np.ndarray(3,)}, DA3's placement in that piece's own frame.
    frame_poses: {key: (center, rotation, path, lat, lon, n_views_kept,
    n_views_total)} -- the fuller per-node data join_segments.py's
    bridge_pieces needs to chain a NEW rigid_align onto this piece's frame
    and gate candidate pairs by real distance, plus view-count diagnostics
    (see services.da3_ops.rate_pano/test_edge -- whichever DA3 call actually
    produced this node's current points); node_positions is just
    frame_poses' own center field, kept separate since it's all the
    simpler GPS-fit path needs.
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
        confirmed = {}  # dot_index -> {key, path, lat, lon, seg_R, seg_t, pose, piece_id} -- has a piece (solo or merged)
        piece_data = {}  # piece_id -> {pts, cols, path_edges}
        next_piece_id = [0]
        visited = set()  # dot indices already given their one chance (whether or not they ended up `confirmed`)
        tests_used = [0]
        rated_cache = {}  # pano key -> (score, pose, pts, cols) from rate_pano, so a candidate never gets re-rated twice

        def rate_one(candidate):
            key, path, lat, lon = candidate
            if key not in rated_cache:
                rated_cache[key] = rate_pano(path)
            return rated_cache[key]

        def rate_sorted(candidates):
            """Best-solo-score-first ordering of a dot's own candidates,
            rated lazily right here (only for a dot the walk actually
            reached) -- never upfront for the whole corridor. No-op
            (original order) with no rater configured, nothing to
            reorder, or the deadline's already passed (graceful degrade,
            not a wasted call). Doesn't rate a lone candidate itself here
            (nothing to sort) -- ensure_piece rates it on demand instead."""
            if rate_pano is None or len(candidates) <= 1 or time.monotonic() >= deadline:
                return candidates
            scored = [(rate_one(c)[0], c) for c in candidates]
            scored.sort(key=lambda sc: sc[0], reverse=True)
            return [c for _, c in scored]

        def ensure_piece(dot):
            """The first time `dot` is ever looked at (as a walk target or
            a seed), rate its own candidates and keep the best-scoring
            one's REAL solo point cloud (even if 'best' still scored
            poorly) as this dot's own one-node piece. Guarantees every dot
            the walk touches ends up with SOME real point data before any
            pairwise test is even attempted -- see test_and_confirm for
            how a later successful edge replaces/merges this baseline with
            higher-quality jointly-reconstructed data, rather than adding
            to it. No-op (dot stays un-piece'd, old drop-on-failure
            behavior) if rate_pano wasn't provided, the deadline's passed,
            there's nothing to rate for this dot on this date, or DA3
            produced no pose at all for the best candidate."""
            if dot in confirmed or rate_pano is None or time.monotonic() >= deadline:
                return
            t0 = time.monotonic()
            raw_candidates = dot_candidates.get(dot, [])
            candidates = rate_sorted(raw_candidates)
            if not candidates:
                return
            key, path, lat, lon = candidates[0]
            score, pose, pts, cols, n_kept, n_total = rate_one((key, path, lat, lon))
            print(f"[timing] ensure_piece(dot={dot}, {len(raw_candidates)} candidate(s) available): "
                  f"{time.monotonic() - t0:.2f}s total, {deadline - time.monotonic():.1f}s left in budget")
            if pose is None:
                return
            pid = next_piece_id[0]
            next_piece_id[0] += 1
            confirmed[dot] = {"key": key, "path": path, "lat": lat, "lon": lon,
                               "seg_R": np.eye(3), "seg_t": np.zeros(3), "pose": pose, "piece_id": pid,
                               "n_views_kept": n_kept, "n_views_total": n_total}
            piece_data[pid] = {"pts": pts, "cols": cols, "path_edges": []}

        def covered_points(dots):
            """A dot's own point is always covered by itself. Any OTHER
            point needs at least 2 distinct confirmed dots within
            point_cover_tolerance_m to count as covered -- a single nearby
            confirmed dot is not enough on its own, since with ensure_piece
            every dot now has its own real, valuable data; only a point
            genuinely flanked by real coverage on multiple sides (a true
            interior gap) is redundant to visit."""
            near_count = {}
            covered = set(dots)
            for d in dots:
                lat, lon = confirmed[d]["lat"], confirmed[d]["lon"]
                for pi in range(len(points)):
                    if pi in covered or pdist(lat, lon, pi) > point_cover_tolerance_m:
                        continue
                    near_count[pi] = near_count.get(pi, 0) + 1
                    if near_count[pi] >= 2:
                        covered.add(pi)
            return covered

        def test_and_confirm(from_dot, from_key, from_path, from_lat, from_lon, to_dot, to_key, to_path, to_lat, to_lon):
            """One real DA3 test. from_dot and to_dot ALWAYS already have
            their own piece by this point (ensure_piece runs on every dot
            before any edge involving it is attempted -- see visit). On
            success, to_dot's own solo/prior piece is discarded and
            replaced by this edge's own per-pano points for to_dot (higher
            quality, jointly reconstructed with from_dot), merged into
            from_dot's existing piece via rigid_align. from_dot's own side
            is left untouched -- never re-added, so an already-established
            node's points don't get duplicated across however many further
            edges touch it."""
            if time.monotonic() >= deadline:
                return False
            t0 = time.monotonic()
            result = test_edge(from_path, to_path, f"{date}_{test_offset + tests_used[0]}")
            t_test = time.monotonic() - t0
            tests_used[0] += 1
            if result is None:
                print(f"[{date}] {from_key} -> {to_key}: FAIL ({t_test:.2f}s, {deadline - time.monotonic():.1f}s left)")
                return False
            pose_a, pose_b, pts, cols, per_pano_pts, per_pano_cols, per_pano_views = result
            to_id = os.path.basename(to_path)
            to_pts = per_pano_pts.get(to_id, np.zeros((0, 3)))
            to_cols = per_pano_cols.get(to_id, np.zeros((0, 3)))
            to_kept, to_total = per_pano_views.get(to_id, (0, 0))

            if from_dot not in confirmed:
                # Bootstrap: from_dot has no piece yet at all -- only
                # possible when rate_pano wasn't provided (ensure_piece
                # guarantees this otherwise, see visit). This edge's own
                # pairwise result founds the piece directly for BOTH
                # sides, no rigid_align needed yet (both poses already
                # share this call's own frame).
                from_id = os.path.basename(from_path)
                from_kept, from_total = per_pano_views.get(from_id, (0, 0))
                pid = next_piece_id[0]
                next_piece_id[0] += 1
                confirmed[from_dot] = {"key": from_key, "path": from_path, "lat": from_lat, "lon": from_lon,
                                        "seg_R": np.eye(3), "seg_t": np.zeros(3), "pose": pose_a, "piece_id": pid,
                                        "n_views_kept": from_kept, "n_views_total": from_total}
                piece_data[pid] = {"pts": pts, "cols": cols, "path_edges": []}
                confirmed[to_dot] = {"key": to_key, "path": to_path, "lat": to_lat, "lon": to_lon,
                                      "seg_R": np.eye(3), "seg_t": np.zeros(3), "pose": pose_b, "piece_id": pid,
                                      "n_views_kept": to_kept, "n_views_total": to_total}
                piece_data[pid]["path_edges"].append((from_key, to_key))
                print(f"[{date}] {from_key} -> {to_key}: OK ({t_test:.2f}s, {deadline - time.monotonic():.1f}s left)")
                return True

            pf = confirmed[from_dot]
            pid = pf["piece_id"]
            if to_dot in confirmed and confirmed[to_dot]["piece_id"] == pid:
                print(f"[{date}] {from_key} -> {to_key}: OK (already same piece, {t_test:.2f}s, {deadline - time.monotonic():.1f}s left)")
                return True
            if to_dot in confirmed:
                # to_dot already has its OWN separate piece (solo from
                # ensure_piece, or already grown further) -- discard it,
                # replace with just its own slice of THIS higher-quality
                # jointly-reconstructed result.
                piece_data.pop(confirmed[to_dot]["piece_id"], None)

            local_R, local_t = rigid_align([pose_a], [pf["pose"]])
            seg_R = pf["seg_R"] @ local_R
            seg_t = pf["seg_R"] @ local_t + pf["seg_t"]
            pd = piece_data[pid]
            pd["pts"] = np.concatenate([pd["pts"], to_pts @ seg_R.T + seg_t], axis=0)
            pd["cols"] = np.concatenate([pd["cols"], to_cols], axis=0)
            pd["path_edges"].append((from_key, to_key))
            confirmed[to_dot] = {"key": to_key, "path": to_path, "lat": to_lat, "lon": to_lon,
                                  "seg_R": seg_R, "seg_t": seg_t, "pose": pose_b, "piece_id": pid,
                                  "n_views_kept": to_kept, "n_views_total": to_total}

            print(f"[{date}] {from_key} -> {to_key}: OK ({t_test:.2f}s, {deadline - time.monotonic():.1f}s left)")
            return True

        def try_target(from_dot, to_dot, to_candidates):
            """Try to_candidates (best solo-score first) against from_dot's
            established candidate, if it has one yet (from ensure_piece,
            or an earlier successful edge); if from_dot has no piece at
            all (only possible when rate_pano wasn't provided), also
            rate-sort ITS OWN candidates and try best-from x best-to
            first, falling back down each list -- the old bootstrap path.
            First success wins."""
            if from_dot in confirmed:
                c = confirmed[from_dot]
                for key, path, lat, lon in rate_sorted(to_candidates):
                    if test_and_confirm(from_dot, c["key"], c["path"], c["lat"], c["lon"], to_dot, key, path, lat, lon):
                        return True
                return False
            from_candidates = rate_sorted(dot_candidates.get(from_dot, []))
            to_candidates_sorted = rate_sorted(to_candidates)
            for fk, fpath, flat, flon in from_candidates:
                for key, path, lat, lon in to_candidates_sorted:
                    if test_and_confirm(from_dot, fk, fpath, flat, flon, to_dot, key, path, lat, lon):
                        return True
            return False

        queue = deque()

        def visit(dot):
            """Give `dot` its one chance to reach every structural
            neighbor out of it. `dot` and every candidate dot looked at
            below get ensure_piece'd first, as a best-effort fallback
            piece for each -- but try_target still works even when that
            didn't produce one (rate_pano not provided), via its own
            bootstrap fallback. No flood-past-empty-dot fallback: a dot
            is a real selection-graph node (not an interpolated sample
            point), so a failed/empty structural neighbor is a genuine
            dead end for that date here, not skipped past. Each dot is
            only ever tested once, ever, across this whole date --
            nothing is ever retried, so no dead-edge tracking is needed."""
            print(f"[timing] visit(dot={dot}): {deadline - time.monotonic():.1f}s left in budget")
            ensure_piece(dot)
            was_confirmed = dot in confirmed

            def confirm_dot():
                nonlocal was_confirmed
                if not was_confirmed:
                    queue.append(dot)
                    was_confirmed = True
                    visited.add(dot)

            for nb in adjacency.get(dot, []):
                if nb in visited or nb in confirmed:
                    continue
                ensure_piece(nb)
                visited.add(nb)
                if try_target(dot, nb, dot_candidates.get(nb, [])):
                    confirm_dot()
                    queue.append(nb)

        def pick_seed(uncovered):
            """Nearest untried non-empty dot to the real start (very
            first seed of this date) or to the nearest still-uncovered
            corridor point (later restarts, once a piece's own growth
            has fully drained but the corridor isn't covered yet)."""
            candidates = [d for d in dot_candidates if d not in visited]
            if not candidates:
                return None
            if not confirmed:
                return min(candidates, key=lambda d: haversine_m(points[d][0], points[d][1], start_lat, start_lon))
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
            visited.add(seed)

        pieces = []
        for pid, pd in piece_data.items():
            dots = [d for d, c in confirmed.items() if c["piece_id"] == pid]
            if not dots:
                continue  # orphaned piece_id (merged away in test_and_confirm) -- shouldn't happen, defensive only
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
                                                   confirmed[d]["path"], confirmed[d]["lat"], confirmed[d]["lon"],
                                                   confirmed[d]["n_views_kept"], confirmed[d]["n_views_total"]) for d in dots}
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
    chosen, leftover_uncovered = _rescue_protected_pieces(chosen, all_pieces, leftover_uncovered, protected_positions)

    reached_all = not leftover_uncovered
    segments = [
        (pts, cols, path_edges, date, reached_all, node_positions, frame_poses)
        for pts, cols, path_edges, node_positions, covered, frame_poses, date in chosen
    ]
    print(f"pathfind: {total_tests} attempts total, {len(date_graphs)} date(s) considered, {len(all_pieces)} piece(s) found, {len(segments)} segment(s) chosen, corridor {'fully' if reached_all else 'partially'} covered ({len(leftover_uncovered)}/{len(points)} point(s) never covered)")
    return segments
