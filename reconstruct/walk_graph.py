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

from services.da3_ops import MIN_KEEP_RATE
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


# Walk cost per dot, used to size the GPU window (see
# services.pipeline_runner.estimate_gpu_seconds) -- the walk itself just
# runs to whatever budget its caller hands it.
#
# First calibrated at 6.0s/dot from ~3.2s/pairwise-test x ~2 tests/dot
# (see tests/debug_solo_score_experiment.py). Too low once dots carry many
# candidates: a 7-dot run (Apple, 6-10 candidates/dot) spent 9.4s just
# rating dot 0's candidates, ~2s per failed pairwise test, and ran out of
# its 42s budget before reaching 4 of the 7 dots or any other date.
SECONDS_PER_DOT_ESTIMATE = 12.0

# A piece at least this many dots long is trusted as it is; every dot
# outside one is weak, and the next date re-walks it (see _patch_dots).
GOOD_PIECE_DOTS = 4
# How far a patch walk reaches past a weak stretch into the good piece on
# either side, in dots. The patch comes from another date, so it is never
# DA3-linked to that piece -- the overlap is shared road for placement's
# cross-road alignment to line the two up by.
PATCH_OVERLAP_DOTS = 1
# Dots in a row that fail to link (none of their candidates linking) after
# which a date is left for the next one -- so one date that DA3 can't make
# sense of can't spend the whole budget. Counted in dots, not tests: with
# up to 3 candidates a dot, a count of tests would give up on a mostly-good
# date at its first two broken dots. 4, not 3: two broken dots side by side
# (o-x-x-o) already fail three links in a row -- into each of them, and
# out of the second.
MAX_FAILED_DOTS_IN_A_ROW = 4
# Panos rated per date before any walking, to order dates by how well DA3
# handles their imagery rather than by coverage alone (see _sample_dates).
DATE_SAMPLES_MAX = 5


def _rescue_protected_pieces(chosen, all_pieces, leftover_uncovered, protected_indices):
    """After set_cover has already picked its coverage-optimal pieces,
    force back in any piece covering a `protected_indices` dot that got
    dropped as geographically redundant -- see
    run_pathfind_reconstruction's own docstring for why. Pure
    bookkeeping, no GPU/network -- factored out from
    run_pathfind_reconstruction so it's directly unit-testable without
    needing a real walk to exercise it (see tests/test_pathfind_scenarios.py).

    Matched by DOT INDEX (piece[6], the raw dot-index set every piece
    tuple carries -- see map_date), not by node key or real distance:
    every date graph walks the exact same points/adjacency object, so a
    dot index is a precise, date-independent structural identity for "this
    real location" -- confirmed empirically (a real chunk-boundary bridge
    failure) that a node KEY is date-specific instead (the same real spot
    gets a totally different pano id on every historical date), so
    exact-key matching can never rescue a location whose winning date
    differs from whichever date the caller's own boundary-node snapshot
    came from. protected_indices: set of dot indices (see
    run_pathfind_reconstruction's own resolution of protected_positions
    -> protected_indices) that must end up in the result if reconstructed
    at all, in ANY date.

    chosen/all_pieces: (clouds, path_edges, node_positions, covered,
    frame_poses, dots, date) tuples -- id()-based membership check
    throughout, NOT ==, since these tuples hold numpy arrays (pts/cols)
    that make `==` ambiguous/raise. Returns (chosen, leftover_uncovered),
    both possibly updated in place... actually returned fresh, not
    mutated -- chosen is the same list object appended to,
    leftover_uncovered is a new set."""
    if not protected_indices:
        return chosen, leftover_uncovered

    chosen_dots = set()
    for p in chosen:
        chosen_dots |= p[5]  # p[5] == dots
    missing = protected_indices - chosen_dots
    if not missing:
        return chosen, leftover_uncovered

    rescued = 0
    chosen_ids = {id(c) for c in chosen}
    for p in all_pieces:
        if not missing:
            break
        if id(p) in chosen_ids:
            continue
        overlap = p[5] & missing
        if overlap:
            chosen.append(p)
            chosen_ids.add(id(p))
            leftover_uncovered = leftover_uncovered - p[3]  # p[3] == covered
            missing -= overlap
            rescued += 1
    if rescued:
        print(f"pathfind: rescued {rescued} piece(s) covering protected dot(s) set_cover had dropped as redundant")
    if missing:
        print(f"pathfind: {len(missing)} protected dot(s) never reconstructed in any date, nothing to rescue: dot indices {sorted(missing)}")
    return chosen, leftover_uncovered


def _sample_dots(dots, k):
    """k of `dots`, evenly spread along the corridor's dot order."""
    dots = sorted(dots)
    if k >= len(dots):
        return dots
    return [dots[round(i * (len(dots) - 1) / (k - 1))] for i in range(k)] if k > 1 else [dots[len(dots) // 2]]


def _sample_dates(date_graphs, n_points, rate):
    """Date graphs in the order to walk them: each one's median solo
    keep-rate over a few sampled panos, times the share of the corridor it
    covers. A date below MIN_KEEP_RATE is dropped -- links between its
    panos would almost all fail -- unless every date is, in which case the
    best one is still walked, so the run has something to show.

    Samples: half the date's own dots, 1 to DATE_SAMPLES_MAX, spread along
    it, rating each dot's closest pano. rate is the walk's cached rater,
    so a sampled pano is never rated twice."""
    scored = []
    for g in date_graphs:
        dots = list(g["dot_candidates"])
        if not dots:
            continue
        k = min(DATE_SAMPLES_MAX, max(1, -(-len(dots) // 2)))
        rates = []
        for d in _sample_dots(dots, k):
            *_, n_kept, n_total = rate(g["dot_candidates"][d][0])
            rates.append(n_kept / n_total if n_total else 0.0)
        median = float(np.median(rates))
        coverage = len(dots) / n_points
        scored.append((median * coverage, median, coverage, g))
        print(f"pathfind: date {g['date']} sampled {len(rates)} pano(s): median keep "
              f"{median:.2f}, covers {coverage:.0%} of the corridor")
    scored.sort(key=lambda t: t[0], reverse=True)
    kept = [t for t in scored if t[1] >= MIN_KEEP_RATE]
    for _, median, _, g in scored:
        if median < MIN_KEEP_RATE:
            print(f"pathfind: date {g['date']} dropped -- median keep {median:.2f} < {MIN_KEEP_RATE:.2f}")
    if not kept and scored:
        print(f"pathfind: every date is below {MIN_KEEP_RATE:.2f} -- walking the best one anyway")
        kept = scored[:1]
    return [g for *_, g in kept]


def _ranges(dots):
    """Dot indices as compact runs for the log, e.g. "0-6, 11-12"."""
    runs, dots = [], sorted(dots)
    for d in dots:
        if runs and d == runs[-1][1] + 1:
            runs[-1][1] = d
        else:
            runs.append([d, d])
    return ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in runs)


def _patch_dots(pieces, n_points, adjacency):
    """The dots the next date should walk: every dot not in a good piece
    (GOOD_PIECE_DOTS or longer, from any date so far), plus
    PATCH_OVERLAP_DOTS of each good piece bordering them. Before any date
    has run that is every dot. Empty means nothing is left to patch."""
    good_len = min(GOOD_PIECE_DOTS, n_points)
    good = set()
    for p in pieces:
        if len(p[5]) >= good_len:  # p[5] == dots
            good |= p[5]
    patch = set(range(n_points)) - good
    frontier = set(patch)
    for _ in range(PATCH_OVERLAP_DOTS if patch else 0):
        frontier = {nb for d in frontier for nb in adjacency.get(d, [])} - patch
        patch |= frontier
    return patch


def run_pathfind_reconstruction(
    date_graphs: list[dict],
    points: list[tuple[float, float]],
    adjacency: dict[int, list[int]],
    start_lat: float,
    start_lon: float,
    test_edge,
    rate_pano,
    point_cover_tolerance_m: float = 15.0,
    max_time_budget_s: float = 220.0,
    protected_positions: set = None,
) -> list[tuple]:
    """Two-phase pathfind.

    - Phase 1 (map_date): per date graph, walk dot-by-dot over the shared
      corridor adjacency (see build_street_graph/fetch_nodes.py --
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
      guaranteed non-empty by the per-dot rating above). A date is left
      early after MAX_FAILED_DOTS_IN_A_ROW dots in a row fail to link.

      Dates are first sampled and reordered (see _sample_dates), then
      walked as patches: the best date walks the whole corridor, and
      each later date walks only what is still weak -- every dot outside
      a piece of GOOD_PIECE_DOTS or more, plus a little overlap (see
      _patch_dots). Earlier pieces are always kept; set_cover picks
      between them and the patch's at the end, so a patch that does
      worse costs nothing. Stops once nothing is weak.

      Bounded by ONE shared wall-clock deadline across ALL dates combined,
      not a per-date call count -- this call runs inside a single
      @spaces.GPU window with a real, fixed wall-clock duration (ZeroGPU
      kills the call outright once it's up, regardless of what's
      mid-flight), so the real constraint was always time, not "how many
      tests." A call-count budget was only ever an approximation of that,
      and a bad one once calls stop being uniform cost (e.g. a future
      solo-pano scoring pass alongside the pairwise tests). The deadline
      is max_time_budget_s -- the walk's share of the caller's own GPU
      window, which is already sized from the dot count (see
      pipeline_runner.estimate_gpu_seconds). Early exit below means an
      easy corridor still finishes well before it.
    - Phase 2 (set_cover): greedy set cover over every piece from every
      date mapped -- picks fewest pieces covering the most corridor.

    Why not search toward goal points directly (earlier v1 design): one
    walk only chases one branch at a time, so N disconnected branches
    force N segments -- even when a single other date's own pieces
    could've covered several branches at once. Not discoverable without
    seeing the whole per-date picture first.

    Inputs (pre-downloaded by caller, no network here):
    - date_graphs: [{"date": str, "dot_candidates": {dot_index: [(key,
      path, lat, lon), ...]}}, ...], ranked by coverage, already
      capped/isolated per date (see build_corridor_graphs) -- reordered
      here by sampled image quality (see _sample_dates).
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
    - rate_pano(path) -> (score, pose, pts, cols). A candidate's
      solo DA3 self-consistency score (higher = more internally coherent,
      correlates with real pairwise success -- see
      tests/debug_solo_score_experiment.py for the real-data validation:
      33% success at score 6 up to 100% at score 13+) PLUS that pano's own
      real solo point cloud (pose/pts/cols, same shape/frame convention as
      test_edge's pose_a/pose_b/pts). When given, a dot's own candidates
      get rated lazily -- only the first time the walk actually reaches
      that dot, never upfront for dots that end up skipped entirely -- the
      best-scored one is tried first for pairwise tests AND becomes that
      dot's guaranteed fallback one-node piece (see ensure_piece). It is
      what makes every node own its own points: a dot always enters the
      result through its own solo cloud or its own slice of a pairwise
      one, never through a joint cloud covering two panoramas at once.
      Was optional, preserving the given candidate
      order -- but then a dot that never pairs with anything is dropped
      instead of falling back to a solo piece (old behavior).

    Segments are NOT stitched together -- each is DA3's own arbitrary
    frame; joining/bridging is entirely the caller's job (see
    reconstruct/join_segments.py's bridge_pieces,
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
    cross-chunk bridge attempt needs. Resolved to the matching dot INDEX
    in `points` (exact match -- every date graph walks the same
    points/adjacency object, so a dot index is a precise, date-
    independent structural identity for "this real location") before
    rescuing, NOT matched by node key or approximate distance -- a node
    key is date-specific (the same real spot gets a different pano id on
    every historical date), so exact-key matching can't rescue a location
    whose winning date differs from whichever date the coordinate itself
    came from (see _rescue_protected_pieces). A protected position with
    zero real candidates anywhere just stays absent -- this only rescues
    a location that WAS reconstructed somewhere but lost the coverage
    competition.

    Returns [(clouds, path_edges, date, reached_all, node_positions,
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
        fails_in_a_row = [0]

        def out_of_time():
            """The shared deadline, or this date given up on (see
            MAX_FAILED_DOTS_IN_A_ROW) -- either way, stop spending on it."""
            return time.monotonic() >= deadline or fails_in_a_row[0] >= MAX_FAILED_DOTS_IN_A_ROW

        def rate_sorted(candidates):
            """Best-solo-score-first ordering of a dot's own candidates,
            rated lazily right here (only for a dot the walk actually
            reached) -- never upfront for the whole corridor. No-op
            (original order) with no rater configured, nothing to
            reorder, or the deadline's already passed (graceful degrade,
            not a wasted call). Doesn't rate a lone candidate itself here
            (nothing to sort) -- ensure_piece rates it on demand instead."""
            if len(candidates) <= 1 or out_of_time():
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
            behavior) if the deadline's passed,
            there's nothing to rate for this dot on this date, or DA3
            produced no pose at all for the best candidate."""
            if dot in confirmed or out_of_time():
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
            piece_data[pid] = {"clouds": {key: (pts, cols)}, "path_edges": []}

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
            if out_of_time():
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
            from_kept, from_total = per_pano_views.get(os.path.basename(from_path), (0, 0))
            edge = (from_key, to_key, [from_kept, from_total], [to_kept, to_total])

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
            pd["clouds"][to_key] = (to_pts @ seg_R.T + seg_t, to_cols)
            pd["path_edges"].append(edge)
            confirmed[to_dot] = {"key": to_key, "path": to_path, "lat": to_lat, "lon": to_lon,
                                  "seg_R": seg_R, "seg_t": seg_t, "pose": pose_b, "piece_id": pid,
                                  "n_views_kept": to_kept, "n_views_total": to_total}

            print(f"[{date}] {from_key} -> {to_key}: OK ({t_test:.2f}s, {deadline - time.monotonic():.1f}s left)")
            return True

        def try_target(from_dot, to_dot, to_candidates):
            """Try to_candidates, best solo-score first, against from_dot's
            established candidate. from_dot always has one by now --
            ensure_piece runs on every dot the walk reaches. First success
            wins."""
            if from_dot not in confirmed:
                return False
            c = confirmed[from_dot]
            tests_before = tests_used[0]
            for key, path, lat, lon in rate_sorted(to_candidates):
                if test_and_confirm(from_dot, c["key"], c["path"], c["lat"], c["lon"], to_dot, key, path, lat, lon):
                    fails_in_a_row[0] = 0
                    return True
            if tests_used[0] > tests_before:  # really tried, not just out of time
                fails_in_a_row[0] += 1
                if fails_in_a_row[0] == MAX_FAILED_DOTS_IN_A_ROW:
                    print(f"[{date}] {MAX_FAILED_DOTS_IN_A_ROW} dots in a row failed to link -- leaving this date")
            return False

        queue = deque()

        def visit(dot):
            """Give `dot` its one chance to reach every structural
            neighbor out of it. `dot` and every candidate dot looked at
            below get ensure_piece'd first, as a best-effort fallback
            piece for each. No flood-past-empty-dot fallback: a dot
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

        while not out_of_time():
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
            # dots (the raw dot-index set) tags along as the LAST field --
            # internal-only, structural identity shared bit-for-bit across
            # every date graph (same points/adjacency object), used by
            # _rescue_protected_pieces for exact dot matching instead of
            # approximate real-distance matching against a node's own
            # (differently-sourced) reported lat/lon.
            pieces.append((pd["clouds"], pd["path_edges"], node_positions, covered_points(dots), frame_poses, set(dots)))
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
            pool.sort(key=lambda p: len(p[3] & uncovered), reverse=True)
            top = pool[0]
            if not (top[3] & uncovered):
                break
            chosen.append(top)
            uncovered -= top[3]
            pool.pop(0)
        return chosen, uncovered

    all_pieces = []  # (pts, cols, path_edges, node_positions, covered, frame_poses, dots, date)
    total_tests = 0
    time_budget_s = max_time_budget_s
    deadline = time.monotonic() + time_budget_s
    print(f"pathfind: time budget {time_budget_s:.0f}s for {len(points)} dot(s)")

    rated_cache = {}  # pano key -> rate_pano's result, shared by sampling and every date's walk

    def rate_one(candidate):
        key, path, lat, lon = candidate
        if key not in rated_cache:
            rated_cache[key] = rate_pano(path)
        return rated_cache[key]

    t_sample = time.monotonic()
    ordered = _sample_dates(date_graphs, len(points), rate_one)
    print(f"timing: date sampling {time.monotonic() - t_sample:.1f}s, {len(rated_cache)} pano(s) rated")

    for date_graph in ordered:
        if time.monotonic() >= deadline:
            print("pathfind: time budget exhausted -- stopping date exploration")
            break
        patch = _patch_dots(all_pieces, len(points), adjacency)
        if not patch:
            print("pathfind: every dot is in a good piece -- stopping date exploration")
            break

        date = date_graph["date"]
        dot_candidates = {d: c for d, c in date_graph["dot_candidates"].items() if d in patch}
        if not dot_candidates:
            print(f"pathfind: date {date} has no panos where patching is needed -- skipped")
            continue
        print(f"pathfind: date {date} walking {len(dot_candidates)} dot(s) "
              f"({'whole corridor' if not all_pieces else 'patch: dots ' + _ranges(dot_candidates)})")
        pieces, tests_used = map_date(date, dot_candidates, total_tests, deadline)
        total_tests += tests_used
        for p in pieces:
            all_pieces.append(p + (date,))
        print(f"pathfind: date {date} mapped into {len(pieces)} piece(s) "
              f"{sorted(len(p[5]) for p in pieces)[::-1]} dot(s), {total_tests} attempts so far")

    protected_indices = None
    if protected_positions:
        pos_to_idx = {pt: i for i, pt in enumerate(points)}
        protected_indices = {pos_to_idx[pos] for pos in protected_positions if pos in pos_to_idx}
        not_in_graph = set(protected_positions) - set(pos_to_idx)
        if not_in_graph:
            print(f"pathfind: {len(not_in_graph)} protected position(s) aren't a dot in this corridor at all: {sorted(not_in_graph)}")

    chosen, leftover_uncovered = set_cover(all_pieces, len(points))
    chosen, leftover_uncovered = _rescue_protected_pieces(chosen, all_pieces, leftover_uncovered, protected_indices)

    reached_all = not leftover_uncovered
    segments = [
        (clouds, path_edges, date, reached_all, node_positions, frame_poses)
        for clouds, path_edges, node_positions, covered, frame_poses, dots, date in chosen
    ]

    print(f"pathfind: {total_tests} attempts total, {len(date_graphs)} date(s) considered, {len(all_pieces)} piece(s) found, {len(segments)} segment(s) chosen, corridor {'fully' if reached_all else 'partially'} covered ({len(leftover_uncovered)}/{len(points)} point(s) never covered)")

    # Diagnostic for whether set_cover's cross-date greedy pick is
    # actually pulling its weight, or just have the OPTION to but never
    # using it -- "N date(s) considered" above only says how many got
    # walked, not whether the CHOSEN combination actually crossed dates.
    dates_used = sorted({s[2] for s in segments})
    if len(dates_used) > 1:
        print(f"pathfind: set_cover MIXED {len(dates_used)} different dates across the chosen segments: {dates_used}")
    elif len(date_graphs) > 1:
        print(f"pathfind: {len(date_graphs)} date(s) were available but every chosen segment came from a single date "
              f"({dates_used[0] if dates_used else 'n/a'}) -- cross-date mixing wasn't needed here")

    return segments
