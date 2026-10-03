"""The walk: from N date graphs, the most complete corridor in the fewest DA3 pieces.

No GPU or downloads here: it calls test_edge/rate_pano callbacks, and runs inside the
caller's one @spaces.GPU call (runner.py) because each next test depends on the last.
"""
import os
import time
from collections import deque

import numpy as np

from streetview_to_3d.common.geo import haversine_m


def rigid_align(shared_from: list[tuple[np.ndarray, np.ndarray]], shared_to: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Average rigid transform (R, t) mapping the 'from' frame onto the 'to' frame, from
    shared anchor poses (center, world-to-pano rotation) in both: R = r_to.T @ r_from."""
    from scipy.spatial.transform import Rotation

    Rs, ts = [], []
    for (c_from, r_from), (c_to, r_to) in zip(shared_from, shared_to):
        R = r_to.T @ r_from
        Rs.append(R)
        ts.append(c_to - R @ c_from)
    quats = np.array([Rotation.from_matrix(R).as_quat() for R in Rs])
    quats *= np.sign(quats @ quats[0])[:, None]
    return Rotation.from_quat(quats.mean(axis=0)).as_matrix(), np.mean(ts, axis=0)


# A piece this many dots long is trusted; every dot outside one is re-walked by the next date.
GOOD_PIECE_DOTS = 4
# Dots a patch reaches into the good piece either side: shared road for placement to align by.
PATCH_OVERLAP_DOTS = 1
# Panos rated per date up front, to order dates by how well DA3 handles their imagery.
DATE_SAMPLES_MAX = 5


def _sample_dots(dots, k):
    """k of `dots`, evenly spread along the corridor's dot order."""
    dots = sorted(dots)
    if k >= len(dots):
        return dots
    return [dots[round(i * (len(dots) - 1) / (k - 1))] for i in range(k)] if k > 1 else [dots[len(dots) // 2]]


def _sample_dates(date_graphs, n_points, rate, out_of_time=lambda: False):
    """Date graphs in walk order: median solo keep-rate of a few sampled panos (half its
    dots, 1 to DATE_SAMPLES_MAX) times the share of the corridor it covers.

    None is dropped: a low date is only walked where earlier ones left the corridor weak,
    and is often the only one there. Past the deadline nothing more is rated."""
    scored = []
    for g in date_graphs:
        dots = list(g["dot_candidates"])
        if not dots:
            continue
        k = min(DATE_SAMPLES_MAX, max(1, -(-len(dots) // 2)))
        rates = []
        for d in _sample_dots(dots, k):
            if out_of_time():
                break
            n_kept, n_total = rate(g["dot_candidates"][d][0])[4:6]
            rates.append(n_kept / n_total if n_total else 0.0)
        median = float(np.median(rates)) if rates else 0.0
        coverage = len(dots) / n_points
        scored.append((median * coverage, median, coverage, g))
        print(f"pathfind: date {g['date']} sampled {len(rates)} pano(s): median keep "
              f"{median:.2f}, covers {coverage:.0%} of the corridor")
    scored.sort(key=lambda t: t[0], reverse=True)
    return [g for *_, g in scored]


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
    """The dots the next date should walk: every dot not in a good piece (GOOD_PIECE_DOTS
    or longer), plus PATCH_OVERLAP_DOTS into the good pieces bordering them."""
    good_len = min(GOOD_PIECE_DOTS, n_points)
    good = set()
    for p in pieces:
        if len(p[4]) >= good_len:  # p[4]: dots
            good |= p[4]
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
) -> list[tuple]:
    """Two phases: map every date into pieces, then pick the fewest covering the most.

    Phase 1 (map_date), per date: walk dot to dot over the shared adjacency. The first
    time a dot is reached, its candidates are rated (rate_pano) and the best one's solo
    cloud becomes a one-node piece, so every dot touched ends up in the output. Each
    neighbour then gets one pair test (test_edge) against it; a pass merges the neighbour
    in with its own slice of the joint run. An empty or failed neighbour is a dead end;
    when the queue drains, a new piece starts near what is still uncovered. Dates are
    walked best first (_sample_dates), each later one only where pieces are still weak
    (_patch_dots), all under one deadline: the GPU window is wall-clock.

    Phase 2 (set_cover): greedy set cover over every piece from every date.

    date_graphs: [{"date", "dot_candidates": {dot: [(key, path, lat, lon)]}}], downloaded.
    test_edge(path_a, path_b, test_id) -> models.da3.test_edge's tuple, or None.
    rate_pano(path) -> models.da3.rate_pano's tuple.

    Returns [(clouds, path_edges, date, frame_poses)], one per chosen piece, each in its
    own DA3 frame. frame_poses: {key: (center, rotation, path, lat, lon, n_views_kept,
    n_views_total)} in that frame.
    """
    if not date_graphs or not points:
        return []

    def pdist(lat, lon, pi):
        return haversine_m(lat, lon, points[pi][0], points[pi][1])

    def map_date(date, dot_candidates, test_offset, deadline):
        """Phase 1 for one date. Returns (pieces, tests_used); a piece is
        (clouds, path_edges, covered, frame_poses, dots). deadline is shared by all dates."""
        confirmed = {}  # dot -> its pano, pose in its piece's frame (seg_R, seg_t) and piece_id
        piece_data = {}  # piece_id -> {clouds, path_edges}
        next_piece_id = [0]
        visited = set()  # dots already given their one chance
        tests_used = [0]

        def out_of_time():
            return time.monotonic() >= deadline

        def rate_sorted(candidates):
            """A dot's candidates, best solo score first, rated lazily (as given if
            only one, or past the deadline)."""
            if len(candidates) <= 1 or out_of_time():
                return candidates
            scored = [(rate_one(c)[0], c) for c in candidates]
            scored.sort(key=lambda sc: sc[0], reverse=True)
            return [c for _, c in scored]

        def ensure_piece(dot):
            """The first time `dot` is looked at, its best-rated candidate's solo cloud
            becomes its one-node piece (skipped past the deadline or with no pose)."""
            if dot in confirmed or out_of_time():
                return
            candidates = rate_sorted(dot_candidates.get(dot, []))
            if not candidates:
                return
            key, path, lat, lon = candidates[0]
            score, pose, pts, cols, n_kept, n_total = rate_one((key, path, lat, lon))
            if pose is None:
                return
            pid = next_piece_id[0]
            next_piece_id[0] += 1
            confirmed[dot] = {"key": key, "path": path, "lat": lat, "lon": lon,
                               "seg_R": np.eye(3), "seg_t": np.zeros(3), "pose": pose, "piece_id": pid,
                               "n_views_kept": n_kept, "n_views_total": n_total, "solo": True}
            piece_data[pid] = {"clouds": {key: (pts, cols)}, "path_edges": []}

        def covered_points(dots):
            """The dots themselves, plus any point with at least 2 of them within
            point_cover_tolerance_m (a gap flanked on both sides)."""
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

        def test_and_confirm(from_dot, from_key, from_path, to_dot, to_key, to_path, to_lat, to_lon):
            """One DA3 pair test. On success to_dot's own piece is dropped and its slice
            of the joint run joins from_dot's piece (rigid_align); from_dot takes its
            slice too while it still holds only its solo cloud, never twice."""
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
            to_kept, to_total = per_pano_views.get(to_id, (0, 0))
            from_kept, from_total = per_pano_views.get(os.path.basename(from_path), (0, 0))
            edge = (from_key, to_key, [from_kept, from_total], [to_kept, to_total])

            pf = confirmed[from_dot]
            pid = pf["piece_id"]
            if to_dot in confirmed and confirmed[to_dot]["piece_id"] == pid:
                print(f"[{date}] {from_key} -> {to_key}: OK (already same piece, {t_test:.2f}s, {deadline - time.monotonic():.1f}s left)")
                return True
            if to_dot in confirmed:
                piece_data.pop(confirmed[to_dot]["piece_id"], None)

            local_R, local_t = rigid_align([pose_a], [pf["pose"]])
            seg_R = pf["seg_R"] @ local_R
            seg_t = pf["seg_R"] @ local_t + pf["seg_t"]
            pd = piece_data[pid]
            def own(pano_id):
                """A pano's slice of this run, (points, colors), in the piece's frame."""
                none = np.zeros((0, 3))
                return per_pano_pts.get(pano_id, none) @ seg_R.T + seg_t, per_pano_cols.get(pano_id, none)

            pd["clouds"][to_key] = own(to_id)
            if pf["solo"]:
                # a link beats from_dot's solo cloud (which may have kept no views)
                pd["clouds"][from_key] = own(os.path.basename(from_path))
                pf.update(solo=False, n_views_kept=from_kept, n_views_total=from_total)
            pd["path_edges"].append(edge)
            confirmed[to_dot] = {"key": to_key, "path": to_path, "lat": to_lat, "lon": to_lon,
                                  "seg_R": seg_R, "seg_t": seg_t, "pose": pose_b, "piece_id": pid,
                                  "n_views_kept": to_kept, "n_views_total": to_total, "solo": False}

            print(f"[{date}] {from_key} -> {to_key}: OK ({t_test:.2f}s, {deadline - time.monotonic():.1f}s left)")
            return True

        def try_target(from_dot, to_dot, to_candidates):
            """Test to_candidates, best first, against from_dot's pano; first success wins."""
            if from_dot not in confirmed:
                return False
            c = confirmed[from_dot]
            for key, path, lat, lon in rate_sorted(to_candidates):
                if test_and_confirm(from_dot, c["key"], c["path"], to_dot, key, path, lat, lon):
                    return True
            return False

        queue = deque()

        def visit(dot):
            """Give `dot` its one chance to reach each untried neighbour; nothing is retried."""
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
            """The untried dot nearest the start (first seed) or the nearest uncovered point."""
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
                continue
            # each node's pose re-expressed in the piece's frame
            frame_poses = {c["key"]: (c["seg_R"] @ c["pose"][0] + c["seg_t"], c["pose"][1] @ c["seg_R"].T,
                                      c["path"], c["lat"], c["lon"], c["n_views_kept"], c["n_views_total"])
                           for c in (confirmed[d] for d in dots)}
            # dots: a date-independent identity for each place
            pieces.append((pd["clouds"], pd["path_edges"], covered_points(dots), frame_poses, set(dots)))
        return pieces, tests_used[0]

    def set_cover(pieces, total_points):
        """Phase 2: repeatedly take the piece covering the most uncovered points; ties go
        to the best-rated panos, which decides a spot no link reached.
        Returns (chosen, leftover_uncovered)."""
        def score(p):
            s = [rated_cache[k][0] for k in p[3] if k in rated_cache]  # p[3]: frame_poses
            return sum(s) / len(s) if s else float("-inf")

        scores = {id(p): score(p) for p in pieces}
        uncovered = set(range(total_points))
        chosen = []
        pool = list(pieces)
        while uncovered and pool:
            pool.sort(key=lambda p: (len(p[2] & uncovered), scores[id(p)]), reverse=True)
            top = pool[0]
            if not (top[2] & uncovered):
                break
            chosen.append(top)
            uncovered -= top[2]
            pool.pop(0)
        return chosen, uncovered

    all_pieces = []  # (clouds, path_edges, covered, frame_poses, dots, date)
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
    ordered = _sample_dates(date_graphs, len(points), rate_one, lambda: time.monotonic() >= deadline)
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
              f"{sorted(len(p[4]) for p in pieces)[::-1]} dot(s), {total_tests} attempts so far")

    chosen, leftover_uncovered = set_cover(all_pieces, len(points))

    reached_all = not leftover_uncovered
    segments = [(clouds, path_edges, date, frame_poses)
                for clouds, path_edges, covered, frame_poses, dots, date in chosen]

    print(f"pathfind: {total_tests} attempts total, {len(date_graphs)} date(s) considered, {len(all_pieces)} piece(s) found, {len(segments)} segment(s) chosen, corridor {'fully' if reached_all else 'partially'} covered ({len(leftover_uncovered)}/{len(points)} point(s) never covered)")

    # whether the chosen pieces actually mix dates
    dates_used = sorted({s[2] for s in segments})
    if len(dates_used) > 1:
        print(f"pathfind: set_cover MIXED {len(dates_used)} different dates across the chosen segments: {dates_used}")
    elif len(date_graphs) > 1:
        print(f"pathfind: {len(date_graphs)} date(s) were available but every chosen segment came from a single date "
              f"({dates_used[0] if dates_used else 'n/a'}) -- cross-date mixing wasn't needed here")

    return segments
