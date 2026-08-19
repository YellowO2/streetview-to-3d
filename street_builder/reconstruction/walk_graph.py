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
import heapq

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


def run_pathfind_reconstruction(
    date_graphs: list[dict],
    points: list[tuple[float, float]],
    start_lat: float,
    start_lon: float,
    test_edge,
    target_hop_m: float = 10.0,
    hop_weight: float = 1.0,
    start_zone_m: float = 5.0,
    point_cover_tolerance_m: float = 15.0,
    max_tests_per_date: int = 50,
    early_exit_segments: int = 4,
) -> list[tuple]:
    """Two-phase pathfind.

    - Phase 1 (map_date): per date graph, best-first walk its OWN already-
      isolated graph (see street_builder/build_graph/build_graph.py --
      each graph only ever contains that one date's own nodes/edges, so
      there's no cross-date filtering left to do here) toward uncovered
      corridor points. On dead end, restart from that date's own closest-
      confirmed node across ALL its pieces so far. Produces N disconnected
      pieces per date. Early-exits date exploration once pieces so far
      already need < early_exit_segments to fully cover the corridor.
    - Phase 2 (set_cover): greedy set cover over every piece from every
      date mapped -- picks fewest pieces covering the most corridor.

    Why not search toward goal points directly (earlier v1 design): one
    walk only chases one branch at a time, so N disconnected branches
    force N segments -- even when a single other date's own pieces
    could've covered several branches at once. Not discoverable without
    seeing the whole per-date picture first.

    Inputs (pre-downloaded by caller, no network here):
    - date_graphs: [{"date": str, "nodes": [(key, path, lat, lon), ...],
      "edges": {key: [(other_key, dist_m), ...]}}, ...], ranked best
      first, already capped/isolated per date (see build_corridor_graphs)
      -- this function tries them in the given order and stops once
      early_exit_segments is satisfied, it doesn't re-rank them.
    - points: corridor's traced spine -- shared, date-independent
      coverage reference (dates never share real panos).
    - test_edge(path_a, path_b, test_id) -> (pose_a, pose_b, pts, cols)
      or None on failure. The only GPU-touching thing this function calls.

    Segments are NOT stitched together -- each is DA3's own arbitrary
    frame; joining (GPS + ICP) is the caller's job.

    Returns [(pts, cols, path_edges, date, reached_all, node_positions), ...],
    phase 2's chosen pieces. reached_all: whole corridor covered.
    node_positions: {key: np.ndarray(3,)}, DA3's placement in that
    piece's own frame, for the caller's join step.
    """
    if not date_graphs or not points:
        return []

    def pdist(lat, lon, pi):
        return haversine_m(lat, lon, points[pi][0], points[pi][1])

    def map_date(date, nodes, edges, test_offset, budget):
        """Phase 1 for ONE date's own already-isolated graph. Returns
        (pieces, tests_used); pieces: list of (pts, cols, path_edges,
        node_positions, covered_point_indices)."""
        node_by_key = {key: (path, lat, lon) for key, path, lat, lon in nodes}

        def nearest_uncovered_dist(key, uncovered):
            _, lat, lon = node_by_key[key]
            return min(pdist(lat, lon, pi) for pi in uncovered)

        def score(child_key, hop, uncovered):
            return nearest_uncovered_dist(child_key, uncovered) + hop_weight * abs(hop - target_hop_m)

        def roots_near(lat0, lon0):
            return [key for key, (_, lat, lon) in node_by_key.items()
                    if haversine_m(lat, lon, lat0, lon0) <= start_zone_m]

        def covered_points(keys):
            """Corridor point-indices within point_cover_tolerance_m of any
            of the given confirmed nodes' real positions."""
            covered = set()
            for k in keys:
                _, lat, lon = node_by_key[k]
                for pi in range(len(points)):
                    if pi not in covered and pdist(lat, lon, pi) <= point_cover_tolerance_m:
                        covered.add(pi)
            return covered

        def search_from(root_keys, test_offset, uncovered, dead_edges, budget, confirmed, piece_data, next_piece_id):
            """Best-first walk of this date's graph, scored toward
            uncovered corridor points (mutated in place). Stops on full
            coverage, dead frontier, or budget (the date's REMAINING test
            budget).

            confirmed, dead_edges, piece_data, next_piece_id are ALL owned
            by map_date and persist across every search_from call for this
            date (mutated in place here) -- a restart resumes growing the
            same tree(s) instead of rebuilding them from scratch. Earlier
            versions reset `confirmed` to {} per call, which meant a
            restart near an already-fully-proven chain re-tested every
            edge in it (paying for a real DA3 call each time) before
            reaching the frontier's actual dead end again -- confirmed on
            a real run where the same already-successful edge got
            re-tested 100+ times, each one a wasted GPU call.

            confirmed[key] = {"seg_R", "seg_t", "pose", "piece_id"} --
            piece_id groups nodes sharing one DA3 base frame (a genuine
            disconnection boundary, not a restart boundary). piece_data[id]
            accumulates that piece's own pts/cols/path_edges across
            however many calls contributed to it.

            Returns tests_done (0 if nothing new got tested)."""
            seed_keys = set(root_keys) | set(confirmed.keys())
            frontier = []  # (score, seq, from_key, to_key, hop)
            seq = 0
            for root_key in seed_keys:
                for other_key, hop in edges.get(root_key, []):
                    if other_key in confirmed or frozenset((root_key, other_key)) in dead_edges:
                        continue
                    heapq.heappush(frontier, (score(other_key, hop, uncovered), seq, root_key, other_key, hop))
                    seq += 1

            tests = 0
            while frontier and tests < budget and uncovered:
                _, _, from_key, to_key, hop = heapq.heappop(frontier)
                if to_key in confirmed or frozenset((from_key, to_key)) in dead_edges:
                    continue

                path_a, path_b = node_by_key[from_key][0], node_by_key[to_key][0]
                result = test_edge(path_a, path_b, f"{date}_{test_offset + tests}")
                tests += 1
                if result is None:
                    dead_edges.add(frozenset((from_key, to_key)))
                    print(f"[{date}] {from_key} -> {to_key}: FAIL")
                    continue
                pose_a, pose_b, pts, cols = result

                if from_key not in confirmed:
                    # from_key is a brand-new root: this edge's frame becomes a new piece's base.
                    pid = next_piece_id[0]
                    next_piece_id[0] += 1
                    confirmed[from_key] = {"seg_R": np.eye(3), "seg_t": np.zeros(3), "pose": pose_a, "piece_id": pid}
                    piece_data[pid] = {"pts": pts, "cols": cols, "path_edges": []}
                else:
                    # from_key already has a proven frame (this call or an earlier one) -- reuse it.
                    pf = confirmed[from_key]
                    pid = pf["piece_id"]
                    local_R, local_t = rigid_align([pose_a], [pf["pose"]])
                    seg_R = pf["seg_R"] @ local_R
                    seg_t = pf["seg_R"] @ local_t + pf["seg_t"]
                    pd = piece_data[pid]
                    pd["pts"] = np.concatenate([pd["pts"], pts @ seg_R.T + seg_t], axis=0)
                    pd["cols"] = np.concatenate([pd["cols"], cols], axis=0)
                    confirmed[to_key] = {"seg_R": seg_R, "seg_t": seg_t, "pose": pose_b, "piece_id": pid}
                    piece_data[pid]["path_edges"].append((from_key, to_key))
                    uncovered -= covered_points([to_key])

                if to_key not in confirmed:
                    # from_key was the brand-new-root case above -- finish confirming to_key too.
                    confirmed[to_key] = {"seg_R": np.eye(3), "seg_t": np.zeros(3), "pose": pose_b, "piece_id": pid}
                    piece_data[pid]["path_edges"].append((from_key, to_key))
                    uncovered -= covered_points([from_key, to_key])

                print(f"[{date}] {from_key} -> {to_key}: OK ({len(uncovered)} pt(s) left)")
                if not uncovered:
                    break  # this success just finished coverage -- score() needs a non-empty uncovered
                for other_key, next_hop in edges.get(to_key, []):
                    if other_key in confirmed or frozenset((to_key, other_key)) in dead_edges:
                        continue
                    heapq.heappush(frontier, (score(other_key, next_hop, uncovered), seq, to_key, other_key, next_hop))
                    seq += 1

            return tests

        tests_used = 0
        uncovered = set(range(len(points)))
        dead_edges = set()
        confirmed = {}  # persists across every restart below -- see search_from
        piece_data = {}
        next_piece_id = [0]
        cur_lat, cur_lon = start_lat, start_lon

        while uncovered and tests_used < budget:
            roots = roots_near(cur_lat, cur_lon)
            if not roots:
                break
            tests = search_from(roots, test_offset + tests_used, uncovered, dead_edges, budget - tests_used, confirmed, piece_data, next_piece_id)
            tests_used += tests
            if tests == 0 or not confirmed:
                # tests == 0: frontier genuinely exhausted, nothing left to try.
                # not confirmed: every attempt this date has made so far
                # failed -- nothing to restart from (min() below would
                # crash on an empty confirmed otherwise).
                break

            if not uncovered:
                break
            next_key = min(confirmed, key=lambda k: nearest_uncovered_dist(k, uncovered))
            _, cur_lat, cur_lon = node_by_key[next_key]

        pieces = []
        for pid, pd in piece_data.items():
            keys = [k for k, c in confirmed.items() if c["piece_id"] == pid]
            node_positions = {k: confirmed[k]["seg_R"] @ confirmed[k]["pose"][0] + confirmed[k]["seg_t"] for k in keys}
            pieces.append((pd["pts"], pd["cols"], pd["path_edges"], node_positions, covered_points(keys)))
        return pieces, tests_used

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

    all_pieces = []  # (pts, cols, path_edges, node_positions, covered, date)
    total_tests = 0

    for date_graph in date_graphs:
        date, nodes, edges = date_graph["date"], date_graph["nodes"], date_graph["edges"]
        pieces, tests_used = map_date(date, nodes, edges, total_tests, max_tests_per_date * 5)
        total_tests += tests_used
        for p in pieces:
            all_pieces.append(p + (date,))
        print(f"pathfind: date {date} mapped into {len(pieces)} piece(s), {total_tests} attempts so far")

        chosen_so_far, uncovered_so_far = set_cover(all_pieces, len(points))
        if chosen_so_far and not uncovered_so_far and len(chosen_so_far) < early_exit_segments:
            print(f"pathfind: {len(chosen_so_far)} segment(s) already cover everything -- stopping date exploration")
            break

    chosen, leftover_uncovered = set_cover(all_pieces, len(points))

    reached_all = not leftover_uncovered
    segments = [
        (pts, cols, path_edges, date, reached_all, node_positions)
        for pts, cols, path_edges, node_positions, covered, date in chosen
    ]
    print(f"pathfind: {total_tests} attempts total, {len(date_graphs)} date(s) considered, {len(all_pieces)} piece(s) found, {len(segments)} segment(s) chosen, corridor {'fully' if reached_all else 'partially'} covered ({len(leftover_uncovered)}/{len(points)} point(s) never covered)")
    return segments
