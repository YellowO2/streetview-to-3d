"""Our own domain-specific decisions about DA3 results: what counts as a
passing edge test, a usable solo rating, or a good bridge candidate. Lives
here rather than in panoramic_da3 on purpose -- these are OUR
pipeline's own thresholds/shape choices (keep-rate cutoffs, what "rating"
or "bridging" means for our corridor search), not something a general
"run DA3 on a list of panos" library should know about. panoramic_da3
exposes exactly one primitive (run_da3); this module is the only place
that calls it and interprets the raw result.
"""
import os

import numpy as np

KEEP_RATE_THRESHOLD = 0.6


def test_edge(path_a, path_b, cfg, views_base, da3, test_id=0, dist_thresh=0.2, angle_thresh=1,
              step_degrees=20, keep_rate_threshold=KEEP_RATE_THRESHOLD):
    """One real pairwise DA3 test between two already-downloaded panos.
    Returns None if either pano fails the keep-rate health check, else
    (pose_a, pose_b, pts, cols, per_pano_pts, per_pano_cols, per_pano_views)."""
    from panoramic_da3 import run_da3
    test_dir = os.path.join(views_base, f"t{test_id}")
    os.makedirs(test_dir, exist_ok=True)
    id_a, id_b = os.path.basename(path_a), os.path.basename(path_b)
    _, res, pts, cols, per_pano_pts, per_pano_cols = run_da3(
        path_a, [path_b], cfg, test_dir,
        da3=da3, dist_thresh=dist_thresh, angle_thresh=angle_thresh, step_degrees=step_degrees,
    )
    ka, ta = res.pano_keep_counts.get(id_a, (0, 1))
    kb, tb = res.pano_keep_counts.get(id_b, (0, 1))
    if (ka / ta) < keep_rate_threshold or (kb / tb) < keep_rate_threshold:
        return None
    pose_a = (res.pano_poses[id_a]["center"], res.pano_poses[id_a]["rotation"])
    pose_b = (res.pano_poses[id_b]["center"], res.pano_poses[id_b]["rotation"])
    per_pano_views = {id_a: (ka, ta), id_b: (kb, tb)}
    return pose_a, pose_b, pts, cols, per_pano_pts, per_pano_cols, per_pano_views


def rate_pano(path, cfg, views_base, da3, rate_id=0, dist_thresh=0.2, angle_thresh=1, step_degrees=20):
    """Run DA3 on this pano ALONE (no partner) to get a solo consistency
    score and a real solo point cloud -- so a dot that never pairs with
    any real neighbor can still contribute its own solo reconstruction
    instead of nothing (see walk_graph.py's ensure_piece).

    Returns (score, pose, pts, cols, n_kept, n_total):
      - score: how many of this pano's own views survived DA3's
        consensus filter. Validated against real data (see
        tests/debug_solo_score_experiment.py): pairwise success rate
        rose monotonically with the weaker candidate's score, 33% at
        score 6 up to 100% at score 13+.
      - pose: (center, rotation), or None if DA3 produced no pose at
        all for this pano (rare).
      - pts, cols: this pano's own backprojected points/colors.
      - n_kept, n_total: view counts surviving DA3's filter."""
    from panoramic_da3 import run_da3
    rate_dir = os.path.join(views_base, f"r{rate_id}")
    os.makedirs(rate_dir, exist_ok=True)
    pano_id = os.path.basename(path)
    filtered_views, res, _, _, per_pano_pts, per_pano_cols = run_da3(
        path, [], cfg, rate_dir, da3=da3, dist_thresh=dist_thresh, angle_thresh=angle_thresh, step_degrees=step_degrees,
    )
    score = len(filtered_views)
    n_kept, n_total = res.pano_keep_counts.get(pano_id, (score, score))
    if pano_id not in res.pano_poses:
        return score, None, np.zeros((0, 3)), np.zeros((0, 3)), n_kept, n_total
    pose = (res.pano_poses[pano_id]["center"], res.pano_poses[pano_id]["rotation"])
    pts = per_pano_pts.get(pano_id, np.zeros((0, 3)))
    cols = per_pano_cols.get(pano_id, np.zeros((0, 3)))
    return score, pose, pts, cols, n_kept, n_total


def bridge_test_edge(path_a, path_b, cfg, views_base, da3, test_id=0, dist_thresh=0.2, angle_thresh=1, step_degrees=20):
    """Diagnostic variant of test_edge for the bridging search (joining two
    already-built pieces -- a real DA3 estimate, even a poor one, is
    trusted over independent GPS placement). Never gates pass/fail itself
    -- the caller (join_segments.py's _try_bridge) ranks several attempts
    using the raw keep-rate/deviation data returned here and always uses
    the best one found, however weak.

    Returns None only if a pano has no pose at all (extremely rare --
    DA3Model always provides a fallback pose regardless of keep-rate).
    Else a dict: pose_a/pose_b, pts, cols, keep_a/keep_b ((kept, total)
    view counts), avg_dev_a/avg_dev_b (average real-world deviation in
    meters among that pano's own kept views only; inf if zero kept)."""
    from panoramic_da3 import run_da3
    test_dir = os.path.join(views_base, f"b{test_id}")
    os.makedirs(test_dir, exist_ok=True)
    id_a, id_b = os.path.basename(path_a), os.path.basename(path_b)
    _, res, pts, cols, _, _ = run_da3(
        path_a, [path_b], cfg, test_dir,
        da3=da3, dist_thresh=dist_thresh, angle_thresh=angle_thresh, step_degrees=step_degrees,
    )
    if id_a not in res.pano_poses or id_b not in res.pano_poses:
        return None
    ka, ta = res.pano_keep_counts.get(id_a, (0, 1))
    kb, tb = res.pano_keep_counts.get(id_b, (0, 1))
    pose_a = (res.pano_poses[id_a]["center"], res.pano_poses[id_a]["rotation"])
    pose_b = (res.pano_poses[id_b]["center"], res.pano_poses[id_b]["rotation"])
    return {
        "pose_a": pose_a, "pose_b": pose_b,
        "pts": pts if pts is not None else np.zeros((0, 3)),
        "cols": cols if cols is not None else np.zeros((0, 3)),
        "keep_a": (ka, ta), "keep_b": (kb, tb),
        "avg_dev_a": res.pano_avg_deviation.get(id_a, float("inf")),
        "avg_dev_b": res.pano_avg_deviation.get(id_b, float("inf")),
    }
