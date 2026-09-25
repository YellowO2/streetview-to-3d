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

# Yaw step for slicing a panorama into DA3 views: 30 gives 12 views. The
# tested middle ground between DA3's own default 20 (18 views) and a too-
# coarse 45 (8 views, which turned 2 of 4 winners from partial acceptance
# to full rejection in an earlier scoring experiment). Every caller uses
# this one value, so a view count means the same thing everywhere.
VIEW_STEP_DEGREES = 30

# Below this share of a pano's views surviving DA3's consensus filter,
# DA3 has not made sense of the pano: a date whose sampled panos sit under
# it is not walked (walk_graph._sample_dates), and a bridge whose best
# attempt sits under it is rejected (join_segments). The solo-score
# experiment (README, Dev notes) saw links mostly fail
# around there and mostly succeed above ~2/3.
MIN_KEEP_RATE = 1.0 / 3

# What fraction of a view's weakest pixels DA3 discards before we ever see
# them (see panoramic_da3's CONF_LOWER_PERCENTILE). Kept as our own default
# rather than panoramic_da3's, so raising it here is a one-line change and
# doesn't require touching that package.
#
# Measured on a real 4-node chunk: 60% kept gave 669,590 points, 80% gave
# 857,247 (+28%), 90% gave 868,398 (+30%) -- CONF_ABS_FLOOR catches most of
# what 90 would additionally let through, so 80 gets nearly all the gain
# for less storage. The UI's "Keep %" slider overrides this per run without
# a redeploy; this is only the fallback when a caller doesn't pass one.
CONF_LOWER_PERCENTILE = 20.0   # keep top 80%


def test_edge(path_a, path_b, cfg, views_base, da3, test_id=0, dist_thresh=0.2, angle_thresh=1,
              step_degrees=VIEW_STEP_DEGREES, keep_rate_threshold=KEEP_RATE_THRESHOLD,
              conf_lower_percentile=CONF_LOWER_PERCENTILE, return_confidence=False):
    """One real pairwise DA3 test between two already-downloaded panos.
    Returns None if either pano fails the keep-rate health check, else
    (pose_a, pose_b, pts, cols, per_pano_pts, per_pano_cols, per_pano_views),
    plus an 8th per_pano_confidence dict when return_confidence is True --
    see panoramic_da3.run_da3's own docstring for what it covers."""
    from panoramic_da3 import run_da3
    test_dir = os.path.join(views_base, f"t{test_id}")
    os.makedirs(test_dir, exist_ok=True)
    id_a, id_b = os.path.basename(path_a), os.path.basename(path_b)
    _, res, pts, cols, per_pano_pts, per_pano_cols = run_da3(
        path_a, [path_b], cfg, test_dir,
        da3=da3, dist_thresh=dist_thresh, angle_thresh=angle_thresh, step_degrees=step_degrees,
        conf_lower_percentile=conf_lower_percentile, return_confidence=return_confidence,
    )
    ka, ta = res.pano_keep_counts.get(id_a, (0, 1))
    kb, tb = res.pano_keep_counts.get(id_b, (0, 1))
    if (ka / ta) < keep_rate_threshold or (kb / tb) < keep_rate_threshold:
        return None
    pose_a = (res.pano_poses[id_a]["center"], res.pano_poses[id_a]["rotation"])
    pose_b = (res.pano_poses[id_b]["center"], res.pano_poses[id_b]["rotation"])
    per_pano_views = {id_a: (ka, ta), id_b: (kb, tb)}
    out = (pose_a, pose_b, pts, cols, per_pano_pts, per_pano_cols, per_pano_views)
    return out + (res.pano_point_confidence,) if return_confidence else out


def rate_pano(path, cfg, views_base, da3, rate_id=0, dist_thresh=0.2, angle_thresh=1, step_degrees=VIEW_STEP_DEGREES,
              conf_lower_percentile=CONF_LOWER_PERCENTILE, return_confidence=False):
    """Run DA3 on this pano ALONE (no partner) to get a solo consistency
    score and a real solo point cloud -- so a dot that never pairs with
    any real neighbor can still contribute its own solo reconstruction
    instead of nothing (see walk_graph.py's ensure_piece).

    Returns (score, pose, pts, cols, n_kept, n_total):
      - score: how many of this pano's own views survived DA3's
        consensus filter. Validated against real data (the solo-score
        experiment, README Dev notes): pairwise success rate
        rose monotonically with the weaker candidate's score, 33% at
        score 6 up to 100% at score 13+.
      - pose: (center, rotation), or None if DA3 produced no pose at
        all for this pano (rare).
      - pts, cols: this pano's own backprojected points/colors.
      - n_kept, n_total: view counts surviving DA3's filter.

    A 7th value, this pano's own per-point confidence array, is appended
    when return_confidence is True -- index-aligned with pts/cols."""
    from panoramic_da3 import run_da3
    rate_dir = os.path.join(views_base, f"r{rate_id}")
    os.makedirs(rate_dir, exist_ok=True)
    pano_id = os.path.basename(path)
    filtered_views, res, _, _, per_pano_pts, per_pano_cols = run_da3(
        path, [], cfg, rate_dir, da3=da3, dist_thresh=dist_thresh, angle_thresh=angle_thresh, step_degrees=step_degrees,
        conf_lower_percentile=conf_lower_percentile, return_confidence=return_confidence,
    )
    score = len(filtered_views)
    n_kept, n_total = res.pano_keep_counts.get(pano_id, (score, score))
    conf = res.pano_point_confidence.get(pano_id, np.zeros((0,), dtype=np.float32))
    if pano_id not in res.pano_poses:
        out = (score, None, np.zeros((0, 3)), np.zeros((0, 3)), n_kept, n_total)
        return out + (conf,) if return_confidence else out
    pose = (res.pano_poses[pano_id]["center"], res.pano_poses[pano_id]["rotation"])
    pts = per_pano_pts.get(pano_id, np.zeros((0, 3)))
    cols = per_pano_cols.get(pano_id, np.zeros((0, 3)))
    out = (score, pose, pts, cols, n_kept, n_total)
    return out + (conf,) if return_confidence else out


def bridge_test_edge(path_a, path_b, cfg, views_base, da3, test_id=0, dist_thresh=0.2, angle_thresh=1, step_degrees=VIEW_STEP_DEGREES,
                     conf_lower_percentile=CONF_LOWER_PERCENTILE, return_confidence=False):
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
    meters among that pano's own kept views only; inf if zero kept).
    conf_a/conf_b (each pano's own per-point confidence array) are added
    when return_confidence is True."""
    from panoramic_da3 import run_da3
    test_dir = os.path.join(views_base, f"b{test_id}")
    os.makedirs(test_dir, exist_ok=True)
    id_a, id_b = os.path.basename(path_a), os.path.basename(path_b)
    _, res, pts, cols, _, _ = run_da3(
        path_a, [path_b], cfg, test_dir,
        da3=da3, dist_thresh=dist_thresh, angle_thresh=angle_thresh, step_degrees=step_degrees,
        conf_lower_percentile=conf_lower_percentile, return_confidence=return_confidence,
    )
    if id_a not in res.pano_poses or id_b not in res.pano_poses:
        return None
    ka, ta = res.pano_keep_counts.get(id_a, (0, 1))
    kb, tb = res.pano_keep_counts.get(id_b, (0, 1))
    pose_a = (res.pano_poses[id_a]["center"], res.pano_poses[id_a]["rotation"])
    pose_b = (res.pano_poses[id_b]["center"], res.pano_poses[id_b]["rotation"])
    out = {
        "pose_a": pose_a, "pose_b": pose_b,
        "pts": pts if pts is not None else np.zeros((0, 3)),
        "cols": cols if cols is not None else np.zeros((0, 3)),
        "keep_a": (ka, ta), "keep_b": (kb, tb),
        "avg_dev_a": res.pano_avg_deviation.get(id_a, float("inf")),
        "avg_dev_b": res.pano_avg_deviation.get(id_b, float("inf")),
    }
    if return_confidence:
        out["conf_a"] = res.pano_point_confidence.get(id_a, np.zeros((0,), dtype=np.float32))
        out["conf_b"] = res.pano_point_confidence.get(id_b, np.zeros((0,), dtype=np.float32))
    return out


def depth_around(target_path, neighbour_paths, cfg, views_base, da3, dist_thresh=0.2, angle_thresh=1,
                 step_degrees=VIEW_STEP_DEGREES, keep_rate_threshold=KEEP_RATE_THRESHOLD,
                 conf_lower_percentile=CONF_LOWER_PERCENTILE):
    """Depth around one panorama: a single joint DA3 run on it and its
    same-date neighbours, for a caller that needs the target's pose and the
    points around it (a splat is scaled against these).

    One joint run, not pairwise tests: there is nothing to chain, and DA3
    reconciles all of them at once. A neighbour that keeps under
    keep_rate_threshold of its views -- the walk's own link bar -- is
    dropped and the run repeated without it. But fewer panos can also make
    DA3 worse on the target itself (on Stockholm it went 6/12 -> 1/12 ->
    0/12 as neighbours were dropped), so every run is kept and the one that
    keeps the most of the target's views wins.

    Returns a dict: points/colors (every kept pano's, in the target's run
    frame), pose (the target's (center, rotation)), views ((kept, total)
    for the target), n_clean (views surviving DA3's filter across the
    run), neighbours (the paths actually used).
    """
    from panoramic_da3 import run_da3
    target_id = os.path.basename(target_path)
    used, best = list(neighbour_paths), None
    for attempt in range(len(used) + 1):
        run_dir = os.path.join(views_base, f"around{attempt}")
        os.makedirs(run_dir, exist_ok=True)
        filtered, res, pts, cols, _, _ = run_da3(
            target_path, used, cfg, run_dir, da3=da3, dist_thresh=dist_thresh,
            angle_thresh=angle_thresh, step_degrees=step_degrees,
            conf_lower_percentile=conf_lower_percentile)
        pose = res.pano_poses.get(target_id)
        run = {
            "points": pts if pts is not None else np.zeros((0, 3)),
            "colors": cols if cols is not None else np.zeros((0, 3)),
            "pose": (pose["center"], pose["rotation"]) if pose else None,
            "views": res.pano_keep_counts.get(target_id, (0, 0)),
            "n_clean": len(filtered),
            "neighbours": list(used),
        }
        print(f"depth_around: {len(used)} neighbour(s): target kept {run['views'][0]}/"
              f"{run['views'][1]}, {run['n_clean']} clean view(s) in all")
        if best is None or (run["views"][0], run["n_clean"]) > (best["views"][0], best["n_clean"]):
            best = run
        rate = {p: k / t if t else 0.0
                for p in used for k, t in [res.pano_keep_counts.get(os.path.basename(p), (0, 1))]}
        bad = [p for p in used if rate[p] < keep_rate_threshold]
        for p in bad:
            print(f"depth_around: dropping neighbour {os.path.basename(p)} "
                  f"(kept {rate[p]:.0%} of its views)")
        if not bad:
            break
        used = [p for p in used if p not in bad]
    return best
