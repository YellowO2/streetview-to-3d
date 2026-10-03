"""Our decisions about DA3 results: thresholds, view settings, what a passing edge
test or a solo rating is. The only caller of panoramic_da3.run_da3."""
import os
from contextlib import contextmanager

import numpy as np

KEEP_RATE_THRESHOLD = 0.6

# Yaw step between DA3 views: 30 gives 12 views, so a view count means the same everywhere.
VIEW_STEP_DEGREES = 30

# How much of each view's least confident pixels DA3 drops; the UI's hidden "Keep %" overrides it.
CONF_LOWER_PERCENTILE = 25.0   # keep top 75%

# Width of each view in degrees: wider kept more points but looked worse.
VIEW_HFOV = 90.0

# The lowest DA3 confidence a pixel needs to become a point, on top of CONF_LOWER_PERCENTILE.
CONF_FLOOR = 1.05

# Leave cars, people and poles out of every point cloud (see services.segment).
MASK_MOVERS = True

# Per-run overrides of VIEW_HFOV, the masker and CONF_FLOOR (see options()).
_options = {"hfov": None, "masker": None, "mask_classes": None, "conf_floor": None}


@contextmanager
def options(hfov=None, masker=None, mask_classes=None, conf_floor=None):
    """Every DA3 run inside uses this view width, masker repo, classes to drop and
    confidence floor; None keeps the defaults."""
    old = dict(_options)
    _options.update(hfov=hfov, masker=masker, mask_classes=mask_classes, conf_floor=conf_floor)
    try:
        yield
    finally:
        _options.update(old)


def run_da3(target, support, *args, **kwargs):
    """panoramic_da3.run_da3 with this pipeline's view width, mask and confidence
    floor. Also used by the HF app."""
    from panoramic_da3 import run_da3 as run
    from panoramic_da3.components.SplatProcessor import utils
    hfov = _options["hfov"] or VIEW_HFOV
    floor = utils.CONF_ABS_FLOOR
    utils.CONF_ABS_FLOOR = CONF_FLOOR if _options["conf_floor"] is None else _options["conf_floor"]
    try:
        return run(target, support, *args, hfov=hfov, drop_mask=_drop_mask([target, *support], hfov),
                   **kwargs)
    finally:
        utils.CONF_ABS_FLOOR = floor


def _drop_mask(panos, hfov):
    if not MASK_MOVERS:
        return None
    from functools import partial
    from streetview_to_3d.services.segment import drop_in_views
    return partial(drop_in_views, panos=panos, hfov=hfov, model_id=_options["masker"], classes=_options["mask_classes"])


def test_edge(path_a, path_b, cfg, views_base, da3, test_id=0, dist_thresh=0.2, angle_thresh=1,
              step_degrees=VIEW_STEP_DEGREES, keep_rate_threshold=KEEP_RATE_THRESHOLD,
              conf_lower_percentile=CONF_LOWER_PERCENTILE):
    """One pairwise DA3 run on two downloaded panos: None if either keeps too few views,
    else (pose_a, pose_b, pts, cols, per_pano_pts, per_pano_cols, per_pano_views)."""
    test_dir = os.path.join(views_base, f"t{test_id}")
    os.makedirs(test_dir, exist_ok=True)
    id_a, id_b = os.path.basename(path_a), os.path.basename(path_b)
    _, res, pts, cols, per_pano_pts, per_pano_cols = run_da3(
        path_a, [path_b], cfg, test_dir,
        da3=da3, dist_thresh=dist_thresh, angle_thresh=angle_thresh, step_degrees=step_degrees,
        conf_lower_percentile=conf_lower_percentile,
    )
    ka, ta = res.pano_keep_counts.get(id_a, (0, 1))
    kb, tb = res.pano_keep_counts.get(id_b, (0, 1))
    if (ka / ta) < keep_rate_threshold or (kb / tb) < keep_rate_threshold:
        return None
    pose_a = (res.pano_poses[id_a]["center"], res.pano_poses[id_a]["rotation"])
    pose_b = (res.pano_poses[id_b]["center"], res.pano_poses[id_b]["rotation"])
    per_pano_views = {id_a: (ka, ta), id_b: (kb, tb)}
    return pose_a, pose_b, pts, cols, per_pano_pts, per_pano_cols, per_pano_views


def rate_pano(path, cfg, views_base, da3, rate_id=0, dist_thresh=0.2, angle_thresh=1, step_degrees=VIEW_STEP_DEGREES,
              conf_lower_percentile=CONF_LOWER_PERCENTILE):
    """DA3 on one pano alone: (score, pose, pts, cols, n_kept, n_total).

    score is how many of its views passed DA3's consensus filter (it predicts pairing
    success); pose is (center, rotation), or None if DA3 gave it none.
    """
    rate_dir = os.path.join(views_base, f"r{rate_id}")
    os.makedirs(rate_dir, exist_ok=True)
    pano_id = os.path.basename(path)
    filtered_views, res, _, _, per_pano_pts, per_pano_cols = run_da3(
        path, [], cfg, rate_dir, da3=da3, dist_thresh=dist_thresh, angle_thresh=angle_thresh, step_degrees=step_degrees,
        conf_lower_percentile=conf_lower_percentile,
    )
    score = len(filtered_views)
    n_kept, n_total = res.pano_keep_counts.get(pano_id, (score, score))
    if pano_id not in res.pano_poses:
        return score, None, np.zeros((0, 3)), np.zeros((0, 3)), n_kept, n_total
    pose = (res.pano_poses[pano_id]["center"], res.pano_poses[pano_id]["rotation"])
    pts = per_pano_pts.get(pano_id, np.zeros((0, 3)))
    cols = per_pano_cols.get(pano_id, np.zeros((0, 3)))
    return score, pose, pts, cols, n_kept, n_total
