"""Run DA3, the depth model, and judge what it gives back.

DA3, the depth model: which one and where it is (DA3Config), and our decisions about its
results: thresholds, view settings, what a passing edge test or a solo rating is. The only
caller of panoramic_da3.panoramic_da3."""
import os
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from huggingface_hub import snapshot_download

# Metres per DA3 unit when a scene can't fit its own (postprocess/place.piece_scale);
# per-scene fits range about 1.17-1.62.
DA3_UNITS_TO_METRES = 1.3

# Nested 1.1: better per-pano clouds than the original Nested; the non-nested model's
# pieces each came out a different scale.
DA3_MODEL_REPO = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


@dataclass
class DA3Config:
    """Which DA3 to load: the model's local path."""
    da3_model: str = ""


def load_da3_config(repo: str = DA3_MODEL_REPO) -> DA3Config:
    return DA3Config(da3_model=snapshot_download(repo_id=repo))


KEEP_RATE_THRESHOLD = 0.6

# Yaw step between DA3 views: 30 gives 12 views, so a view count means the same everywhere.
VIEW_STEP_DEGREES = 30

# How much of each view's least confident pixels DA3 drops; the UI's hidden "Keep %" overrides it.
CONF_LOWER_PERCENTILE = 25.0   # keep top 75%

# Width of each view in degrees: wider kept more points but looked worse.
VIEW_HFOV = 90.0

# The lowest DA3 confidence a pixel needs to become a point, on top of CONF_LOWER_PERCENTILE.
CONF_FLOOR = 1.05

# Leave cars, people and poles out of every point cloud (see models.segment).
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


def run_da3(paths, cfg, da3, **settings):
    """panoramic_da3 on the panos at `paths`, with this pipeline's view width, mask and
    confidence floor: (point_clouds, da3_result), each keyed by the pano's file name.
    `settings` are panoramic_da3.Settings fields. Also used by the HF app."""
    from panoramic_da3 import Loader, Settings, panoramic_da3
    settings = Settings(
        da3_model=cfg.da3_model,
        hfov=_options["hfov"] or VIEW_HFOV,
        conf_floor=CONF_FLOOR if _options["conf_floor"] is None else _options["conf_floor"],
        drop_mask=_drop_mask(paths),
        **settings,
    )
    return panoramic_da3(Loader(paths), settings, da3=da3)


def _drop_mask(paths):
    if not MASK_MOVERS:
        return None
    from functools import partial
    from streetview_to_3d.models.segment import drop_in_views
    panos = {os.path.basename(path): path for path in paths}
    return partial(drop_in_views, panos=panos, model_id=_options["masker"], classes=_options["mask_classes"])


def test_edge(path_a, path_b, cfg, da3, dist_thresh=0.2, angle_thresh=1,
              step_degrees=VIEW_STEP_DEGREES, keep_rate_threshold=KEEP_RATE_THRESHOLD,
              conf_lower_percentile=CONF_LOWER_PERCENTILE):
    """One pairwise DA3 run on two downloaded panos: None if either keeps too few views,
    else (pose_a, pose_b, pts, cols, per_pano_pts, per_pano_cols, per_pano_views)."""
    from panoramic_da3 import PointCloud
    id_a, id_b = os.path.basename(path_a), os.path.basename(path_b)
    point_clouds, res = run_da3(
        [path_a, path_b], cfg, da3, dist_thresh=dist_thresh, angle_thresh=angle_thresh,
        step_degrees=step_degrees, conf_lower_percentile=conf_lower_percentile,
    )
    merged = PointCloud.merge(point_clouds.values())
    pts, cols = (merged.points, merged.colours) if point_clouds else (None, None)
    per_pano_pts = {pano: cloud.points for pano, cloud in point_clouds.items()}
    per_pano_cols = {pano: cloud.colours for pano, cloud in point_clouds.items()}
    ka, ta = res.pano_keep_counts.get(id_a, (0, 1))
    kb, tb = res.pano_keep_counts.get(id_b, (0, 1))
    if (ka / ta) < keep_rate_threshold or (kb / tb) < keep_rate_threshold:
        return None
    pose_a = (res.pano_poses[id_a]["center"], res.pano_poses[id_a]["rotation"])
    pose_b = (res.pano_poses[id_b]["center"], res.pano_poses[id_b]["rotation"])
    per_pano_views = {id_a: (ka, ta), id_b: (kb, tb)}
    return pose_a, pose_b, pts, cols, per_pano_pts, per_pano_cols, per_pano_views


def rate_pano(path, cfg, da3, dist_thresh=0.2, angle_thresh=1, step_degrees=VIEW_STEP_DEGREES,
              conf_lower_percentile=CONF_LOWER_PERCENTILE):
    """DA3 on one pano alone: (score, pose, pts, cols, n_kept, n_total).

    score is how many of its views passed DA3's consensus filter (it predicts pairing
    success); pose is (center, rotation), or None if DA3 gave it none.
    """
    pano_id = os.path.basename(path)
    point_clouds, res = run_da3(
        [path], cfg, da3, dist_thresh=dist_thresh, angle_thresh=angle_thresh, step_degrees=step_degrees,
        conf_lower_percentile=conf_lower_percentile,
    )
    score = len(res.views)
    n_kept, n_total = res.pano_keep_counts.get(pano_id, (score, score))
    if pano_id not in res.pano_poses:
        return score, None, np.zeros((0, 3)), np.zeros((0, 3)), n_kept, n_total
    pose = (res.pano_poses[pano_id]["center"], res.pano_poses[pano_id]["rotation"])
    cloud = point_clouds.get(pano_id)
    pts = cloud.points if cloud else np.zeros((0, 3))
    cols = cloud.colours if cloud else np.zeros((0, 3))
    return score, pose, pts, cols, n_kept, n_total
