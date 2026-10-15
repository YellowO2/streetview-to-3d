"""Run DA3 over the downloaded panos in one ZeroGPU window."""
import logging
import os

import numpy as np

from streetview_to_3d.models.on_gpu import on_gpu
from streetview_to_3d.reconstruct.WalkSettings import WalkSettings
from streetview_to_3d.reconstruct.estimate_gpu_seconds import MODEL_LOAD_S, SAVE_BUFFER_S, estimate_gpu_seconds

logger = logging.getLogger(__name__)


def _gpu_seconds(points, gpu_seconds=None) -> float:
    """The caller's window, else the estimate for this many dots."""
    return float(gpu_seconds) if gpu_seconds else estimate_gpu_seconds(len(points))


def walk(corridor, settings=WalkSettings()):
    """The walk over the corridor's date graphs (panos.main.panos) in one GPU window (see
    streetview_to_3d.models.on_gpu): [(clouds, metadata), ...], one per piece (reconstruct.pieces).
    Raises RuntimeError if no piece came out."""
    start_lat, start_lon = corridor.start
    pieces = on_gpu(_run_walk_impl, corridor.date_graphs, corridor.points, corridor.adjacency, start_lat, start_lon,
                    settings, seconds=_gpu_seconds(corridor.points, settings.gpu_seconds))
    if not pieces:
        raise RuntimeError("No connected path found from start toward any goal.")
    return pieces


def _run_walk_impl(date_graphs, points, adjacency, start_lat, start_lon, settings):
    """The walk on the downloaded panos, given the window less model load and save headroom.
    Returns [(clouds, metadata), ...], one per piece (reconstruct.pieces)."""
    import time

    import torch
    from streetview_to_3d.models.da3 import CONF_LOWER_PERCENTILE, VIEW_STEP_DEGREES, da3
    from streetview_to_3d.reconstruct.pieces import pieces
    from streetview_to_3d.reconstruct.run_da3 import run_da3
    from streetview_to_3d.reconstruct.walk_graph import walk_graph

    conf_lower_percentile = (CONF_LOWER_PERCENTILE if settings.conf_lower_percentile is None
                             else settings.conf_lower_percentile)
    step_degrees = VIEW_STEP_DEGREES if settings.step_degrees is None else settings.step_degrees

    t0 = time.monotonic()
    total_s = _gpu_seconds(points, settings.gpu_seconds)
    logger.info(f"GPU window: {total_s:.0f}s for {len(points)} dot(s)")

    cfg, model = da3(settings.model)
    # "timing:" lines calibrate the constants above (grep the Space's logs)
    logger.info(f"timing: model load {time.monotonic() - t0:.1f}s")

    def run(paths):
        """DA3 on these panos with this run's settings: run_da3's (point_clouds, da3_result)."""
        return run_da3(paths, cfg, model, masker=settings.masker, mask_classes=settings.mask_classes,
                       hfov=settings.hfov, conf_floor=settings.conf_floor, dist_thresh=0.2, angle_thresh=1,
                       step_degrees=step_degrees, conf_lower_percentile=conf_lower_percentile)

    try:
        segments = walk_graph(
            date_graphs, points, adjacency, start_lat, start_lon,
            lambda path_a, path_b, test_id: _test_edge(run, path_a, path_b),
            rate_pano=lambda path: _rate_pano(run, path),
            max_time_budget_s=max(0.0, total_s - MODEL_LOAD_S - SAVE_BUFFER_S))
        logger.info(f"timing: walk {time.monotonic() - t0:.1f}s for {len(points)} dot(s) "
                    f"-> {len(segments or [])} piece(s), of {total_s:.0f}s")
        return pieces(segments or [])
    finally:
        torch.cuda.empty_cache()


def _test_edge(run, path_a, path_b):
    """One pairwise DA3 run on two downloaded panos: None if either keeps too few views,
    else (pose_a, pose_b, pts, cols, per_pano_pts, per_pano_cols, per_pano_views)."""
    from panoramic_da3 import PointCloud
    from streetview_to_3d.models.da3 import KEEP_RATE_THRESHOLD
    id_a, id_b = os.path.basename(path_a), os.path.basename(path_b)
    point_clouds, res = run([path_a, path_b])
    merged = PointCloud.merge(point_clouds.values())
    pts, cols = (merged.points, merged.colours) if point_clouds else (None, None)
    per_pano_pts = {pano: cloud.points for pano, cloud in point_clouds.items()}
    per_pano_cols = {pano: cloud.colours for pano, cloud in point_clouds.items()}
    ka, ta = res.pano_keep_counts.get(id_a, (0, 1))
    kb, tb = res.pano_keep_counts.get(id_b, (0, 1))
    if (ka / ta) < KEEP_RATE_THRESHOLD or (kb / tb) < KEEP_RATE_THRESHOLD:
        return None
    pose_a = (res.pano_poses[id_a]["center"], res.pano_poses[id_a]["rotation"])
    pose_b = (res.pano_poses[id_b]["center"], res.pano_poses[id_b]["rotation"])
    per_pano_views = {id_a: (ka, ta), id_b: (kb, tb)}
    return pose_a, pose_b, pts, cols, per_pano_pts, per_pano_cols, per_pano_views


def _rate_pano(run, path):
    """DA3 on one pano alone: (score, pose, pts, cols, n_kept, n_total).

    score is how many of its views passed DA3's consensus filter (it predicts pairing
    success); pose is (center, rotation), or None if DA3 gave it none.
    """
    pano_id = os.path.basename(path)
    point_clouds, res = run([path])
    score = len(res.views)
    n_kept, n_total = res.pano_keep_counts.get(pano_id, (score, score))
    if pano_id not in res.pano_poses:
        return score, None, np.zeros((0, 3)), np.zeros((0, 3)), n_kept, n_total
    pose = (res.pano_poses[pano_id]["center"], res.pano_poses[pano_id]["rotation"])
    cloud = point_clouds.get(pano_id)
    pts = cloud.points if cloud else np.zeros((0, 3))
    cols = cloud.colours if cloud else np.zeros((0, 3))
    return score, pose, pts, cols, n_kept, n_total
