"""The street reconstruction's GPU task: the walk, in one window sized
from the dot count (estimate_gpu_seconds). The GPU itself and
the DA3 model come from streetview_to_3d.gpu.
"""
from streetview_to_3d import gpu

# Headroom left after the walk for saving the result before the hard
# ZeroGPU window closes. Carved OUT of the window (see _walk_budget_s).
SAVE_BUFFER_S = 10.0
# DA3 is loaded at startup (see streetview_to_3d.gpu), so a call only waits
# for ZeroGPU to move it onto the GPU: "timing: model load 0.0s" measured.
# Was 45 when it loaded from disk inside the call (17-41 s).
MODEL_LOAD_S = 5.0


# Date sampling, per dot: measured 7 s for 4 dots, 13 s for 7.
DATE_SAMPLING_PER_DOT_S = 2.0


def estimate_gpu_seconds(n_dots: int) -> float:
    """The GPU window a run over n_dots needs: date sampling and the walk's
    own per-dot estimate (walk_graph.SECONDS_PER_DOT_ESTIMATE), plus model
    load and the save headroom. Shown to the user as they select, and the
    window asked for unless they override it."""
    from streetview_to_3d.reconstruct.walk_graph import SECONDS_PER_DOT_ESTIMATE
    return MODEL_LOAD_S + n_dots * (DATE_SAMPLING_PER_DOT_S + SECONDS_PER_DOT_ESTIMATE) + SAVE_BUFFER_S


def _gpu_seconds(points, gpu_seconds=None) -> float:
    """The window this call actually gets: the caller's override, else the
    estimate for this many dots."""
    return float(gpu_seconds) if gpu_seconds else estimate_gpu_seconds(len(points))


def run_walk_gpu(date_graphs, points, adjacency, start_lat, start_lon,
                 step_degrees=None, conf_lower_percentile=None, gpu_seconds=None, model=None,
                 hfov=None, masker=None, mask_classes=None, conf_floor=None):
    """The walk in one GPU window (see streetview_to_3d.gpu).

    gpu_seconds: the window to ask for. None sizes it from the dot count
    (estimate_gpu_seconds). model: a DA3 repo; None is config.DA3_MODEL_REPO.
    hfov, masker, mask_classes, conf_floor: per-run view width, masker and
    confidence floor (services.da3_ops.options)."""
    return gpu.run(_run_walk_impl, date_graphs, points, adjacency, start_lat, start_lon,
                   step_degrees=step_degrees, conf_lower_percentile=conf_lower_percentile,
                   gpu_seconds=gpu_seconds, model=model, hfov=hfov, masker=masker, mask_classes=mask_classes, conf_floor=conf_floor,
                   seconds=_gpu_seconds(points, gpu_seconds))


def _run_walk_impl(date_graphs, points, adjacency, start_lat, start_lon,
                   step_degrees=None, conf_lower_percentile=None, gpu_seconds=None, model=None,
                   hfov=None, masker=None, mask_classes=None, conf_floor=None):
    """The walk (run_pathfind_reconstruction) on the downloaded panos, with
    the whole window but model load and the save headroom.

    Returns [(clouds, metadata), ...], one per piece (reconstruct.pieces)."""
    import itertools
    import tempfile
    import time

    import torch
    from streetview_to_3d.services.da3_ops import (
        CONF_LOWER_PERCENTILE, VIEW_STEP_DEGREES, options, rate_pano as da3_rate_pano,
        test_edge as da3_test_edge,
    )
    from streetview_to_3d.reconstruct.pieces import pieces_to_output
    from streetview_to_3d.reconstruct.walk_graph import run_pathfind_reconstruction

    if conf_lower_percentile is None:
        conf_lower_percentile = CONF_LOWER_PERCENTILE
    if step_degrees is None:
        step_degrees = VIEW_STEP_DEGREES

    t0 = time.monotonic()
    total_s = _gpu_seconds(points, gpu_seconds)
    print(f"GPU window: {total_s:.0f}s for {len(points)} dot(s)"
          f"{' (override)' if gpu_seconds else ''}", flush=True)

    cfg = gpu.get_da3_config(model)
    da3 = gpu.get_da3(model)
    # "timing:" lines are what the per-phase constants above get
    # calibrated from -- grep the Space's logs for them.
    print(f"timing: model load {time.monotonic() - t0:.1f}s", flush=True)
    try:
        with tempfile.TemporaryDirectory() as views_base, options(hfov=hfov, masker=masker, mask_classes=mask_classes, conf_floor=conf_floor):
            def test_edge(path_a, path_b, test_id):
                return da3_test_edge(path_a, path_b, cfg, views_base, da3, test_id=test_id,
                                     step_degrees=step_degrees, conf_lower_percentile=conf_lower_percentile)

            rate_ids = itertools.count()

            def rate_pano(path):
                return da3_rate_pano(path, cfg, views_base, da3, rate_id=next(rate_ids),
                                     step_degrees=step_degrees, conf_lower_percentile=conf_lower_percentile)

            segments = run_pathfind_reconstruction(
                date_graphs, points, adjacency, start_lat, start_lon, test_edge, rate_pano=rate_pano,
                max_time_budget_s=max(0.0, total_s - MODEL_LOAD_S - SAVE_BUFFER_S))
            print(f"timing: walk {time.monotonic() - t0:.1f}s for {len(points)} dot(s) "
                  f"-> {len(segments or [])} piece(s), of {total_s:.0f}s", flush=True)
            return pieces_to_output(segments or [])
    finally:
        torch.cuda.empty_cache()


def run_solo_gpu(places, catalog, conf_lower_percentile=None, gpu_seconds=None, model=None,
                 hfov=None, masker=None, mask_classes=None, conf_floor=None):
    """Solo mode's GPU task (see reconstruct.solo): every place's candidates
    rated alone, the best one's cloud kept. model: a DA3 repo, None is
    config.DA3_MODEL_REPO; hfov, masker: as run_walk_gpu."""
    from streetview_to_3d.reconstruct import solo
    seconds = float(gpu_seconds) if gpu_seconds else solo.estimate_gpu_seconds(places)
    return gpu.run(_run_solo_impl, places, catalog, conf_lower_percentile=conf_lower_percentile,
                   model=model, hfov=hfov, masker=masker, mask_classes=mask_classes, conf_floor=conf_floor, total_s=seconds, seconds=seconds)


def _run_solo_impl(places, catalog, conf_lower_percentile=None, model=None, hfov=None, masker=None, mask_classes=None, conf_floor=None,
                   total_s=None):
    import itertools
    import tempfile
    import time

    import torch
    from streetview_to_3d.reconstruct import solo
    from streetview_to_3d.services.da3_ops import CONF_LOWER_PERCENTILE, options, rate_pano as da3_rate_pano

    if conf_lower_percentile is None:
        conf_lower_percentile = CONF_LOWER_PERCENTILE
    t0 = time.monotonic()
    total_s = total_s or solo.estimate_gpu_seconds(places)
    cfg, da3 = gpu.get_da3_config(model), gpu.get_da3(model)
    print(f"timing: model load {time.monotonic() - t0:.1f}s", flush=True)
    try:
        with tempfile.TemporaryDirectory() as views_base, options(hfov=hfov, masker=masker, mask_classes=mask_classes, conf_floor=conf_floor):
            rate_ids = itertools.count()

            def rate_pano(path):
                return da3_rate_pano(path, cfg, views_base, da3, rate_id=next(rate_ids),
                                     conf_lower_percentile=conf_lower_percentile)

            pieces = solo.reconstruct(places, catalog, rate_pano, t0 + total_s - SAVE_BUFFER_S)
            print(f"timing: solo {time.monotonic() - t0:.1f}s for {len(places)} place(s) "
                  f"-> {len(pieces)} pano(s)", flush=True)
            return pieces
    finally:
        torch.cuda.empty_cache()


def save_pointcloud(points, colors, path):
    """Not GPU-wrapped -- pure disk I/O (numpy/manual PLY write, no
    open3d -- see Saver._voxel_downsample's docstring for why), no CUDA
    involved. Lazy import to match get_da3_config()'s pattern, so this
    module still imports cleanly on machines without panoramic_da3
    installed."""
    from panoramic_da3 import save_da3_pointcloud
    return save_da3_pointcloud(points, colors, path)
