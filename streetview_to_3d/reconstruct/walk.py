"""Run DA3 over the downloaded panos in one ZeroGPU window."""
from streetview_to_3d.models import gpu
from streetview_to_3d.reconstruct.WalkSettings import WalkSettings
from streetview_to_3d.reconstruct.estimate_gpu_seconds import MODEL_LOAD_S, SAVE_BUFFER_S, estimate_gpu_seconds


def _gpu_seconds(points, gpu_seconds=None) -> float:
    """The caller's window, else the estimate for this many dots."""
    return float(gpu_seconds) if gpu_seconds else estimate_gpu_seconds(len(points))


def walk(corridor, settings=WalkSettings()):
    """The walk over the corridor's date graphs (panos.main.panos) in one GPU window (see
    streetview_to_3d.models.gpu): [(clouds, metadata), ...], one per piece (reconstruct.pieces).
    Raises RuntimeError if no piece came out."""
    start_lat, start_lon = corridor.start
    pieces = gpu.run(_run_walk_impl, corridor.date_graphs, corridor.points, corridor.adjacency, start_lat, start_lon,
                     settings, seconds=_gpu_seconds(corridor.points, settings.gpu_seconds))
    if not pieces:
        raise RuntimeError("No connected path found from start toward any goal.")
    return pieces


def _run_walk_impl(date_graphs, points, adjacency, start_lat, start_lon, settings):
    """The walk on the downloaded panos, given the window less model load and save headroom.
    Returns [(clouds, metadata), ...], one per piece (reconstruct.pieces)."""
    import time

    import torch
    from streetview_to_3d.models.da3 import (
        CONF_LOWER_PERCENTILE, VIEW_STEP_DEGREES, options, rate_pano as da3_rate_pano,
        test_edge as da3_test_edge,
    )
    from streetview_to_3d.reconstruct.pieces import pieces
    from streetview_to_3d.reconstruct.walk_graph import run_pathfind_reconstruction

    conf_lower_percentile = (CONF_LOWER_PERCENTILE if settings.conf_lower_percentile is None
                             else settings.conf_lower_percentile)
    step_degrees = VIEW_STEP_DEGREES if settings.step_degrees is None else settings.step_degrees

    t0 = time.monotonic()
    total_s = _gpu_seconds(points, settings.gpu_seconds)
    print(f"GPU window: {total_s:.0f}s for {len(points)} dot(s)", flush=True)

    cfg = gpu.get_da3_config(settings.model)
    da3 = gpu.get_da3(settings.model)
    # "timing:" lines calibrate the constants above (grep the Space's logs)
    print(f"timing: model load {time.monotonic() - t0:.1f}s", flush=True)
    try:
        with options(hfov=settings.hfov, masker=settings.masker, mask_classes=settings.mask_classes,
                     conf_floor=settings.conf_floor):
            def test_edge(path_a, path_b, test_id):
                return da3_test_edge(path_a, path_b, cfg, da3,
                                     step_degrees=step_degrees, conf_lower_percentile=conf_lower_percentile)

            def rate_pano(path):
                return da3_rate_pano(path, cfg, da3,
                                     step_degrees=step_degrees, conf_lower_percentile=conf_lower_percentile)

            segments = run_pathfind_reconstruction(
                date_graphs, points, adjacency, start_lat, start_lon, test_edge, rate_pano=rate_pano,
                max_time_budget_s=max(0.0, total_s - MODEL_LOAD_S - SAVE_BUFFER_S))
            print(f"timing: walk {time.monotonic() - t0:.1f}s for {len(points)} dot(s) "
                  f"-> {len(segments or [])} piece(s), of {total_s:.0f}s", flush=True)
            return pieces(segments or [])
    finally:
        torch.cuda.empty_cache()
