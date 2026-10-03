"""The street reconstruction's GPU task: the walk, in one ZeroGPU window sized from the dot count."""
from dataclasses import dataclass

from streetview_to_3d.models import gpu

# Headroom after the walk for saving the result before the window closes.
SAVE_BUFFER_S = 10.0
# DA3 is loaded at startup (streetview_to_3d.models.gpu), so a call only waits for ZeroGPU to move it.
MODEL_LOAD_S = 5.0


# How hard to try, as GPU seconds per spot: more time lets more capture dates patch weak stretches.
EFFORT_SECONDS_PER_SPOT = {"Quick": 4.0, "Normal": 8.0, "Thorough": 14.0}
DEFAULT_EFFORT = "Normal"


def estimate_gpu_seconds(n_dots: int, effort: str = DEFAULT_EFFORT) -> float:
    """The GPU window for a run over n_dots, model load and save headroom included.
    Kept no bigger than needed: it is checked against the user's ZeroGPU quota."""
    return MODEL_LOAD_S + n_dots * EFFORT_SECONDS_PER_SPOT[effort] + SAVE_BUFFER_S


# The CPU time around the window (downloads, then postprocess): mostly fixed, a little per spot.
OTHER_BASE_S, OTHER_SECONDS_PER_SPOT = 90.0, 4.0


def estimate_other_seconds(n_dots: int) -> float:
    """Roughly how long a run over n_dots takes off the GPU (no quota, only waiting)."""
    return OTHER_BASE_S + n_dots * OTHER_SECONDS_PER_SPOT


def _gpu_seconds(points, gpu_seconds=None) -> float:
    """The caller's window, else the estimate for this many dots."""
    return float(gpu_seconds) if gpu_seconds else estimate_gpu_seconds(len(points))


@dataclass(frozen=True)
class WalkSettings:
    """Per-run settings for the walk; None keeps each default. Picklable for ZeroGPU.

    gpu_seconds: the window (None: estimate_gpu_seconds). model: a DA3 repo (models.da3.DA3_MODEL_REPO).
    hfov, masker, mask_classes, conf_floor: see models.da3.options.
    """
    step_degrees: int | None = None
    conf_lower_percentile: float | None = None
    gpu_seconds: float | None = None
    model: str | None = None
    hfov: float | None = None
    masker: str | None = None
    mask_classes: list[str] | None = None
    conf_floor: float | None = None


def run_walk_gpu(date_graphs, points, adjacency, start_lat, start_lon, settings=WalkSettings()):
    """The walk in one GPU window (see streetview_to_3d.models.gpu)."""
    return gpu.run(_run_walk_impl, date_graphs, points, adjacency, start_lat, start_lon, settings,
                   seconds=_gpu_seconds(points, settings.gpu_seconds))


def _run_walk_impl(date_graphs, points, adjacency, start_lat, start_lon, settings):
    """The walk on the downloaded panos, given the window less model load and save headroom.
    Returns [(clouds, metadata), ...], one per piece (reconstruct.pieces)."""
    import itertools
    import tempfile
    import time

    import torch
    from streetview_to_3d.models.da3 import (
        CONF_LOWER_PERCENTILE, VIEW_STEP_DEGREES, options, rate_pano as da3_rate_pano,
        test_edge as da3_test_edge,
    )
    from streetview_to_3d.reconstruct.pieces import pieces_to_output
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
        with tempfile.TemporaryDirectory() as views_base, options(
                hfov=settings.hfov, masker=settings.masker, mask_classes=settings.mask_classes,
                conf_floor=settings.conf_floor):
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


def save_pointcloud(points, colors, path):
    """Write a point cloud .ply (CPU only; lazy import so this module loads without panoramic_da3)."""
    from panoramic_da3 import save_da3_pointcloud
    return save_da3_pointcloud(points, colors, path)
