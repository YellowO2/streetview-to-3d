"""The GPU side of the app: the ZeroGPU setup, the one DA3 model, and the
one @spaces.GPU call (run_pathfind_and_join_gpu) that walks and joins.

One decorated function, like DA3's own official Space: everything inside
it is plain Python, and the window it asks for is sized from the dot count
(estimate_gpu_seconds). get_da3() is the single DA3Model for the whole app.
"""
import os

# Headroom left after the join for saving the result before the hard
# ZeroGPU window closes. Carved OUT of the window (see _walk_budget_s).
SAVE_BUFFER_S = 10.0
# Loading DA3 inside the GPU window: 40.9s on the first run after a restart
# (it downloads the 6.76 GB weights), less once they are cached. Allowed
# for in full, so a cold start doesn't eat the walk's share.
MODEL_LOAD_S = 45.0

# The join/bridge phase's allowance after the walk: a floor plus a share
# per dot, since a bigger area tends to come out of the walk in more
# pieces to bridge. NOT measured yet -- each run logs "timing:" lines
# (model load, walk, join) to calibrate these against. join_segments
# stops at its own deadline, so an allowance that is too small leaves
# pieces unbridged rather than failing the run.
JOIN_BASE_S = 80.0
JOIN_PER_DOT_S = 3.0


def _join_allowance_s(n_dots: int) -> float:
    return JOIN_BASE_S + n_dots * JOIN_PER_DOT_S


# One solo DA3 rating: 1.36s average in the solo-score experiment
# (README, Dev notes).
SECONDS_PER_RATING = 1.4


def _date_sampling_s(n_dots: int) -> float:
    """At most: every kept date sampled at half its dots, capped at
    DATE_SAMPLES_MAX each (see walk_graph._sample_dates)."""
    from streetview_to_3d.build_street_graph.date_ranking import DATE_TOP_N
    from streetview_to_3d.reconstruct.walk_graph import DATE_SAMPLES_MAX
    per_date = min(DATE_SAMPLES_MAX, max(1, -(-n_dots // 2)))
    return DATE_TOP_N * per_date * SECONDS_PER_RATING


def estimate_gpu_seconds(n_dots: int) -> float:
    """The GPU window a run over n_dots needs: date sampling, the walk's
    own per-dot estimate (walk_graph.SECONDS_PER_DOT_ESTIMATE), then the
    join, plus model load and the save headroom. Shown to the user
    after Prepare, and the window asked for unless they override it."""
    from streetview_to_3d.reconstruct.walk_graph import SECONDS_PER_DOT_ESTIMATE
    return (MODEL_LOAD_S + _date_sampling_s(n_dots) + n_dots * SECONDS_PER_DOT_ESTIMATE
            + _join_allowance_s(n_dots) + SAVE_BUFFER_S)


def _gpu_seconds(points, gpu_seconds=None) -> float:
    """The window this call actually gets: the caller's override, else the
    estimate for this many dots."""
    return float(gpu_seconds) if gpu_seconds else estimate_gpu_seconds(len(points))


def _walk_budget_s(total_s: float, n_dots: int) -> float:
    """The walk's share of a total_s window: everything except model load,
    the join and the save headroom. The join's
    share is capped at a third so a small override still leaves the walk
    room."""
    join_s = min(_join_allowance_s(n_dots), total_s / 3)
    return max(0.0, total_s - MODEL_LOAD_S - join_s - SAVE_BUFFER_S)


try:
    import spaces

    # spaces is also installed locally via requirements.txt, so gate on SPACE_ID
    # which HF Spaces always sets but local machines don't have.
    ON_SPACES = bool(os.getenv("SPACE_ID"))

    def _gpu_duration(date_graphs, points, *args, gpu_seconds=None, **kwargs):
        """spaces.GPU calls this with the decorated function's own
        arguments, so the window follows the dot count (or the override)
        rather than one flat size for every area."""
        return _gpu_seconds(points, gpu_seconds)

    if ON_SPACES:
        GPU_DISPATCH = spaces.GPU(duration=_gpu_duration)
    else:
        GPU_DISPATCH = lambda fn: fn
except ImportError:
    GPU_DISPATCH = lambda fn: fn  # no-op outside HF Spaces
    ON_SPACES = False

_da3_config = None
_da3 = None


def get_da3_config():
    global _da3_config
    if _da3_config is None:
        from streetview_to_3d.config import load_da3_config
        _da3_config = load_da3_config()
    return _da3_config


def get_da3():
    """Lazily built on first real use, INSIDE the GPU call -- not at
    module level (building it before any @spaces.GPU call has attached a
    real GPU segfaults on pycolmap's own raw CUDA calls, which bypass
    spaces' PyTorch-only .to()/.cuda() emulation). Cached in a module-
    level global and REUSED across calls -- matches DA3's own official
    Space (depth_anything_3/app/modules/model_inference.py's
    _MODEL_CACHE/initialize_model): never deleted, but re-checked and
    re-attached to 'cuda' on every single call, not just the first, in
    case it drifted back to CPU between calls (their own code does this
    exact check every time, not just once)."""
    global _da3
    if _da3 is None:
        from panoramic_da3 import DA3Model
        _da3 = DA3Model(get_da3_config().da3_model)
    elif next(_da3.model.parameters()).device.type != "cuda":
        _da3.model = _da3.model.to(device="cuda")
    return _da3


@GPU_DISPATCH
def run_pathfind_and_join_gpu(date_graphs, points, adjacency, start_lat, start_lon,
                               edge_max_dist_m=None, step_degrees=None,
                               conf_lower_percentile=None, gpu_seconds=None):
    """The ONE @spaces.GPU-decorated entry point for this whole app -- see
    this module's own docstring for why there is exactly one. The work
    itself is in _run_pathfind_and_join_impl.

    gpu_seconds: the GPU window to ask for. None sizes it from the dot
    count (estimate_gpu_seconds). Passed by keyword, since _gpu_duration
    reads it by name."""
    return _run_pathfind_and_join_impl(date_graphs, points, adjacency, start_lat, start_lon,
                                        edge_max_dist_m=edge_max_dist_m, step_degrees=step_degrees,
                                        conf_lower_percentile=conf_lower_percentile,
                                        gpu_seconds=gpu_seconds)


def _run_pathfind_and_join_impl(date_graphs, points, adjacency, start_lat, start_lon,
                                 edge_max_dist_m=None, step_degrees=None,
                                 conf_lower_percentile=None, gpu_seconds=None):
    """The walk (run_pathfind_reconstruction), then the join
    (join_segments), in one GPU session on the same downloaded panos.

    Splits ONE wall-clock window (_gpu_seconds: the override, else sized
    from the dot count) between the two phases sequentially: corridor
    search first (capped at _walk_budget_s of it), then whatever's left
    for join/bridging.

    Returns (segments, pieces) -- pieces is a list of (pts, cols,
    metadata) or None if there was only ever one segment (nothing to
    join). See join_segments for what metadata contains."""
    import itertools
    import tempfile
    import time

    import torch
    from streetview_to_3d.services.da3_ops import (
        CONF_LOWER_PERCENTILE, VIEW_STEP_DEGREES, bridge_test_edge as da3_bridge_test_edge,
        rate_pano as da3_rate_pano, test_edge as da3_test_edge,
    )
    from streetview_to_3d.reconstruct.join_segments import BRIDGE_MAX_DIST_M, join_segments
    from streetview_to_3d.reconstruct.walk_graph import run_pathfind_reconstruction

    if edge_max_dist_m is None:
        edge_max_dist_m = BRIDGE_MAX_DIST_M
    if conf_lower_percentile is None:
        conf_lower_percentile = CONF_LOWER_PERCENTILE
    if step_degrees is None:
        step_degrees = VIEW_STEP_DEGREES

    t0 = time.monotonic()
    total_s = _gpu_seconds(points, gpu_seconds)
    hard_deadline = t0 + total_s - SAVE_BUFFER_S
    print(f"GPU window: {total_s:.0f}s for {len(points)} dot(s)"
          f"{' (override)' if gpu_seconds else ''}", flush=True)

    cfg = get_da3_config()
    da3 = get_da3()
    # "timing:" lines are what the per-phase constants above get
    # calibrated from -- grep the Space's logs for them.
    print(f"timing: model load {time.monotonic() - t0:.1f}s", flush=True)
    try:
        with tempfile.TemporaryDirectory() as views_base:
            def test_edge(path_a, path_b, test_id):
                return da3_test_edge(path_a, path_b, cfg, views_base, da3, test_id=test_id,
                                     step_degrees=step_degrees, conf_lower_percentile=conf_lower_percentile)

            rate_ids = itertools.count()

            def rate_pano(path):
                return da3_rate_pano(path, cfg, views_base, da3, rate_id=next(rate_ids),
                                     step_degrees=step_degrees, conf_lower_percentile=conf_lower_percentile)

            def bridge_test_edge(path_a, path_b, test_id):
                return da3_bridge_test_edge(path_a, path_b, cfg, views_base, da3, test_id=test_id,
                                            step_degrees=step_degrees, conf_lower_percentile=conf_lower_percentile)

            t_walk = time.monotonic()
            segments = run_pathfind_reconstruction(date_graphs, points, adjacency, start_lat, start_lon, test_edge,
                                                    rate_pano=rate_pano,
                                                    max_time_budget_s=_walk_budget_s(total_s, len(points)))
            print(f"timing: walk {time.monotonic() - t_walk:.1f}s for {len(points)} dot(s) "
                  f"-> {len(segments or [])} segment(s)", flush=True)
            if not segments or len(segments) < 2:
                print(f"timing: GPU total {time.monotonic() - t0:.1f}s of {total_s:.0f}s", flush=True)
                return segments, None

            # Whatever's left of the hard GPU-session deadline, not the
            # walk's own (smaller) budget. Clamped at 0 so join_segments is
            # never asked to run past hard_deadline, even if the walk
            # overran its own budget.
            remaining_s = max(0.0, hard_deadline - time.monotonic())
            t_join = time.monotonic()
            pieces = join_segments(segments, bridge_test_edge, edge_max_dist_m=edge_max_dist_m, max_time_budget_s=remaining_s)
            print(f"timing: join {time.monotonic() - t_join:.1f}s of {remaining_s:.0f}s allowed, "
                  f"{len(segments)} segment(s) -> {len(pieces)} piece(s)", flush=True)
            print(f"timing: GPU total {time.monotonic() - t0:.1f}s of {total_s:.0f}s", flush=True)
            return segments, pieces
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
