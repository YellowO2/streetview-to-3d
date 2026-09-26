"""The street reconstruction's GPU task: the walk then the join, in one
window sized from the dot count (estimate_gpu_seconds). The GPU itself and
the DA3 model come from streetview_to_3d.gpu.
"""
from streetview_to_3d import gpu

# Headroom left after the join for saving the result before the hard
# ZeroGPU window closes. Carved OUT of the window (see _walk_budget_s).
SAVE_BUFFER_S = 10.0
# DA3 is loaded at startup (see streetview_to_3d.gpu), so a call only waits
# for ZeroGPU to move it onto the GPU: "timing: model load 0.0s" measured.
# Was 45 when it loaded from disk inside the call (17-41 s).
MODEL_LOAD_S = 5.0

# The join/bridge phase's allowance after the walk: a floor plus a share
# per dot, since a bigger area tends to come out of the walk in more
# pieces to bridge. Measured on the Space: 4 dots 23 s, 7 dots 12-98 s
# (3 to 7 segments). join_segments stops at its own deadline, so an
# allowance that is too small leaves pieces unbridged rather than failing.
JOIN_BASE_S = 10.0
JOIN_PER_DOT_S = 8.0


def _join_allowance_s(n_dots: int) -> float:
    return JOIN_BASE_S + n_dots * JOIN_PER_DOT_S


# Date sampling, per dot: measured 7 s for 4 dots, 13 s for 7.
DATE_SAMPLING_PER_DOT_S = 2.0


def estimate_gpu_seconds(n_dots: int) -> float:
    """The GPU window a run over n_dots needs: date sampling, the walk's
    own per-dot estimate (walk_graph.SECONDS_PER_DOT_ESTIMATE), then the
    join, plus model load and the save headroom. Shown to the user
    as they select, and the window asked for unless they override it."""
    from streetview_to_3d.reconstruct.walk_graph import SECONDS_PER_DOT_ESTIMATE
    return (MODEL_LOAD_S + n_dots * (DATE_SAMPLING_PER_DOT_S + SECONDS_PER_DOT_ESTIMATE)
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


def run_pathfind_and_join_gpu(date_graphs, points, adjacency, start_lat, start_lon,
                               edge_max_dist_m=None, step_degrees=None,
                               conf_lower_percentile=None, gpu_seconds=None):
    """Walk and join in one GPU window (see streetview_to_3d.gpu).

    gpu_seconds: the window to ask for. None sizes it from the dot count
    (estimate_gpu_seconds)."""
    return gpu.run(_run_pathfind_and_join_impl, date_graphs, points, adjacency,
                   start_lat, start_lon, edge_max_dist_m=edge_max_dist_m,
                   step_degrees=step_degrees, conf_lower_percentile=conf_lower_percentile,
                   gpu_seconds=gpu_seconds, seconds=_gpu_seconds(points, gpu_seconds))


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

    cfg = gpu.get_da3_config()
    da3 = gpu.get_da3()
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


def run_solo_gpu(places, catalog, conf_lower_percentile=None, gpu_seconds=None,
                 hfov=None, ring_pitches=None):
    """Solo mode's GPU task (see reconstruct.solo): every place's candidates
    rated alone with the solo model, the best one's cloud kept. hfov and
    ring_pitches: how panos are cut into views; None keeps solo's defaults."""
    from streetview_to_3d.reconstruct import solo
    views = dict(hfov=hfov or solo.VIEW_HFOV,
                 ring_pitches=tuple(solo.RING_PITCHES if ring_pitches is None else ring_pitches))
    seconds = float(gpu_seconds) if gpu_seconds else solo.estimate_gpu_seconds(places, views["ring_pitches"])
    return gpu.run(_run_solo_impl, places, catalog, conf_lower_percentile=conf_lower_percentile,
                   views=views, total_s=seconds, seconds=seconds)


def _run_solo_impl(places, catalog, conf_lower_percentile=None, views=None, total_s=None):
    import itertools
    import tempfile
    import time

    import torch
    from streetview_to_3d.config import DA3_SOLO_MODEL_REPO
    from streetview_to_3d.reconstruct import solo
    from streetview_to_3d.services.da3_ops import CONF_LOWER_PERCENTILE, rate_pano as da3_rate_pano

    if conf_lower_percentile is None:
        conf_lower_percentile = CONF_LOWER_PERCENTILE
    t0 = time.monotonic()
    views = views or {}
    total_s = total_s or solo.estimate_gpu_seconds(places)
    cfg, da3 = gpu.get_da3_config(DA3_SOLO_MODEL_REPO), gpu.get_da3(DA3_SOLO_MODEL_REPO)
    print(f"timing: model load {time.monotonic() - t0:.1f}s; views {views}", flush=True)
    try:
        with tempfile.TemporaryDirectory() as views_base:
            rate_ids = itertools.count()

            def rate_pano(path):
                return da3_rate_pano(path, cfg, views_base, da3, rate_id=next(rate_ids),
                                     conf_lower_percentile=conf_lower_percentile, **views)

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
