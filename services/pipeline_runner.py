"""GPU-wrapped pipeline runners: DA3-only point cloud generation, plus
reconstruct's corridor pathfinding/join tasks. Also owns the
ZeroGPU/@spaces.GPU decorator setup, since that setup exists purely to wrap
these calls.

get_da3() is the single DA3Model singleton for this whole app --
reconstruct's handlers call through here, rather than each loading a
separate copy.

There is exactly ONE @spaces.GPU-decorated function in this whole module
(_gpu_dispatch) -- matches DA3's own official Space (app.py wraps a single
ModelInference.run_inference, everything else is plain Python calling into
it), instead of one decorated function per task. Every public run_*_gpu
function below is a thin, undecorated wrapper that calls _gpu_dispatch
with its own task name -- callers (app.py, reconstruct/build.py,
tests/) don't need to change at all, since these functions keep their
same names/signatures. The actual per-task work lives in the _run_*_impl
functions, plain Python, called only from inside _gpu_dispatch where a
real GPU is guaranteed attached.
"""
import os

# Self-bridge/join's own guaranteed minimum window after the walk,
# regardless of how much of PATHFIND_MAX_TIME_BUDGET_S the walk itself
# used -- see _run_pathfind_reconstruction_impl's own docstring.
# SAVE_BUFFER_S: headroom left after bridging for saving/uploading the
# result before the hard ZeroGPU wall-clock window closes. Both are
# subtracted from GPU_WINDOWED_DURATION_S to get the walk's own budget
# (PATHFIND_MAX_TIME_BUDGET_S) -- carved OUT of the one hard window, not
# added on top of it.
SELF_BRIDGE_MIN_S = 20.0
SAVE_BUFFER_S = 10.0

try:
    import spaces

    # spaces is also installed locally via requirements.txt, so gate on SPACE_ID
    # which HF Spaces always sets but local machines don't have.
    ON_SPACES = bool(os.getenv("SPACE_ID"))
    # Flat duration -- started at 120 to match DA3's own official Space
    # (duration=120, not a per-task callable; the real cause of the
    # second-@spaces.GPU-call segfault this was originally chasing turned
    # out to be unrelated -- open3d's persistent background thread pool
    # in Saver.save_point_cloud, fixed in panoramic-da3). Bumped to 180:
    # self-bridge kept getting starved of real time to work with once the
    # walk alone routinely used most of a 120s window (confirmed on real
    # data: a 20-dot chunk's walk took 94.6s), and our chunk sizes are
    # consistently similar (~20 dots), so a flat bump is simpler than a
    # dynamic per-task duration -- bump further if 180 still isn't enough.
    GPU_WINDOWED_DURATION_S = 180
    # pathfind_and_join runs the walk AND the join/bridge phase
    # sequentially in ONE call, sharing whatever window it gets -- giving
    # it just GPU_WINDOWED_DURATION_S (sized for walk-alone/join-alone)
    # would starve whichever phase runs second. Sized instead as the
    # previous walk budget PLUS join's own standalone default (see
    # join_segments.join_segments's own 200s default), so combining the
    # two steps into one call doesn't cost either phase the time it'd
    # get running separately.
    RUN_AND_JOIN_DURATION_S = GPU_WINDOWED_DURATION_S + 200.0
    PATHFIND_MAX_TIME_BUDGET_S = GPU_WINDOWED_DURATION_S - SELF_BRIDGE_MIN_S - SAVE_BUFFER_S

    def _gpu_duration(task, *args, **kwargs):
        """Per-call duration for the ONE @spaces.GPU-decorated dispatch
        (see _gpu_dispatch's own docstring for why there's only one) --
        every task gets the normal shared window except pathfind_and_join,
        which needs room for both the walk AND a genuinely unhurried join
        afterward (see RUN_AND_JOIN_DURATION_S)."""
        return RUN_AND_JOIN_DURATION_S if task == "pathfind_and_join" else GPU_WINDOWED_DURATION_S

    if ON_SPACES:
        GPU_DISPATCH = spaces.GPU(duration=_gpu_duration)
    else:
        GPU_DISPATCH = lambda fn: fn
except ImportError:
    GPU_DISPATCH = lambda fn: fn  # no-op outside HF Spaces
    ON_SPACES = False
    GPU_WINDOWED_DURATION_S = 180
    RUN_AND_JOIN_DURATION_S = GPU_WINDOWED_DURATION_S + 200.0
    PATHFIND_MAX_TIME_BUDGET_S = GPU_WINDOWED_DURATION_S - SELF_BRIDGE_MIN_S - SAVE_BUFFER_S

_da3_config = None
_da3 = None


def get_da3_config():
    global _da3_config
    if _da3_config is None:
        from config import load_da3_config
        _da3_config = load_da3_config()
    return _da3_config


def get_da3():
    """Lazily built on first real use, INSIDE _gpu_dispatch -- not at
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
def _gpu_dispatch(task, *args, **kwargs):
    """The ONE @spaces.GPU-decorated entry point for this whole app. Every
    run_*_gpu function below routes through here with its own task name
    -- see this module's own docstring for why."""
    impl = {
        "pointcloud": _run_pointcloud_impl,
        "pathfind_reconstruction": _run_pathfind_reconstruction_impl,
        "join_segments": _join_segments_impl,
        "pathfind_and_join": _run_pathfind_and_join_impl,
        "bridge_incremental": _bridge_incremental_impl,
        "bridge_metadata": _bridge_metadata_impl,
    }[task]
    return impl(*args, **kwargs)






def run_pathfind_reconstruction_gpu(date_graphs, points, adjacency, start_lat, start_lon, step_degrees=20, protected_positions=None):
    """See reconstruct/build.py's module docstring for why the whole
    pathfind search runs inside one GPU call, not several."""
    return _gpu_dispatch("pathfind_reconstruction", date_graphs, points, adjacency, start_lat, start_lon,
                          step_degrees=step_degrees, protected_positions=protected_positions)
























def run_pathfind_and_join_gpu(date_graphs, points, adjacency, start_lat, start_lon,
                               edge_max_dist_m=None, step_degrees=20):
    """See _run_pathfind_and_join_impl for the real docstring -- this is
    just the thin dispatch wrapper (see this module's own docstring for
    why)."""
    return _gpu_dispatch("pathfind_and_join", date_graphs, points, adjacency, start_lat, start_lon,
                          edge_max_dist_m=edge_max_dist_m, step_degrees=step_degrees)


def _run_pathfind_and_join_impl(date_graphs, points, adjacency, start_lat, start_lon,
                                 edge_max_dist_m=None, step_degrees=20):
    """Convenience combined task: corridor search (run_pathfind_reconstruction)
    AND join/bridging (join_segments) in ONE GPU session, using the same
    already-downloaded local image paths for both phases -- Join re-run
    as a separate task needs to re-fetch each candidate pano fresh
    instead (see _join_segments_impl's refetch_path), since a separate
    call has no guarantee of landing on the same worker/disk. Use when
    you just want the final result end-to-end and don't need to iterate
    on join/bridging separately -- pathfind_reconstruction/join_segments
    (split) are still the right choice for re-testing Join alone against
    an already-saved segments bundle, without redoing the whole (much
    more expensive) corridor search.

    Splits ONE wall-clock window (RUN_AND_JOIN_DURATION_S, sized as the
    walk's own normal budget PLUS join's own standalone default -- see
    that constant's own comment) between the two phases sequentially:
    corridor search first (still capped at the same PATHFIND_MAX_TIME_BUDGET_S
    a plain walk gets), then whatever's left of the bigger window for
    join/bridging -- which now amounts to roughly a full, unhurried join
    budget rather than the walk's leftover scraps.

    Returns (segments, pieces) -- pieces is a list of (pts, cols,
    metadata) or None if there was only ever one segment (nothing to
    join). See join_segments for what metadata contains."""
    import itertools
    import tempfile
    import time

    import torch
    from services.da3_ops import bridge_test_edge as da3_bridge_test_edge, rate_pano as da3_rate_pano, test_edge as da3_test_edge
    from reconstruct.join_segments import BRIDGE_MAX_DIST_M, join_segments
    from reconstruct.walk_graph import run_pathfind_reconstruction

    if edge_max_dist_m is None:
        edge_max_dist_m = BRIDGE_MAX_DIST_M

    t0 = time.monotonic()
    hard_deadline = t0 + RUN_AND_JOIN_DURATION_S - SAVE_BUFFER_S

    cfg = get_da3_config()
    da3 = get_da3()
    try:
        with tempfile.TemporaryDirectory() as views_base:
            def test_edge(path_a, path_b, test_id):
                return da3_test_edge(path_a, path_b, cfg, views_base, da3, test_id=test_id, step_degrees=step_degrees)

            rate_ids = itertools.count()

            def rate_pano(path):
                return da3_rate_pano(path, cfg, views_base, da3, rate_id=next(rate_ids), step_degrees=step_degrees)

            def bridge_test_edge(path_a, path_b, test_id):
                return da3_bridge_test_edge(path_a, path_b, cfg, views_base, da3, test_id=test_id, step_degrees=step_degrees)

            segments = run_pathfind_reconstruction(date_graphs, points, adjacency, start_lat, start_lon, test_edge,
                                                    rate_pano=rate_pano, max_time_budget_s=PATHFIND_MAX_TIME_BUDGET_S)
            if not segments or len(segments) < 2:
                return segments, None

            # Whatever's left of the hard GPU-session deadline, not the
            # walk's own (smaller) budget -- see SELF_BRIDGE_MIN_S's own
            # docstring. Clamped at 0 so this can never go negative (and
            # so never asks join_segments to run past hard_deadline) even
            # if the walk somehow overran its own budget.
            remaining_s = max(0.0, hard_deadline - time.monotonic())
            pieces = join_segments(segments, bridge_test_edge, edge_max_dist_m=edge_max_dist_m, max_time_budget_s=remaining_s)
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
