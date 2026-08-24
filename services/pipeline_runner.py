"""GPU-wrapped pipeline runners: SHARP/DA3 3DGS generation, DA3-only point
cloud generation, and the Flux image editor. Also owns the ZeroGPU/@spaces.GPU
decorator setup, since that setup exists purely to wrap these calls.

get_pipeline() is the single Pipeline singleton for this whole app -- both
app.py's Gradio handlers and street_builder's own handlers call through
here, rather than each loading a separate copy of the DA3/SHARP models.

There is exactly ONE @spaces.GPU-decorated function in this whole module
(_gpu_dispatch) -- matches DA3's own official Space (app.py wraps a single
ModelInference.run_inference, everything else is plain Python calling into
it), instead of one decorated function per task. Every public run_*_gpu
function below is a thin, undecorated wrapper that calls _gpu_dispatch
with its own task name -- callers (app.py, street_builder/main.py,
tests/) don't need to change at all, since these functions keep their
same names/signatures. The actual per-task work lives in the _run_*_impl
functions, plain Python, called only from inside _gpu_dispatch where a
real GPU is guaranteed attached.
"""
import os

try:
    import spaces

    # spaces is also installed locally via requirements.txt, so gate on SPACE_ID
    # which HF Spaces always sets but local machines don't have.
    ON_SPACES = bool(os.getenv("SPACE_ID"))
    # TEMP DIAGNOSTIC: hardcoded flat duration, matching DA3's own official
    # Space exactly (duration=120, not a callable) -- testing whether our
    # dynamic per-task duration=callable is itself somehow implicated in
    # the post-GPU-call segfault we've been chasing. If this changes
    # anything, the callable was the culprit; if not, ruled out.
    GPU_WINDOWED_DURATION_S = 120
    PATHFIND_MAX_TIME_BUDGET_S = GPU_WINDOWED_DURATION_S - 30

    if ON_SPACES:
        GPU_DISPATCH = spaces.GPU(duration=120)
    else:
        GPU_DISPATCH = lambda fn: fn
except ImportError:
    GPU_DISPATCH = lambda fn: fn  # no-op outside HF Spaces
    ON_SPACES = False
    GPU_WINDOWED_DURATION_S = 120
    PATHFIND_MAX_TIME_BUDGET_S = GPU_WINDOWED_DURATION_S - 30

_pipeline = None
_flux_editor = None
_da3 = None
if ON_SPACES:
    from editors.flux_editor import FluxEditor
    _flux_editor = FluxEditor(offload=False)


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from panoramic_to_3dgs import Pipeline
        from config import load_pipeline_config
        _pipeline = Pipeline(load_pipeline_config())
    return _pipeline


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
        from panoramic_to_3dgs import DA3Model
        _da3 = DA3Model(get_pipeline().config.da3_model)
    elif next(_da3.model.parameters()).device.type != "cuda":
        _da3.model = _da3.model.to(device="cuda")
    return _da3


@GPU_DISPATCH
def _gpu_dispatch(task, *args, **kwargs):
    """The ONE @spaces.GPU-decorated entry point for this whole app. Every
    run_*_gpu function below routes through here with its own task name
    -- see this module's own docstring for why."""
    impl = {
        "editor": _run_editor_impl,
        "pipeline": _run_pipeline_impl,
        "pointcloud": _run_pointcloud_impl,
        "pathfind_reconstruction": _run_pathfind_reconstruction_impl,
        "join_segments": _join_segments_impl,
        "pathfind_and_join": _run_pathfind_and_join_impl,
    }[task]
    return impl(*args, **kwargs)


def run_editor_gpu(image_path, prompt, mode, output_path):
    return _gpu_dispatch("editor", image_path, prompt, mode, output_path)


def _run_editor_impl(image_path, prompt, mode, output_path):
    global _flux_editor
    if _flux_editor is None:
        from editors.flux_editor import FluxEditor
        _flux_editor = FluxEditor(offload=True)
    _flux_editor.edit(image_path, prompt, mode=mode, output_path=output_path)
    return output_path


def run_pipeline_gpu(target_appearance_path, output_dir, scale_mode, gs_backend, support_paths=None, target_depth_path=None):
    return _gpu_dispatch("pipeline", target_appearance_path, output_dir, scale_mode, gs_backend,
                          support_paths=support_paths, target_depth_path=target_depth_path)


def _run_pipeline_impl(target_appearance_path, output_dir, scale_mode, gs_backend, support_paths=None, target_depth_path=None):
    pipeline = get_pipeline()
    os.makedirs(output_dir, exist_ok=True)
    pipeline.config.scale_mode = scale_mode
    pipeline.config.gs_backend = gs_backend
    pipeline.run(
        target_appearance_path=target_appearance_path,
        output_dir=output_dir,
        target_depth_path=target_depth_path,
        support_paths=support_paths,
    )

    ply = os.path.join(output_dir, "final_output.ply")
    return ply if os.path.exists(ply) else None


def run_pointcloud_gpu(target_depth_path, output_dir, support_paths=None, step_degrees=20):
    return _gpu_dispatch("pointcloud", target_depth_path, output_dir,
                          support_paths=support_paths, step_degrees=step_degrees)


def _run_pointcloud_impl(target_depth_path, output_dir, support_paths=None, step_degrees=20):
    pipeline = get_pipeline()
    os.makedirs(output_dir, exist_ok=True)
    pipeline.run_da3_pointcloud(
        target_depth_path=target_depth_path,
        output_dir=output_dir,
        support_paths=support_paths,
        step_degrees=step_degrees,
    )

    ply = os.path.join(output_dir, "da3_pointcloud.ply")
    return ply if os.path.exists(ply) else None


def run_pathfind_reconstruction_gpu(date_graphs, points, adjacency, start_lat, start_lon, step_degrees=20):
    """See street_builder/main.py's module docstring for why the whole
    pathfind search runs inside one GPU call, not several."""
    return _gpu_dispatch("pathfind_reconstruction", date_graphs, points, adjacency, start_lat, start_lon,
                          step_degrees=step_degrees)


def _run_pathfind_reconstruction_impl(date_graphs, points, adjacency, start_lat, start_lon, step_degrees=20):
    """This function's only job is to hand the actual algorithm
    (street_builder/reconstruction/walk_graph.py) a way to test one edge,
    using the shared cached DA3 model (see get_da3) -- it knows nothing
    about corridors, dates, or coverage itself.

    date_graphs: already ranked/capped/isolated per date, dot_candidates
    shape -- see street_builder/build_graph/build_graph.py's
    build_corridor_graphs. adjacency: the corridor's shared dot-to-dot
    structural graph, same source."""
    import itertools
    import tempfile

    import torch
    from services.da3_ops import rate_pano as da3_rate_pano, test_edge as da3_test_edge
    from street_builder.reconstruction.walk_graph import run_pathfind_reconstruction

    pipeline = get_pipeline()
    da3 = get_da3()
    try:
        with tempfile.TemporaryDirectory() as views_base:
            def test_edge(path_a, path_b, test_id):
                return da3_test_edge(path_a, path_b, pipeline.config, views_base, da3, test_id=test_id, step_degrees=step_degrees)

            rate_ids = itertools.count()

            def rate_pano(path):
                return da3_rate_pano(path, pipeline.config, views_base, da3, rate_id=next(rate_ids), step_degrees=step_degrees)

            return run_pathfind_reconstruction(date_graphs, points, adjacency, start_lat, start_lon, test_edge,
                                                rate_pano=rate_pano, max_time_budget_s=PATHFIND_MAX_TIME_BUDGET_S)
    finally:
        # Release CACHED (unused) allocator memory, like DA3's own official
        # Space does after every call -- NOT del da3, which would force a
        # full reconstruction (and re-trigger the module-import segfault
        # risk) next call. The model itself stays alive in get_da3()'s cache.
        torch.cuda.empty_cache()


def join_segments_gpu(segments, edge_max_dist_m=None, step_degrees=20,
                       chunk_ids=None, known_adjacent_chunk_pairs=None):
    """See _join_segments_impl for the real docstring -- this is just the
    thin dispatch wrapper (see this module's own docstring for why)."""
    return _gpu_dispatch("join_segments", segments, edge_max_dist_m=edge_max_dist_m, step_degrees=step_degrees,
                          chunk_ids=chunk_ids, known_adjacent_chunk_pairs=known_adjacent_chunk_pairs)


def _join_segments_impl(segments, edge_max_dist_m=None, step_degrees=20,
                         chunk_ids=None, known_adjacent_chunk_pairs=None):
    """The join step's bridging search (see
    street_builder/reconstruction/join_segments.py) -- its own separate
    task from pathfind_reconstruction's, since bridging only needs each
    segment's own already-confirmed nodes (no candidate pool, no
    corridor/date data), not anything from the corridor search itself.
    Keeping it separate means bridging behavior can be iterated on (or
    Join re-run on a saved segments bundle) without re-paying for the
    whole corridor search each time.

    edge_max_dist_m: None (default) defers to join_segments.py's own
    BRIDGE_MAX_DIST_M -- kept as a single source of truth instead of a
    second hardcoded default here, so the two can't silently drift apart.
    chunk_ids/known_adjacent_chunk_pairs: passed straight through to
    join_segments -- see its own docstring (restricts bridging to known-
    adjacent chunk pairs instead of a blind all-pairs scan, for a large-
    scale multi-chunk reconstruction where the caller already knows
    which pieces are meant to connect).

    Returns a list of (points, colors, metadata) -- see join_segments.

    Re-fetches each candidate pano fresh (by key) right before testing it,
    rather than trusting frame_poses' stored path -- a separately-called
    join_segments task has no guarantee the ORIGINAL downloaded image
    files that produced `segments` still exist on whatever worker/disk
    this call lands on (see refetch_path in join_segments.py's
    _try_bridge). Google-only for now -- Apple has no fetch-by-id-alone
    helper yet, so an Apple node's pair attempts are skipped like a
    normal per-attempt failure rather than erroring."""
    import tempfile

    import torch
    from services.da3_ops import bridge_test_edge as da3_bridge_test_edge
    from services.streetview_fetch import DA3_ONLY_ZOOM, download_pano_by_id, run_async
    from street_builder.reconstruction.join_segments import BRIDGE_MAX_DIST_M, join_segments

    if edge_max_dist_m is None:
        edge_max_dist_m = BRIDGE_MAX_DIST_M

    def refetch_path(key):
        source, pano_id = key.split(":", 1)
        if source != "google":
            return None
        try:
            return run_async(download_pano_by_id(pano_id, zoom=DA3_ONLY_ZOOM))
        except Exception as e:
            print(f"[bridge] refetch failed for {key}: {e}")
            return None

    pipeline = get_pipeline()
    da3 = get_da3()
    try:
        with tempfile.TemporaryDirectory() as views_base:
            def bridge_test_edge(path_a, path_b, test_id):
                return da3_bridge_test_edge(path_a, path_b, pipeline.config, views_base, da3, test_id=test_id, step_degrees=step_degrees)

            return join_segments(segments, bridge_test_edge, edge_max_dist_m=edge_max_dist_m,
                                  chunk_ids=chunk_ids, known_adjacent_chunk_pairs=known_adjacent_chunk_pairs,
                                  refetch_path=refetch_path)
    finally:
        torch.cuda.empty_cache()


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

    Splits the one GPU_WINDOWED time budget between the two phases
    sequentially (corridor search first, then whatever's left over for
    join/bridging) rather than each phase getting its own full budget --
    they're sharing one real wall-clock window here, not two.

    Returns (segments, pieces) -- pieces is a list of (pts, cols,
    metadata) or None if there was only ever one segment (nothing to
    join). See join_segments for what metadata contains."""
    import itertools
    import tempfile
    import time

    import torch
    from services.da3_ops import bridge_test_edge as da3_bridge_test_edge, rate_pano as da3_rate_pano, test_edge as da3_test_edge
    from street_builder.reconstruction.join_segments import BRIDGE_MAX_DIST_M, join_segments
    from street_builder.reconstruction.walk_graph import run_pathfind_reconstruction

    if edge_max_dist_m is None:
        edge_max_dist_m = BRIDGE_MAX_DIST_M

    t0 = time.monotonic()
    overall_deadline = t0 + PATHFIND_MAX_TIME_BUDGET_S

    pipeline = get_pipeline()
    da3 = get_da3()
    try:
        with tempfile.TemporaryDirectory() as views_base:
            def test_edge(path_a, path_b, test_id):
                return da3_test_edge(path_a, path_b, pipeline.config, views_base, da3, test_id=test_id, step_degrees=step_degrees)

            rate_ids = itertools.count()

            def rate_pano(path):
                return da3_rate_pano(path, pipeline.config, views_base, da3, rate_id=next(rate_ids), step_degrees=step_degrees)

            def bridge_test_edge(path_a, path_b, test_id):
                return da3_bridge_test_edge(path_a, path_b, pipeline.config, views_base, da3, test_id=test_id, step_degrees=step_degrees)

            segments = run_pathfind_reconstruction(date_graphs, points, adjacency, start_lat, start_lon, test_edge,
                                                    rate_pano=rate_pano, max_time_budget_s=PATHFIND_MAX_TIME_BUDGET_S)
            if not segments or len(segments) < 2:
                return segments, None

            remaining_s = max(10.0, overall_deadline - time.monotonic())
            pieces = join_segments(segments, bridge_test_edge, edge_max_dist_m=edge_max_dist_m, max_time_budget_s=remaining_s)
            return segments, pieces
    finally:
        torch.cuda.empty_cache()


def save_pointcloud(points, colors, path):
    """Not GPU-wrapped -- pure disk I/O (open3d write), no CUDA involved.
    Lazy import to match get_pipeline()'s pattern, so this module still
    imports cleanly on machines without panoramic_to_3dgs installed."""
    from panoramic_to_3dgs import save_da3_pointcloud
    return save_da3_pointcloud(points, colors, path)
