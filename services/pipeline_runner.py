"""GPU-wrapped pipeline runners: DA3-only point cloud generation, plus
street_builder's corridor pathfinding/join tasks. Also owns the
ZeroGPU/@spaces.GPU decorator setup, since that setup exists purely to wrap
these calls.

get_da3() is the single DA3Model singleton for this whole app --
street_builder's handlers call through here, rather than each loading a
separate copy.

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
    PATHFIND_MAX_TIME_BUDGET_S = GPU_WINDOWED_DURATION_S - SELF_BRIDGE_MIN_S - SAVE_BUFFER_S

    if ON_SPACES:
        GPU_DISPATCH = spaces.GPU(duration=GPU_WINDOWED_DURATION_S)
    else:
        GPU_DISPATCH = lambda fn: fn
except ImportError:
    GPU_DISPATCH = lambda fn: fn  # no-op outside HF Spaces
    ON_SPACES = False
    GPU_WINDOWED_DURATION_S = 180
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


def run_pointcloud_gpu(target_depth_path, output_dir, support_paths=None, step_degrees=20):
    return _gpu_dispatch("pointcloud", target_depth_path, output_dir,
                          support_paths=support_paths, step_degrees=step_degrees)


def _run_pointcloud_impl(target_depth_path, output_dir, support_paths=None, step_degrees=20):
    import tempfile

    import torch
    from panoramic_da3 import run_da3, save_da3_pointcloud

    os.makedirs(output_dir, exist_ok=True)
    cfg = get_da3_config()
    da3 = get_da3()
    try:
        with tempfile.TemporaryDirectory() as views_base:
            _, _, pts, cols, _, _ = run_da3(target_depth_path, support_paths or [], cfg, views_base,
                                             da3=da3, step_degrees=step_degrees)
        if pts is None:
            return None
        ply = os.path.join(output_dir, "da3_pointcloud.ply")
        save_da3_pointcloud(pts, cols, ply)
        return ply
    finally:
        torch.cuda.empty_cache()


def run_pathfind_reconstruction_gpu(date_graphs, points, adjacency, start_lat, start_lon, step_degrees=20, protected_positions=None):
    """See street_builder/main.py's module docstring for why the whole
    pathfind search runs inside one GPU call, not several."""
    return _gpu_dispatch("pathfind_reconstruction", date_graphs, points, adjacency, start_lat, start_lon,
                          step_degrees=step_degrees, protected_positions=protected_positions)


def _run_pathfind_reconstruction_impl(date_graphs, points, adjacency, start_lat, start_lon, step_degrees=20, protected_positions=None):
    """This function's only job is to hand the actual algorithm
    (street_builder/reconstruction/walk_graph.py) a way to test one edge,
    using the shared cached DA3 model (see get_da3) -- it knows nothing
    about corridors, dates, or coverage itself.

    date_graphs: already ranked/capped/isolated per date, dot_candidates
    shape -- see street_builder/build_graph/build_graph.py's
    build_corridor_graphs. adjacency: the corridor's shared dot-to-dot
    structural graph, same source. protected_positions: passed straight
    through to run_pathfind_reconstruction -- see its own docstring.

    After the walk, runs a SELF-bridge pass (bridge_pieces) before
    returning -- a blind, unrestricted all-pairs scan among just this
    call's own segments (no chunk_ids/known_adjacent_chunk_pairs).
    Tried restricting this to structural_pairs (real one-hop dot-graph
    neighbors, computed by run_pathfind_reconstruction itself) and
    reverted it: two fragments can be genuinely close (well within
    BRIDGE_MAX_DIST_M) and still not be direct one-hop graph neighbors
    -- e.g. the dot directly between them itself failed and became its
    own stranded fragment -- so the one-hop restriction silently skipped
    real, biddable connections instead of just saving redundant work.
    Confirmed on real data: a 20-dot chunk's own 6 self-bridge fragments
    were all mutually reachable via a real <=30m chain, but the one-hop
    version left them separate. A blind scan here is genuinely cheap
    (a chunk has at most a handful of fragments, not hundreds), so there
    was nothing to actually save by restricting it -- unlike cross-chunk
    bridging (_bridge_incremental_impl), which operates over many more
    pieces at full-campus scale and keeps its own chunk-id-level
    restriction for that reason. One failed real DA3 test during the
    walk permanently leaves that specific dot pair disconnected (see
    walk_graph.visit's own docstring -- no retry, ever); this gives every
    fragment one more real shot at every other fragment before this
    chunk's result is ever saved/handed to cross-chunk bridging. Same GPU
    session/already-open da3 model as the walk itself -- no separate
    @spaces.GPU call.

    Gets its own guaranteed SELF_BRIDGE_MIN_S window after the walk, NOT
    just whatever's left of the walk's own PATHFIND_MAX_TIME_BUDGET_S --
    a chunk with several fragments needing several real DA3 attempts each
    could otherwise get ~0s if the walk alone used its whole budget
    (confirmed on real data: a 20-dot chunk's walk took 94.6s, leaving
    self-bridge nothing, so its own 6 fragments never got a real chance
    to merge even though a real <=30m chain connected all of them).
    Carved out of the SAME hard GPU_WINDOWED_DURATION_S wall-clock window
    (via hard_deadline), not extra time on top of it."""
    import itertools
    import tempfile
    import time

    import torch
    from services.da3_ops import bridge_test_edge as da3_bridge_test_edge, rate_pano as da3_rate_pano, test_edge as da3_test_edge
    from street_builder.reconstruction.join_segments import BRIDGE_MAX_DIST_M, bridge_pieces
    from street_builder.reconstruction.walk_graph import run_pathfind_reconstruction

    t0 = time.monotonic()
    hard_deadline = t0 + GPU_WINDOWED_DURATION_S - SAVE_BUFFER_S
    cfg = get_da3_config()
    da3 = get_da3()
    try:
        with tempfile.TemporaryDirectory() as views_base:
            def test_edge(path_a, path_b, test_id):
                return da3_test_edge(path_a, path_b, cfg, views_base, da3, test_id=test_id, step_degrees=step_degrees)

            rate_ids = itertools.count()

            def rate_pano(path):
                return da3_rate_pano(path, cfg, views_base, da3, rate_id=next(rate_ids), step_degrees=step_degrees)

            segments = run_pathfind_reconstruction(date_graphs, points, adjacency, start_lat, start_lon, test_edge,
                                                    rate_pano=rate_pano, max_time_budget_s=PATHFIND_MAX_TIME_BUDGET_S,
                                                    protected_positions=protected_positions)
            if len(segments) < 2 or time.monotonic() >= hard_deadline:
                return segments

            def bridge_test_edge(path_a, path_b, test_id):
                return da3_bridge_test_edge(path_a, path_b, cfg, views_base, da3, test_id=test_id, step_degrees=step_degrees)

            # Whatever's left of the hard GPU-session deadline, not the
            # walk's own (smaller) budget -- see SELF_BRIDGE_MIN_S's own
            # docstring. Never exceeds hard_deadline (the true ZeroGPU
            # wall-clock ceiling), and typically has ~SELF_BRIDGE_MIN_S
            # to work with since the walk's own budget is capped well
            # below hard_deadline -- but uses MORE if the walk finished
            # early, rather than wasting the unused time.
            return bridge_pieces(segments, bridge_test_edge, edge_max_dist_m=BRIDGE_MAX_DIST_M, deadline=hard_deadline,
                                  raise_on_unsatisfied=False)
    finally:
        # Release CACHED (unused) allocator memory, like DA3's own official
        # Space does after every call -- NOT del da3, which would force a
        # full reconstruction (and re-trigger the module-import segfault
        # risk) next call. The model itself stays alive in get_da3()'s cache.
        torch.cuda.empty_cache()


def _refetch_bridge_pano(key, lat, lon):
    """Shared refetch_path implementation for both _join_segments_impl and
    _bridge_incremental_impl -- re-downloads ONE candidate pano fresh, by
    key, right before a bridge DA3 test (see join_segments.py's
    _try_bridge for why: a separately-called bridge task has no
    guarantee the ORIGINAL image file still exists on whatever worker/
    disk this call lands on). Google fetches by id alone; Apple has no
    such lookup (streetlevel.lookaround only offers fetch-by-coverage-
    tile, see map_selection.candidates.apple_tile_panos), so lat/lon
    (the node's own real position, already carried in frame_poses) picks
    which tile neighborhood to search, then the id picks the pano within
    it -- same two-step lookup main.py's prepare_pathfind_from_cover_chunk
    already does for its own Apple re-fetch."""
    from services.lookaround_fetch import DA3_ONLY_APPLE_ZOOM, download_lookaround
    from services.streetview_fetch import DA3_ONLY_ZOOM, download_pano_by_id, run_async
    from street_builder.map_selection.candidates import apple_tile_panos

    source, pano_id = key.split(":", 1)
    try:
        if source == "google":
            return run_async(download_pano_by_id(pano_id, zoom=DA3_ONLY_ZOOM))
        if source == "apple":
            pano = apple_tile_panos(lat, lon).get(int(pano_id))
            if pano is None:
                print(f"[bridge] refetch failed for {key}: not found in the coverage tile near ({lat}, {lon})")
                return None
            return download_lookaround(pano, zoom=DA3_ONLY_APPLE_ZOOM)
        print(f"[bridge] refetch failed for {key}: unknown source {source!r}")
        return None
    except Exception as e:
        print(f"[bridge] refetch failed for {key}: {e}")
        return None


def _frame_pose_positions(segments):
    """{key: (lat, lon)} across every given segment's own frame_poses,
    deduped by key -- plain CPU data (no GPU needed), used to prefetch
    every candidate pano's image BEFORE the GPU call touches it (see
    _prefetch_bridge_panos)."""
    positions = {}
    for seg in segments:
        frame_poses = seg[6]
        for key, (_, _, _, lat, lon, _, _) in frame_poses.items():
            positions[key] = (lat, lon)
    return positions


def _prefetch_bridge_panos(positions, concurrency=10):
    """positions: {key: (lat, lon)}. Downloads every one of them
    CONCURRENTLY, network-only, BEFORE any @spaces.GPU call touches them
    -- refetch_path used to do this same per-key download live INSIDE
    the GPU session (see _refetch_bridge_pano), burning GPU-billed
    wall-clock on pure network I/O for every bridge attempt, when the
    download itself never needed a GPU at all. Returns {key: path or
    None (download failed)}, fed straight into the GPU call as a plain
    dict lookup instead of a live fetch."""
    from concurrent.futures import ThreadPoolExecutor

    if not positions:
        return {}
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_refetch_bridge_pano, key, lat, lon): key for key, (lat, lon) in positions.items()}
        return {futures[fut]: fut.result() for fut in futures}


def join_segments_gpu(segments, edge_max_dist_m=None, step_degrees=20,
                       chunk_ids=None, known_adjacent_chunk_pairs=None):
    """See _join_segments_impl for the real docstring -- this is just the
    thin dispatch wrapper (see this module's own docstring for why).
    Prefetches every candidate pano (see _prefetch_bridge_panos) here,
    OUTSIDE the GPU call, before dispatching."""
    path_by_key = _prefetch_bridge_panos(_frame_pose_positions(segments))
    return _gpu_dispatch("join_segments", segments, path_by_key, edge_max_dist_m=edge_max_dist_m, step_degrees=step_degrees,
                          chunk_ids=chunk_ids, known_adjacent_chunk_pairs=known_adjacent_chunk_pairs)


def _join_segments_impl(segments, path_by_key, edge_max_dist_m=None, step_degrees=20,
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

    path_by_key: {key: path or None} -- prefetched by join_segments_gpu
    BEFORE this GPU call (see _prefetch_bridge_panos), since a
    separately-called join_segments task has no guarantee the ORIGINAL
    downloaded image files that produced `segments` still exist on
    whatever worker/disk this call lands on (see refetch_path in
    join_segments.py's _try_bridge) -- refetching used to happen live,
    INSIDE this GPU session; a plain dict lookup here instead."""
    import tempfile

    import torch
    from services.da3_ops import bridge_test_edge as da3_bridge_test_edge
    from street_builder.reconstruction.join_segments import BRIDGE_MAX_DIST_M, join_segments

    if edge_max_dist_m is None:
        edge_max_dist_m = BRIDGE_MAX_DIST_M

    cfg = get_da3_config()
    da3 = get_da3()
    try:
        with tempfile.TemporaryDirectory() as views_base:
            def bridge_test_edge(path_a, path_b, test_id):
                return da3_bridge_test_edge(path_a, path_b, cfg, views_base, da3, test_id=test_id, step_degrees=step_degrees)

            return join_segments(segments, bridge_test_edge, edge_max_dist_m=edge_max_dist_m,
                                  chunk_ids=chunk_ids, known_adjacent_chunk_pairs=known_adjacent_chunk_pairs,
                                  refetch_path=lambda key, lat, lon: path_by_key.get(key))
    finally:
        torch.cuda.empty_cache()


def bridge_incremental_gpu(existing_pieces, existing_ids, new_segments, new_chunk_id,
                            adjacent_ids, edge_max_dist_m=None, step_degrees=20):
    """See _bridge_incremental_impl for the real docstring -- this is just
    the thin dispatch wrapper (see this module's own docstring for why).
    Prefetches every candidate pano (see _prefetch_bridge_panos) here,
    OUTSIDE the GPU call, before dispatching."""
    path_by_key = _prefetch_bridge_panos(_frame_pose_positions(list(existing_pieces) + list(new_segments)))
    return _gpu_dispatch("bridge_incremental", existing_pieces, existing_ids, new_segments, new_chunk_id,
                          adjacent_ids, path_by_key, edge_max_dist_m=edge_max_dist_m, step_degrees=step_degrees)


def _bridge_incremental_impl(existing_pieces, existing_ids, new_segments, new_chunk_id,
                              adjacent_ids, path_by_key, edge_max_dist_m=None, step_degrees=20):
    """Bridges ONE new chunk's segments onto whatever's already been
    merged so far, WITHOUT re-verifying pairs already merged in a
    previous call -- unlike _join_segments_impl (which always re-bridges
    from scratch over its whole input every time), this is the
    genuinely-incremental version: existing_pieces/existing_ids are
    exactly what a previous call (to this function, or the first
    new-chunk call with existing_pieces=[]) returned, fed straight back
    in. Restricting known_adjacent_chunk_pairs to (new_chunk_id, x) for x
    in adjacent_ids means an existing piece only ever gets tested against
    the genuinely new segments, never against other existing pieces
    (those pairs, if any, were already resolved -- or deliberately never
    attempted -- in whichever earlier call produced existing_pieces).

    existing_pieces: [] on the very first chunk, otherwise the `pieces`
    this function itself returned last time. existing_ids: parallel list
    of each existing piece's own (possibly multi-chunk) id list, as
    returned by bridge_pieces' return_ids=True. new_segments: this
    chunk's own run_pathfind_reconstruction_gpu output, chunk_id
    new_chunk_id. adjacent_ids: every existing chunk id (NOT necessarily
    matching existing_pieces' current ids 1:1, since those may already be
    merged unions) that the corridor's real graph says new_chunk_id
    touches -- from the caller's own known_adjacent_chunk_pairs (see
    street_builder.map_selection.candidates.split_into_chunks).

    Returns (pieces, id_sets) -- same shape as existing_pieces/
    existing_ids, ready to feed straight back into the next call. Once no
    more chunks are coming, convert the final pieces via
    join_segments.pieces_to_output (lossy, one-way -- see its own
    docstring) instead of calling this again.

    path_by_key: {key: path or None} -- prefetched by bridge_incremental_gpu
    BEFORE this GPU call (see _prefetch_bridge_panos); refetching used to
    happen live, INSIDE this GPU session, a plain dict lookup here instead."""
    import tempfile
    import time

    import torch
    from services.da3_ops import bridge_test_edge as da3_bridge_test_edge
    from street_builder.reconstruction.join_segments import BRIDGE_MAX_DIST_M, bridge_pieces

    if edge_max_dist_m is None:
        edge_max_dist_m = BRIDGE_MAX_DIST_M

    segments = list(existing_pieces) + list(new_segments)
    chunk_ids = list(existing_ids) + [new_chunk_id] * len(new_segments)
    known_adjacent_chunk_pairs = [(new_chunk_id, other) for other in adjacent_ids]

    cfg = get_da3_config()
    da3 = get_da3()
    try:
        with tempfile.TemporaryDirectory() as views_base:
            def bridge_test_edge(path_a, path_b, test_id):
                return da3_bridge_test_edge(path_a, path_b, cfg, views_base, da3, test_id=test_id, step_degrees=step_degrees)

            deadline = time.monotonic() + 200.0
            return bridge_pieces(segments, bridge_test_edge, edge_max_dist_m=edge_max_dist_m, deadline=deadline,
                                  chunk_ids=chunk_ids, known_adjacent_chunk_pairs=known_adjacent_chunk_pairs,
                                  refetch_path=lambda key, lat, lon: path_by_key.get(key), return_ids=True)
    finally:
        torch.cuda.empty_cache()


def _frame_pose_positions_meta(meta_pieces):
    """{key: (lat, lon)} across every given META piece's own frame_poses --
    identical purpose to _frame_pose_positions, just adapted for meta
    pieces' own 5-tuple shape (leaf_refs, path_edges, date, reached,
    frame_poses), where frame_poses lives at index 4, not 6."""
    positions = {}
    for mp in meta_pieces:
        frame_poses = mp[4]
        for key, (_, _, _, lat, lon, _, _) in frame_poses.items():
            positions[key] = (lat, lon)
    return positions


def bridge_metadata_gpu(meta_pieces, chunk_ids, known_adjacent_chunk_pairs, edge_max_dist_m=None, step_degrees=20):
    """See _bridge_metadata_impl for the real docstring -- this is just the
    thin dispatch wrapper (see this module's own docstring for why).
    Prefetches every candidate pano (see _prefetch_bridge_panos) here,
    OUTSIDE the GPU call, before dispatching -- same reasoning as
    join_segments_gpu/bridge_incremental_gpu, just over meta pieces'
    frame_poses instead of real segments'."""
    path_by_key = _prefetch_bridge_panos(_frame_pose_positions_meta(meta_pieces))
    return _gpu_dispatch("bridge_metadata", meta_pieces, chunk_ids, known_adjacent_chunk_pairs, path_by_key,
                          edge_max_dist_m=edge_max_dist_m, step_degrees=step_degrees)


def _bridge_metadata_impl(meta_pieces, chunk_ids, known_adjacent_chunk_pairs, path_by_key, edge_max_dist_m=None, step_degrees=20):
    """Metadata-only sibling of _bridge_incremental_impl -- bridges a batch
    of meta pieces (see join_segments.bridge_metadata) using real DA3
    tests, restricted to known_adjacent_chunk_pairs exactly like
    _bridge_incremental_impl (same reasoning: this runs at large-scale,
    many-group-in-a-tree scope, where a blind all-pairs scan risks
    wrongly bridging two groups that are merely geographically close but
    not actually meant to connect). Used by street_builder/tab.py's
    handle_cli_merge_group -- one binary-tree merge step, combining two
    groups' worth of meta pieces (each group itself possibly still
    several un-bridged fragments) into one new set of meta pieces, never
    touching point-cloud-sized data (see join_segments.py's own module
    docstring for the whole metadata-only design).

    Returns (pieces, id_sets) -- same shape as bridge_incremental_gpu,
    just meta pieces instead of real segments."""
    import tempfile
    import time

    import torch
    from services.da3_ops import bridge_test_edge as da3_bridge_test_edge
    from street_builder.reconstruction.join_segments import BRIDGE_MAX_DIST_M, bridge_metadata

    if edge_max_dist_m is None:
        edge_max_dist_m = BRIDGE_MAX_DIST_M

    cfg = get_da3_config()
    da3 = get_da3()
    try:
        with tempfile.TemporaryDirectory() as views_base:
            def bridge_test_edge(path_a, path_b, test_id):
                return da3_bridge_test_edge(path_a, path_b, cfg, views_base, da3, test_id=test_id, step_degrees=step_degrees)

            deadline = time.monotonic() + 200.0
            return bridge_metadata(meta_pieces, bridge_test_edge, edge_max_dist_m=edge_max_dist_m, deadline=deadline,
                                    chunk_ids=chunk_ids, known_adjacent_chunk_pairs=known_adjacent_chunk_pairs,
                                    refetch_path=lambda key, lat, lon: path_by_key.get(key), return_ids=True)
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
    hard_deadline = t0 + GPU_WINDOWED_DURATION_S - SAVE_BUFFER_S

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
            pieces = join_segments(segments, bridge_test_edge, edge_max_dist_m=edge_max_dist_m, max_time_budget_s=remaining_s,
                                    raise_on_unsatisfied=False)
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
