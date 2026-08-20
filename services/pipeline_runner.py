"""GPU-wrapped pipeline runners: SHARP/DA3 3DGS generation, DA3-only point
cloud generation, and the Flux image editor. Also owns the ZeroGPU/@spaces.GPU
decorator setup, since that setup exists purely to wrap these calls.

get_pipeline() is the single Pipeline singleton for this whole app -- both
app.py's Gradio handlers and street_builder's own handlers call through
here, rather than each loading a separate copy of the DA3/SHARP models.
"""
import os

try:
    import spaces

    # spaces is also installed locally via requirements.txt, so gate on SPACE_ID
    # which HF Spaces always sets but local machines don't have.
    ON_SPACES = bool(os.getenv("SPACE_ID"))
    # Longer budget for run_pathfind_reconstruction_gpu: it does the entire
    # multi-date pathfind search (map_date/set_cover, potentially dozens of
    # DA3 tests) inside one GPU call, specifically to avoid the 'Expired
    # ZeroGPU proxy token' failure hit when that was split into several
    # smaller calls instead. PATHFIND_MAX_TIME_BUDGET_S is a CEILING passed
    # to run_pathfind_reconstruction, not its actual deadline -- the
    # algorithm scales its own deadline to corridor size and only caps at
    # this. Kept well under GPU_WINDOWED_DURATION_S to leave real margin
    # for model load (before the search even starts) and save/teardown after.
    GPU_WINDOWED_DURATION_S = 280
    PATHFIND_MAX_TIME_BUDGET_S = GPU_WINDOWED_DURATION_S - 60

    if ON_SPACES:
        GPU = spaces.GPU(duration=108)
        GPU_EDIT = spaces.GPU(duration=72)
        GPU_WINDOWED = spaces.GPU(duration=GPU_WINDOWED_DURATION_S)
    else:
        GPU = lambda fn: fn
        GPU_EDIT = lambda fn: fn
        GPU_WINDOWED = lambda fn: fn
except ImportError:
    GPU = lambda fn: fn  # no-op outside HF Spaces
    GPU_EDIT = lambda fn: fn
    GPU_WINDOWED = lambda fn: fn
    ON_SPACES = False
    GPU_WINDOWED_DURATION_S = 280
    PATHFIND_MAX_TIME_BUDGET_S = GPU_WINDOWED_DURATION_S - 60

_pipeline = None
_flux_editor = None
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


@GPU_EDIT
def run_editor_gpu(image_path, prompt, mode, output_path):
    global _flux_editor
    if _flux_editor is None:
        from editors.flux_editor import FluxEditor
        _flux_editor = FluxEditor(offload=True)
    _flux_editor.edit(image_path, prompt, mode=mode, output_path=output_path)
    return output_path


@GPU
def run_pipeline_gpu(target_appearance_path, output_dir, scale_mode, gs_backend, support_paths=None, target_depth_path=None):
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


@GPU
def run_pointcloud_gpu(target_depth_path, output_dir, support_paths=None, step_degrees=20):
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


@GPU_WINDOWED
def run_pathfind_reconstruction_gpu(date_graphs, points, adjacency, start_lat, start_lon, step_degrees=20):
    """The one @spaces.GPU call for the whole pathfind flow (see
    street_builder/main.py's module docstring for why it's one call, not
    several). This function's only job is to own the loaded DA3Model for
    the session and hand the actual algorithm
    (street_builder/reconstruction/walk_graph.py) a way to test one edge
    -- it knows nothing about corridors, dates, or coverage itself.

    date_graphs: already ranked/capped/isolated per date, dot_candidates
    shape -- see street_builder/build_graph/build_graph.py's
    build_corridor_graphs. adjacency: the corridor's shared dot-to-dot
    structural graph, same source."""
    import itertools
    import tempfile

    import torch
    from panoramic_to_3dgs import DA3Model, score_pano_da3, test_edge_da3, test_edge_da3_bridge
    from street_builder.reconstruction.walk_graph import run_pathfind_reconstruction

    pipeline = get_pipeline()
    da3 = DA3Model(pipeline.config.da3_model)
    try:
        with tempfile.TemporaryDirectory() as views_base:
            def test_edge(path_a, path_b, test_id):
                return test_edge_da3(path_a, path_b, pipeline.config, views_base, da3, test_id=test_id, step_degrees=step_degrees)

            score_ids = itertools.count()

            def score_pano(path):
                return score_pano_da3(path, pipeline.config, views_base, da3, score_id=next(score_ids), step_degrees=step_degrees)

            def bridge_test_edge(path_a, path_b, test_id):
                return test_edge_da3_bridge(path_a, path_b, pipeline.config, views_base, da3, test_id=test_id, step_degrees=step_degrees)

            return run_pathfind_reconstruction(date_graphs, points, adjacency, start_lat, start_lon, test_edge,
                                                score_pano=score_pano, bridge_test_edge=bridge_test_edge,
                                                max_time_budget_s=PATHFIND_MAX_TIME_BUDGET_S)
    finally:
        del da3
        torch.cuda.empty_cache()


def save_pointcloud(points, colors, path):
    """Not GPU-wrapped -- pure disk I/O (open3d write), no CUDA involved.
    Lazy import to match get_pipeline()'s pattern, so this module still
    imports cleanly on machines without panoramic_to_3dgs installed."""
    from panoramic_to_3dgs import save_da3_pointcloud
    return save_da3_pointcloud(points, colors, path)
