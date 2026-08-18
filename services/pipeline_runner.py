"""GPU-wrapped pipeline runners: SHARP/DA3 3DGS generation, DA3-only point
cloud generation, and the Flux image editor. Also owns the ZeroGPU/@spaces.GPU
decorator setup, since that setup exists purely to wrap these calls.

get_pipeline() is the single Pipeline singleton for this whole app -- both
app.py's Gradio handlers and street_builder/reconstruct.py call through
here, rather than each loading a separate copy of the DA3/SHARP models.
"""
import os

try:
    import spaces

    # spaces is also installed locally via requirements.txt, so gate on SPACE_ID
    # which HF Spaces always sets but local machines don't have.
    ON_SPACES = bool(os.getenv("SPACE_ID"))
    if ON_SPACES:
        GPU = spaces.GPU(duration=108)
        GPU_EDIT = spaces.GPU(duration=72)
        # Longer budget for run_windowed_reconstruction_gpu: it does an
        # entire multi-window chunk+connect job (scoring + reconstruction for
        # every window) inside one GPU call, specifically to avoid the
        # 'Expired ZeroGPU proxy token' failure hit when that was split into
        # several small per-window calls instead.
        GPU_WINDOWED = spaces.GPU(duration=280)
    else:
        GPU = lambda fn: fn
        GPU_EDIT = lambda fn: fn
        GPU_WINDOWED = lambda fn: fn
except ImportError:
    GPU = lambda fn: fn  # no-op outside HF Spaces
    GPU_EDIT = lambda fn: fn
    GPU_WINDOWED = lambda fn: fn
    ON_SPACES = False

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


@GPU
def run_pointcloud_sweep_gpu(target_depth_path, output_dir, threshold_levels, support_paths=None):
    """Debug helper: one DA3 inference pass, one point cloud saved per
    (dist_thresh, angle_thresh) in threshold_levels. See
    Pipeline.run_da3_pointcloud_sweep for why this doesn't cost a GPU
    forward pass per level."""
    pipeline = get_pipeline()
    os.makedirs(output_dir, exist_ok=True)
    return pipeline.run_da3_pointcloud_sweep(
        target_depth_path=target_depth_path,
        output_dir=output_dir,
        threshold_levels=threshold_levels,
        support_paths=support_paths,
    )


@GPU
def score_candidates_gpu(candidate_paths, dist_thresh=0.2, angle_thresh=1, step_degrees=20):
    pipeline = get_pipeline()
    return pipeline.score_candidates(
        candidate_paths, dist_thresh=dist_thresh, angle_thresh=angle_thresh, step_degrees=step_degrees
    )


@GPU_WINDOWED
def run_windowed_reconstruction_gpu(windows, boundary_coords, final_count=4, forced_overlap=2):
    pipeline = get_pipeline()
    return pipeline.run_windowed_reconstruction(
        windows, boundary_coords, final_count=final_count, forced_overlap=forced_overlap
    )


@GPU_WINDOWED
def run_greedy_pass_reconstruction_gpu(
    node_candidates, try_order, keep_rate_threshold=0.5, max_attempts_per_position=3, step_degrees=20
):
    pipeline = get_pipeline()
    return pipeline.run_greedy_pass_reconstruction(
        node_candidates,
        try_order,
        keep_rate_threshold=keep_rate_threshold,
        max_attempts_per_position=max_attempts_per_position,
        step_degrees=step_degrees,
    )


@GPU_WINDOWED
def run_pathfind_reconstruction_gpu(nodes, edges, start_lat, start_lon, goals, step_degrees=20, date_order=None):
    pipeline = get_pipeline()
    return pipeline.run_pathfind_reconstruction(
        nodes, edges, start_lat, start_lon, goals, step_degrees=step_degrees, date_order=date_order
    )


def save_pointcloud(points, colors, path):
    """Not GPU-wrapped -- pure disk I/O (open3d write), no CUDA involved.
    Lazy import to match get_pipeline()'s pattern, so this module still
    imports cleanly on machines without panoramic_to_3dgs installed."""
    from panoramic_to_3dgs import save_da3_pointcloud
    return save_da3_pointcloud(points, colors, path)
