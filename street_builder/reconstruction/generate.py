"""Simple strategy: one joint DA3 pass over the chain's own images plus
per-node Apple support. Called by the normal "Generate" button."""
from services.pipeline_runner import run_pointcloud_gpu
from street_builder.reconstruction.common import download_chain_and_support


def reconstruct_chain(nodes: list[dict], output_dir: str) -> str:
    """Download the chain's images plus per-node Apple support panos, and run
    one joint DA3 pass over all of them. Returns the path to the merged ply."""
    target_depth_path, support_paths = download_chain_and_support(nodes)

    ply_path = run_pointcloud_gpu(
        target_depth_path=target_depth_path,
        output_dir=output_dir,
        support_paths=support_paths,
    )
    if not ply_path:
        raise RuntimeError("Pipeline finished but no point cloud was produced.")
    return ply_path
