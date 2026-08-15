"""Turns a street-builder chain into an actual point cloud: download the real
panorama images for the selected nodes and run a single DA3 pass on the
whole chain jointly. Called directly by the Generate button in tab.py.

No Apple support panos and no windowing yet -- this reconstructs the whole
selected chain in one DA3 call. Fine for short chains; long chains will need
splitting into overlapping windows stitched back together (future work).

Requires a CUDA GPU (via the panoramic_to_3dgs/depth_anything_3/sharp
dependencies) -- not runnable on this machine locally. Verified here only as
far as syntax/imports; the actual generation needs to be tried on HF Spaces
or a GPU box.
"""
import asyncio

from services.pipeline_runner import run_pointcloud_gpu
from services.streetview_fetch import download_images_for_nodes


def reconstruct_chain(nodes: list[dict], output_dir: str) -> str:
    """Download the chain's images and run one joint DA3 pass over all of
    them (first node as target, rest as support -- functionally symmetric,
    DA3 reconstructs them jointly regardless of which one is nominally
    "target"; only the target's pose gets used as the output's origin).

    Returns the path to the merged da3_pointcloud.ply.
    """
    if len(nodes) < 2:
        raise ValueError("Need at least 2 nodes in the chain for DA3 to have multi-view context.")

    # Same download helper (services/streetview_fetch.py) app.py's
    # single-pano tab uses for its support panos -- not a separate copy.
    image_paths = asyncio.run(download_images_for_nodes(nodes))

    # Reuses app.py's own pipeline runner/singleton (services/pipeline_runner.py)
    # rather than loading a second separate copy of the DA3/SHARP models.
    ply_path = run_pointcloud_gpu(
        target_depth_path=image_paths[0],
        output_dir=output_dir,
        support_paths=image_paths[1:],
    )
    if not ply_path:
        raise RuntimeError("Pipeline finished but no point cloud was produced.")
    return ply_path
