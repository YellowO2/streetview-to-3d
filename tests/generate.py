"""Simple strategy: one joint DA3 pass over the chain's own images plus
per-node Apple support. Called by the normal "Generate" button."""
import asyncio

from services.lookaround_fetch import DA3_ONLY_APPLE_ZOOM, apple_candidates, download_lookaround
from services.pipeline_runner import run_pointcloud_gpu
from services.streetview_fetch import DA3_ONLY_ZOOM, download_images_for_nodes

# Per node, how many nearest Apple Look Around panos to pull in as extra
# support context. street_builder is DA3-only everywhere (no SHARP splat
# generation), so this uses the low-res DA3 zoom, not the SHARP default.
APPLE_SUPPORT_PER_NODE = 1


def _gather_apple_support(nodes: list[dict]) -> list[str]:
    """Closest APPLE_SUPPORT_PER_NODE Look Around pano(s) per node, downloaded
    and stitched to equirectangular. Best-effort per node."""
    paths = []
    for node in nodes:
        try:
            candidates = apple_candidates(node["lat"], node["lon"], k=APPLE_SUPPORT_PER_NODE)
            for pano in candidates:
                print(f"Downloading Apple support pano for node {node['id']}: {pano.id}")
                paths.append(download_lookaround(pano, zoom=DA3_ONLY_APPLE_ZOOM))
        except Exception as e:
            print(f"Apple support lookup failed for node {node['id']}: {e}")
    return paths


def reconstruct_chain(nodes: list[dict], output_dir: str) -> str:
    """Download the chain's images plus per-node Apple support panos, and run
    one joint DA3 pass over all of them. Returns the path to the merged ply."""
    if len(nodes) < 2:
        raise ValueError("Need at least 2 nodes in the chain for DA3 to have multi-view context.")
    image_paths = asyncio.run(download_images_for_nodes(nodes, zoom=DA3_ONLY_ZOOM))
    target_depth_path = image_paths[0]
    support_paths = image_paths[1:] + _gather_apple_support(nodes)

    ply_path = run_pointcloud_gpu(
        target_depth_path=target_depth_path,
        output_dir=output_dir,
        support_paths=support_paths,
    )
    if not ply_path:
        raise RuntimeError("Pipeline finished but no point cloud was produced.")
    return ply_path
