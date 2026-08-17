"""Chunk + connect: for chains too long for one DA3 call. Splits the chain
into overlapping windows and hands the whole multi-window job to
Pipeline.run_windowed_reconstruction in ONE GPU call (splitting into
separate per-window GPU calls hit ZeroGPU's proxy-token lifetime in
testing -- 'Expired ZeroGPU proxy token').
"""
import os

from services.pipeline_runner import run_windowed_reconstruction_gpu, save_pointcloud
from street_builder.reconstruction.best4 import BEST4_FINAL_COUNT, reconstruct_chain_best4
from street_builder.reconstruction.common import gather_candidate_pool

WINDOW_NODE_SIZE = 2
WINDOW_STRIDE = 1
WINDOW_FORCED_OVERLAP = 2


def _chain_windows(nodes: list[dict], size: int = WINDOW_NODE_SIZE, stride: int = WINDOW_STRIDE) -> list[list[dict]]:
    """[A,B,C,D] with size=2, stride=1 -> [[A,B], [B,C], [C,D]]."""
    return [nodes[i:i + size] for i in range(0, len(nodes) - size + 1, stride)]


def reconstruct_chain_windowed(nodes: list[dict], output_dir: str) -> str:
    """Falls back to reconstruct_chain_best4 for a 2-node chain (nothing to stitch)."""
    if len(nodes) < 2:
        raise ValueError("Need at least 2 nodes in the chain for DA3 to have multi-view context.")
    if len(nodes) == 2:
        return reconstruct_chain_best4(nodes, output_dir)

    windows = _chain_windows(nodes)
    pools = [gather_candidate_pool(w) for w in windows]
    for i, (pool, window_nodes) in enumerate(zip(pools, windows)):
        if len(pool) < 2:
            raise ValueError(f"Window {i} ({[n['id'] for n in window_nodes]}) has too few candidates to score.")

    # Plain tuples, not Candidate namedtuples -- keeps the ZeroGPU payload simple.
    pool_tuples = [[tuple(c) for c in pool] for pool in pools]
    # Windows overlap by one raw node (WINDOW_STRIDE=1): the shared node
    # between window i and i+1 is always windows[i+1][0].
    boundary_coords = [(windows[i + 1][0]["lat"], windows[i + 1][0]["lon"]) for i in range(len(windows) - 1)]

    pts, cols = run_windowed_reconstruction_gpu(
        pool_tuples, boundary_coords, final_count=BEST4_FINAL_COUNT, forced_overlap=WINDOW_FORCED_OVERLAP
    )

    os.makedirs(output_dir, exist_ok=True)
    return save_pointcloud(pts, cols, os.path.join(output_dir, "da3_pointcloud.ply"))
