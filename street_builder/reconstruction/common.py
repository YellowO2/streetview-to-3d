"""Shared helpers used by multiple reconstruction strategies."""
import asyncio
from collections import namedtuple

from services.lookaround_fetch import DA3_ONLY_APPLE_ZOOM, apple_candidates, download_lookaround
from services.streetview_fetch import DA3_ONLY_ZOOM, download_images_for_nodes

# street_builder is DA3-only everywhere (no SHARP splat generation), so
# every download here uses the low-res DA3 zoom, not the SHARP default.

Candidate = namedtuple("Candidate", ["label", "path", "lat", "lon"])

# Per node, how many nearest Apple Look Around panos to pull in as extra
# support context or scoring candidates.
APPLE_SUPPORT_PER_NODE = 1
CANDIDATE_POOL_APPLE_PER_NODE = 4

# Yaw step for DA3's view slicing, shared default across the client-driven
# reconstruction flows. 30 (12 slices) is the tested middle ground between
# DA3's own default 20 (18 slices) and the too-coarse 45 (8 slices, caused
# 2/4 winners to go from partial acceptance to fully rejected in an
# earlier scoring experiment).
DEFAULT_STEP_DEGREES = 30


def gather_apple_support(nodes: list[dict]) -> list[str]:
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


def download_chain_and_support(nodes: list[dict]) -> tuple[str, list[str]]:
    """Download the chain's own images plus per-node Apple support panos.
    Returns (target_depth_path, support_paths)."""
    if len(nodes) < 2:
        raise ValueError("Need at least 2 nodes in the chain for DA3 to have multi-view context.")
    image_paths = asyncio.run(download_images_for_nodes(nodes, zoom=DA3_ONLY_ZOOM))
    support_paths = image_paths[1:] + gather_apple_support(nodes)
    return image_paths[0], support_paths


