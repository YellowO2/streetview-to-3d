"""Shared helpers used by multiple reconstruction strategies."""
import asyncio
import os
from collections import namedtuple

from services.geo import haversine_m
from services.lookaround_fetch import apple_candidates, download_lookaround
from services.streetview_fetch import download_images_for_nodes
from street_builder.map_selection.candidates import APPLE_CANDIDATE_MAX_DIST_M

# lat/lon needed to pick which of a window's winners are closest to a
# window boundary (windowed.py); best4.py only uses label/path.
Candidate = namedtuple("Candidate", ["label", "path", "lat", "lon"])

# Per node, how many nearest Apple Look Around panos to pull in as extra
# support context or scoring candidates.
APPLE_SUPPORT_PER_NODE = 1
CANDIDATE_POOL_APPLE_PER_NODE = 4


def gather_apple_support(nodes: list[dict]) -> list[str]:
    """Closest APPLE_SUPPORT_PER_NODE Look Around pano(s) per node, downloaded
    and stitched to equirectangular. Best-effort per node."""
    paths = []
    for node in nodes:
        try:
            candidates = apple_candidates(node["lat"], node["lon"], k=APPLE_SUPPORT_PER_NODE)
            for pano in candidates:
                print(f"Downloading Apple support pano for node {node['id']}: {pano.id}")
                paths.append(download_lookaround(pano))
        except Exception as e:
            print(f"Apple support lookup failed for node {node['id']}: {e}")
    return paths


def download_chain_and_support(nodes: list[dict]) -> tuple[str, list[str]]:
    """Download the chain's own images plus per-node Apple support panos.
    Returns (target_depth_path, support_paths)."""
    if len(nodes) < 2:
        raise ValueError("Need at least 2 nodes in the chain for DA3 to have multi-view context.")
    image_paths = asyncio.run(download_images_for_nodes(nodes))
    support_paths = image_paths[1:] + gather_apple_support(nodes)
    return image_paths[0], support_paths


def gather_candidate_pool(nodes: list[dict]) -> list[Candidate]:
    """Candidates for every given node's own Google image plus nearby Apple
    candidates. The nodes only mark where to search; they compete in this
    pool and aren't guaranteed a spot in the final reconstruction."""
    pool = []
    seen_paths = set()

    image_paths = asyncio.run(download_images_for_nodes(nodes))
    for node, path in zip(nodes, image_paths):
        if path not in seen_paths:
            pool.append(Candidate(f"google:{node['id']}", path, node["lat"], node["lon"]))
            seen_paths.add(path)

    for node in nodes:
        try:
            candidates = apple_candidates(node["lat"], node["lon"], k=CANDIDATE_POOL_APPLE_PER_NODE)
        except Exception as e:
            print(f"Apple candidate lookup failed for node {node['id']}: {e}")
            continue
        for pano in candidates:
            dist = haversine_m(node["lat"], node["lon"], pano.lat, pano.lon)
            if dist > APPLE_CANDIDATE_MAX_DIST_M:
                print(f"Apple candidate {pano.id} skipped: {dist:.1f}m from node {node['id']} (> {APPLE_CANDIDATE_MAX_DIST_M}m cap)")
                continue
            try:
                path = download_lookaround(pano)
            except Exception as e:
                print(f"Apple candidate download failed for {pano.id}: {e}")
                continue
            if path not in seen_paths:
                pool.append(Candidate(f"apple:{pano.id}", path, pano.lat, pano.lon))
                seen_paths.add(path)

    return pool


def labeled_alias(candidate: Candidate, alias_dir: str) -> str:
    """Symlink to candidate.path named by source + lat/lon instead of its
    opaque pano ID. panoramic-to-3dgs's DA3 log identifies a pano by
    os.path.basename(path), so this is what actually shows up there."""
    source = candidate.label.split(":", 1)[0]
    ext = os.path.splitext(candidate.path)[1]
    alias_path = os.path.join(alias_dir, f"{source}_lat{candidate.lat:.6f}_lon{candidate.lon:.6f}{ext}")
    if not os.path.exists(alias_path):
        os.symlink(os.path.abspath(candidate.path), alias_path)
    return alias_path
