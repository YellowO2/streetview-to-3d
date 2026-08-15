"""Turns a street-builder chain into an actual point cloud: download the real
panorama images for the selected nodes, plus one closest Apple Look Around
support pano per node, and run a single DA3 pass on the whole batch jointly.
Called directly by the Generate button in tab.py.

No windowing/stitching yet -- this reconstructs the whole selected chain in
one DA3 call. Fine for short chains (that's what's being tested right now);
long chains will need splitting into overlapping windows stitched back
together (future work, deliberately deferred).

Google historical-date variants (other captures at the same node) are not
gathered here yet -- unlike Apple candidates, they don't have a meaningful
"closest" (same lat/lon as the node itself, so distance-based top-K can't
rank them against Apple candidates). Left as a separate follow-up rather than
folded into this same closest-K pick.

Requires a CUDA GPU (via the panoramic_to_3dgs/depth_anything_3/sharp
dependencies) -- not runnable on this machine locally. Verified here only as
far as syntax/imports; the actual generation needs to be tried on HF Spaces
or a GPU box.
"""
import asyncio

from services.lookaround_fetch import apple_candidates, download_lookaround
from services.pipeline_runner import run_pointcloud_gpu, run_pointcloud_sweep_gpu, score_candidates_gpu
from services.streetview_fetch import download_images_for_nodes

# Per node, how many nearest Apple Look Around panos to pull in as extra
# support context. No distance cutoff yet -- closest-K only.
APPLE_SUPPORT_PER_NODE = 1

# reconstruct_chain_best4: how many Apple candidates per node go into the
# scored pool (wider than APPLE_SUPPORT_PER_NODE, since here every candidate
# competes on its own solo score rather than being trusted by distance alone),
# and how many total winners the final DA3 call gets.
CANDIDATE_POOL_APPLE_PER_NODE = 4
BEST4_FINAL_COUNT = 4

# (label, dist_thresh_m, angle_thresh_deg), loosening from DA3Model's current
# default down to no filter at all -- for reconstruct_chain_filter_sweep,
# to check how much the consensus filter is actually doing before deciding
# whether/how to change it.
FILTER_SWEEP_LEVELS = [
    ("Current (0.2m, 1°)", 0.2, 1),
    ("Loose (0.5m, 3°)", 0.5, 3),
    ("Looser (1.0m, 5°)", 1.0, 5),
    ("Very loose (2.0m, 10°)", 2.0, 10),
    ("No filter", float("inf"), float("inf")),
]


def _gather_apple_support(nodes: list[dict]) -> list[str]:
    """Closest APPLE_SUPPORT_PER_NODE Look Around pano(s) per node, downloaded
    and stitched to equirectangular. Best-effort per node -- a lookup/download
    failure for one node's Apple support shouldn't abort the whole chain."""
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


def _download_chain_and_support(nodes: list[dict]) -> tuple[str, list[str]]:
    """Shared by reconstruct_chain and reconstruct_chain_filter_sweep: download
    the chain's own images plus per-node Apple support panos. Returns
    (target_depth_path, support_paths)."""
    if len(nodes) < 2:
        raise ValueError("Need at least 2 nodes in the chain for DA3 to have multi-view context.")

    # Same download helper (services/streetview_fetch.py) app.py's
    # single-pano tab uses for its support panos -- not a separate copy.
    image_paths = asyncio.run(download_images_for_nodes(nodes))
    support_paths = image_paths[1:] + _gather_apple_support(nodes)
    return image_paths[0], support_paths


def reconstruct_chain(nodes: list[dict], output_dir: str) -> str:
    """Download the chain's images plus per-node Apple support panos, and run
    one joint DA3 pass over all of them (first node as target, rest as
    support -- functionally symmetric, DA3 reconstructs them jointly
    regardless of which one is nominally "target"; only the target's pose
    gets used as the output's origin).

    Returns the path to the merged da3_pointcloud.ply.
    """
    target_depth_path, support_paths = _download_chain_and_support(nodes)

    # Reuses app.py's own pipeline runner/singleton (services/pipeline_runner.py)
    # rather than loading a second separate copy of the DA3/SHARP models.
    ply_path = run_pointcloud_gpu(
        target_depth_path=target_depth_path,
        output_dir=output_dir,
        support_paths=support_paths,
    )
    if not ply_path:
        raise RuntimeError("Pipeline finished but no point cloud was produced.")
    return ply_path


def reconstruct_chain_filter_sweep(nodes: list[dict], output_dir: str) -> list[tuple[str, str | None]]:
    """Debug helper for street_builder/tab.py's filter-sweep button: same
    inputs as reconstruct_chain, but runs DA3 inference once and saves one
    point cloud per FILTER_SWEEP_LEVELS threshold instead of a single merged
    result -- to check how much the consensus filter actually matters before
    deciding whether/how to change it.

    Returns a list of (label, ply_path_or_None) pairs, same order as
    FILTER_SWEEP_LEVELS.
    """
    target_depth_path, support_paths = _download_chain_and_support(nodes)

    threshold_levels = [(dist, angle) for _, dist, angle in FILTER_SWEEP_LEVELS]
    out_paths = run_pointcloud_sweep_gpu(
        target_depth_path=target_depth_path,
        output_dir=output_dir,
        threshold_levels=threshold_levels,
        support_paths=support_paths,
    )
    return [(label, path) for (label, _, _), path in zip(FILTER_SWEEP_LEVELS, out_paths)]


def _gather_candidate_pool(nodes: list[dict]) -> list[tuple[str, str]]:
    """(label, image_path) pairs for every chain node's own Google image plus
    nearby Apple candidates -- the pool reconstruct_chain_best4 scores and
    ranks, rather than trusting the chain nodes or closest-K blindly. The
    chain's Google nodes only mark where to search; they compete in this same
    pool and aren't guaranteed a spot in the final reconstruction."""
    pool = []
    seen_paths = set()

    image_paths = asyncio.run(download_images_for_nodes(nodes))
    for node, path in zip(nodes, image_paths):
        if path not in seen_paths:
            pool.append((f"google:{node['id']}", path))
            seen_paths.add(path)

    for node in nodes:
        try:
            candidates = apple_candidates(node["lat"], node["lon"], k=CANDIDATE_POOL_APPLE_PER_NODE)
        except Exception as e:
            print(f"Apple candidate lookup failed for node {node['id']}: {e}")
            continue
        for pano in candidates:
            try:
                path = download_lookaround(pano)
            except Exception as e:
                print(f"Apple candidate download failed for {pano.id}: {e}")
                continue
            if path not in seen_paths:
                pool.append((f"apple:{pano.id}", path))
                seen_paths.add(path)

    return pool


def reconstruct_chain_best4(nodes: list[dict], output_dir: str) -> str:
    """Instead of trusting the chain's own Google nodes (plus closest-K Apple
    support) by default, builds a candidate pool (the chain's Google nodes +
    nearby Apple panos), scores each candidate SOLO through DA3 -- its own
    ~18 view-slices' self-consistency keep-rate, no other pano in the batch
    (see Pipeline.score_candidates) -- and reconstructs using only the
    BEST4_FINAL_COUNT highest-scoring candidates.

    Note this only measures each candidate's own internal coherence, not
    whether it'll correlate well with the others once combined -- two
    individually clean panos that are just too far apart could still both
    score high solo and fail to line up in the final joint DA3 call. That's
    exactly the open question this whole experiment is testing.
    """
    pool = _gather_candidate_pool(nodes)
    if len(pool) < 2:
        raise ValueError("Need at least 2 candidate panos (chain nodes + Apple support) to score.")

    labels, paths = zip(*pool)
    scores = score_candidates_gpu(list(paths))
    ranked = sorted(zip(labels, paths, scores), key=lambda x: x[2], reverse=True)
    print(f"Candidate scores (label, keep-count/18): {[(l, s) for l, _, s in ranked]}")

    winners = ranked[:BEST4_FINAL_COUNT]
    if len(winners) < 2:
        raise ValueError("Not enough candidates survived scoring for multi-view reconstruction.")

    winner_paths = [p for _, p, _ in winners]
    print(f"Reconstructing with top {len(winner_paths)}: {[l for l, _, _ in winners]}")
    ply_path = run_pointcloud_gpu(
        target_depth_path=winner_paths[0],
        output_dir=output_dir,
        support_paths=list(winner_paths[1:]),
    )
    if not ply_path:
        raise RuntimeError("Pipeline finished but no point cloud was produced.")
    return ply_path
