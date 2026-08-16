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
import os
from collections import namedtuple

from services.geo import haversine_m
from services.lookaround_fetch import apple_candidates, download_lookaround
from services.pipeline_runner import (
    run_pointcloud_gpu,
    run_pointcloud_sweep_gpu,
    run_windowed_reconstruction_full_pool_gpu,
    run_windowed_reconstruction_gpu,
    save_pointcloud,
    score_candidates_gpu,
)
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

# Yaw step for extract_views_for_da3's slicing (default 20 -> 18 slices/pano,
# ~78% overlap between neighbors). Used as reconstruct_chain_best4's default
# step_degrees (comparing against the original 20-degree result) and for the
# full-pool experiments, where a coarser step is needed to keep the
# unfiltered pool's image count down.
FULL_POOL_STEP_DEGREES = 45

# _gather_candidate_pool's Apple lookup is nearest-K only, with no cap on how
# far "nearest" might actually be if local Apple coverage is sparse -- this
# bounds it, so a pool built for one window can't reach into territory well
# outside that window's own span. ~2.5x the ~10m consecutive-node spacing
# we've measured on our one real test street; sparser streets may need this
# retuned, but there's no data yet to justify a fancier per-street estimate.
APPLE_CANDIDATE_MAX_DIST_M = 25.0

# reconstruct_chain_windowed: raw chain nodes per window's own candidate pool
# (WINDOW_NODE_SIZE), how many raw nodes consecutive windows overlap by
# (WINDOW_STRIDE controls this: overlap = WINDOW_NODE_SIZE - WINDOW_STRIDE),
# and of each window's BEST4_FINAL_COUNT final picks, how many are forced to
# carry over from the previous window (rather than freshly picked) so the two
# windows share a literal identical image to rigid-align on.
WINDOW_NODE_SIZE = 2
WINDOW_STRIDE = 1
WINDOW_FORCED_OVERLAP = 2

# (label, path, lat, lon) -- lat/lon needed by reconstruct_chain_windowed to
# pick which of a window's winners are "closest to the boundary" with the
# next window; reconstruct_chain_best4 only uses label/path.
Candidate = namedtuple("Candidate", ["label", "path", "lat", "lon"])

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


def _gather_candidate_pool(nodes: list[dict]) -> list[Candidate]:
    """Candidates for every given node's own Google image plus nearby Apple
    candidates -- the pool reconstruct_chain_best4 (and, per-window,
    reconstruct_chain_windowed) scores and ranks, rather than trusting the
    chain nodes or closest-K blindly. The given Google nodes only mark where
    to search; they compete in this same pool and aren't guaranteed a spot in
    the final reconstruction."""
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


def _score_and_rank(pool: list[Candidate]) -> list[Candidate]:
    """Solo-score every candidate in the pool and return them sorted
    best-first. Shared by reconstruct_chain_best4 and reconstruct_chain_windowed
    (one window's pool, in the latter case)."""
    scores = score_candidates_gpu([c.path for c in pool])
    ranked = [c for c, _ in sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)]
    print(f"Candidate scores (label, keep-count/18): {list(zip((c.label for c in pool), scores))}")
    return ranked


def reconstruct_chain_best4(nodes: list[dict], output_dir: str, step_degrees: int = FULL_POOL_STEP_DEGREES) -> str:
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

    step_degrees only affects the final reconstruction call, not the solo-
    scoring pass (which stays at DA3Model's own default) -- currently
    defaulted to FULL_POOL_STEP_DEGREES (45) rather than DA3's usual 20, to
    directly compare against the earlier step=20 best-4 result (36/72 kept)
    and check whether coarser slicing holds up as well on the same winners.
    Pass step_degrees=20 explicitly to go back to the original behavior.
    """
    pool = _gather_candidate_pool(nodes)
    if len(pool) < 2:
        raise ValueError("Need at least 2 candidate panos (chain nodes + Apple support) to score.")

    ranked = _score_and_rank(pool)
    winners = ranked[:BEST4_FINAL_COUNT]
    if len(winners) < 2:
        raise ValueError("Not enough candidates survived scoring for multi-view reconstruction.")

    print(f"Reconstructing with top {len(winners)} (step={step_degrees}): {[c.label for c in winners]}")
    ply_path = run_pointcloud_gpu(
        target_depth_path=winners[0].path,
        output_dir=output_dir,
        support_paths=[c.path for c in winners[1:]],
        step_degrees=step_degrees,
    )
    if not ply_path:
        raise RuntimeError("Pipeline finished but no point cloud was produced.")
    return ply_path


def _chain_windows(nodes: list[dict], size: int = WINDOW_NODE_SIZE, stride: int = WINDOW_STRIDE) -> list[list[dict]]:
    """Slice the ordered chain into overlapping raw-node windows, e.g.
    [A,B,C,D] with size=2, stride=1 -> [[A,B], [B,C], [C,D]]."""
    return [nodes[i:i + size] for i in range(0, len(nodes) - size + 1, stride)]


def reconstruct_chain_windowed(nodes: list[dict], output_dir: str) -> str:
    """Chunk + connect, for chains too long to reconstruct in a single DA3
    call: splits the chain into overlapping raw-node windows (_chain_windows),
    gathers each window's own candidate pool (chain nodes + nearby Apple
    panos, same as reconstruct_chain_best4's pool), and hands the whole
    multi-window job to Pipeline.run_windowed_reconstruction in ONE GPU call.

    That single call does, per window: scoring every candidate solo, forced-
    overlap selection (WINDOW_FORCED_OVERLAP of each window's picks are
    forced to be the previous window's own winners closest to the shared
    boundary node -- see run_windowed_reconstruction's docstring for why),
    DA3 reconstruction, and rigid alignment onto a running global frame.

    The consolidation into one call is necessary, not just an optimization:
    an earlier version called separate small GPU functions once per window
    (mirroring reconstruct_chain_best4's own score-then-reconstruct split),
    and that hit ZeroGPU's proxy-token lifetime after only 2 windows (4
    sequential @spaces.GPU calls) in real testing -- 'Expired ZeroGPU proxy
    token'. Every @spaces.GPU call is a fresh GPU acquisition and, since
    Pipeline doesn't cache a DA3Model across calls, a fresh ~35GB model
    reload; one call per whole job avoids paying that N times over.

    Falls back to reconstruct_chain_best4 directly for a 2-node chain (a
    single window -- nothing to stitch).
    """
    if len(nodes) < 2:
        raise ValueError("Need at least 2 nodes in the chain for DA3 to have multi-view context.")
    if len(nodes) == 2:
        return reconstruct_chain_best4(nodes, output_dir)

    windows = _chain_windows(nodes)
    pools = [_gather_candidate_pool(w) for w in windows]
    for i, (pool, window_nodes) in enumerate(zip(pools, windows)):
        if len(pool) < 2:
            raise ValueError(f"Window {i} ({[n['id'] for n in window_nodes]}) has too few candidates to score.")

    # Plain tuples, not Candidate namedtuples -- keeps the pickled payload
    # across the ZeroGPU call boundary simple/robust.
    pool_tuples = [[tuple(c) for c in pool] for pool in pools]
    # Windows overlap by one raw node (WINDOW_STRIDE=1): the node shared
    # between window i and window i+1 is always windows[i+1][0].
    boundary_coords = [(windows[i + 1][0]["lat"], windows[i + 1][0]["lon"]) for i in range(len(windows) - 1)]

    pts, cols = run_windowed_reconstruction_gpu(
        pool_tuples, boundary_coords, final_count=BEST4_FINAL_COUNT, forced_overlap=WINDOW_FORCED_OVERLAP
    )

    os.makedirs(output_dir, exist_ok=True)
    return save_pointcloud(pts, cols, os.path.join(output_dir, "da3_pointcloud.ply"))


def reconstruct_chain_windowed_full_pool(nodes: list[dict], output_dir: str) -> str:
    """Chunk + connect, same windowing as reconstruct_chain_windowed, but the
    opposite bet on what goes into each window: no solo-scoring, no best4
    down-selection -- every window's FULL local candidate pool (chain nodes +
    nearby Apple panos, same _gather_candidate_pool as reconstruct_chain_best4)
    goes into that window's DA3 call.

    Tests whether the quality problems best4 was built to fix were really
    about distance (which windowing already fixes structurally, since a
    window's pool only spans ~1 raw node's worth of range -- see
    APPLE_CANDIDATE_MAX_DIST_M) rather than genuine per-pano capture-quality
    variance that scoring was filtering out. We have direct evidence of the
    latter too (the raw chain nodes solo-scored 5/18 and 2/18 on their own,
    nothing to do with distance from other panos), so this is a real open
    question, not a foregone conclusion either way.

    Uses FULL_POOL_STEP_DEGREES (45, vs. DA3's own default of 20) to keep the
    image count reasonable despite not down-selecting.

    Alignment between adjacent windows doesn't need an explicit forced-
    carryover step here (unlike reconstruct_chain_windowed): windows overlap
    by one raw chain node, and since both windows independently gather that
    same node's own image + its nearest-K Apple candidates, those candidates
    already show up in both windows' full pools automatically. See
    Pipeline.run_windowed_reconstruction_full_pool for how that natural
    overlap gets used.

    Falls back to a single non-windowed DA3 call directly for a 2-node chain
    (a single window -- nothing to stitch).
    """
    if len(nodes) < 2:
        raise ValueError("Need at least 2 nodes in the chain for DA3 to have multi-view context.")

    if len(nodes) == 2:
        pool = _gather_candidate_pool(nodes)
        if len(pool) < 2:
            raise ValueError("Need at least 2 candidate panos (chain nodes + Apple support) to reconstruct.")
        print(f"Reconstructing with full pool ({len(pool)} candidates, step={FULL_POOL_STEP_DEGREES}): {[c.label for c in pool]}")
        ply_path = run_pointcloud_gpu(
            target_depth_path=pool[0].path,
            output_dir=output_dir,
            support_paths=[c.path for c in pool[1:]],
            step_degrees=FULL_POOL_STEP_DEGREES,
        )
        if not ply_path:
            raise RuntimeError("Pipeline finished but no point cloud was produced.")
        return ply_path

    windows = _chain_windows(nodes)
    pools = [_gather_candidate_pool(w) for w in windows]
    for i, (pool, window_nodes) in enumerate(zip(pools, windows)):
        if len(pool) < 2:
            raise ValueError(f"Window {i} ({[n['id'] for n in window_nodes]}) has too few candidates to reconstruct.")

    pool_tuples = [[tuple(c) for c in pool] for pool in pools]
    pts, cols = run_windowed_reconstruction_full_pool_gpu(pool_tuples, step_degrees=FULL_POOL_STEP_DEGREES)

    os.makedirs(output_dir, exist_ok=True)
    return save_pointcloud(pts, cols, os.path.join(output_dir, "da3_pointcloud.ply"))
