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
import tempfile
from collections import namedtuple

from services.geo import haversine_m
from services.lookaround_fetch import apple_candidates, download_lookaround
from services.pipeline_runner import (
    run_greedy_pass_reconstruction_gpu,
    run_pointcloud_gpu,
    run_pointcloud_sweep_gpu,
    run_windowed_reconstruction_gpu,
    save_pointcloud,
    score_candidates_gpu,
)
from services.streetview_fetch import download_images_for_nodes, download_pano_by_id

# Per node, how many nearest Apple Look Around panos to pull in as extra
# support context. No distance cutoff yet -- closest-K only.
APPLE_SUPPORT_PER_NODE = 1

# reconstruct_chain_best4: how many Apple candidates per node go into the
# scored pool (wider than APPLE_SUPPORT_PER_NODE, since here every candidate
# competes on its own solo score rather than being trusted by distance alone),
# and how many total winners the final DA3 call gets.
CANDIDATE_POOL_APPLE_PER_NODE = 4
BEST4_FINAL_COUNT = 4

# Yaw step for extract_views_for_da3's slicing. DA3's own default (20 ->
# 18 slices/pano, ~78% overlap) turned out to matter for more than
# redundancy: at step=45 (8 slices/pano), 2 of 4 best-4 winners went from
# partial acceptance to fully rejected -- fewer slices makes a pano's own
# per-slice consensus noisier, not just less redundant. 30 (12 slices, still
# evenly divides 360) is the middle ground being tested now, used for BOTH
# scoring and final reconstruction so the two stay consistent with each
# other (reconstruct_chain_best4's step_degrees default).
BEST4_STEP_DEGREES = 30

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

# reconstruct_chain_greedy: how many of a node's most-recent Google historical
# captures count as separate "passes" to try (node["dates"] is newest-first,
# so this is just a head-truncation) -- capped for the same reason
# CANDIDATE_POOL_APPLE_PER_NODE is capped: every option here gets downloaded
# eagerly (no per-window lazy fetch inside the GPU call), so this bounds how
# many Google downloads one street selection costs.
GOOGLE_DATE_OPTIONS_PER_NODE = 3

# Fraction of a candidate's own view-slices that must survive DA3's consensus
# filter for a greedy-walk window to count as "healthy". 0.5 matches the
# clean empirical separation observed all session: healthy same-pass pairs
# kept 9-12/12 of their slices, collapsed cross-pass pairs kept 0-1/12 --
# there's been no messy middle ground so far, so this doesn't need tuning.
GREEDY_KEEP_RATE_THRESHOLD = 0.5

# How many candidate passes to try at a given street position before giving
# up and closing out the current segment there. Bounds worst-case GPU calls
# per position rather than exhaustively trying every available pass.
GREEDY_MAX_ATTEMPTS_PER_POSITION = 3

GREEDY_STEP_DEGREES = BEST4_STEP_DEGREES

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


def _pass_options(nodes: list[dict]) -> list[dict[tuple[str, str], object]]:
    """Per node, every available capture pass, metadata only (no downloads):
    Apple candidates grouped by build_id (within APPLE_CANDIDATE_MAX_DIST_M,
    nearest pano per build_id), and Google historical captures grouped by
    date label -- pulled straight off the node's own "dates" field
    (services/streetview_fetch.py's pano_to_meta), so the grouping itself
    costs zero extra network calls. Keyed by (source, pass_key) so passes
    never mix across sources -- only same-source/same-pass compatibility has
    been validated (see this session's build_id findings).

    Values: an Apple `LookaroundPanorama` object for ("apple", build_id)
    keys, or a bare pano ID string for ("google", date_label) keys (Google's
    own metadata/image for a specific historical ID is only fetched once a
    pass is actually chosen to try -- see reconstruct_chain_greedy).

    This is purely a cheap way to decide which pass to *try first*; the
    actual accept/reject decision is always a real pairwise DA3 call on the
    exact candidate pair (see _rank_passes_at's docstring).
    """
    options = []
    for node in nodes:
        node_options = {}

        for entry in node.get("dates", [])[:GOOGLE_DATE_OPTIONS_PER_NODE]:
            node_options[("google", entry["label"])] = entry["id"]

        try:
            candidates = apple_candidates(node["lat"], node["lon"], k=CANDIDATE_POOL_APPLE_PER_NODE)
        except Exception as e:
            print(f"Apple candidate lookup failed for node {node['id']}: {e}")
            candidates = []
        for pano in candidates:
            dist = haversine_m(node["lat"], node["lon"], pano.lat, pano.lon)
            if dist > APPLE_CANDIDATE_MAX_DIST_M:
                continue
            key = ("apple", pano.build_id)
            existing = node_options.get(key)
            if existing is None or dist < haversine_m(node["lat"], node["lon"], existing.lat, existing.lon):
                node_options[key] = pano

        options.append(node_options)
    return options


def _rank_passes_at(options: list[dict], start: int, exclude: set | None = None) -> list[tuple[str, str]]:
    """Rank the (source, pass_key) pairs available at node index `start` by
    how many consecutive nodes starting there also have that same pass --
    pure index math over _pass_options' already-gathered metadata, no
    GPU/network cost. This only picks the order to *try* passes in; it is
    not a substitute for the real pairwise DA3 health check (grading is
    always done on the actual candidate pair, not this heuristic)."""
    exclude = exclude or set()
    candidates = [key for key in options[start] if key not in exclude]

    def run_length(key):
        n, i = 0, start
        while i < len(options) and key in options[i]:
            n += 1
            i += 1
        return n

    return sorted(candidates, key=run_length, reverse=True)


def _resolve_pass_candidates(nodes: list[dict], options: list[dict[tuple[str, str], object]]) -> list[dict[tuple[str, str], Candidate]]:
    """Downloads whichever specific candidates _pass_options found (bounded
    by CANDIDATE_POOL_APPLE_PER_NODE's build_id cap and
    GOOGLE_DATE_OPTIONS_PER_NODE) and resolves each to a Candidate. Runs
    eagerly for every option here, not lazily per-attempt inside the GPU
    call -- keeps network I/O outside the ZeroGPU boundary, matching every
    other reconstruction path in this file.

    Google historical captures don't get their own lat/lon lookup (that'd
    be a second network call per candidate just for a value nothing in the
    greedy walk actually reads -- see Pipeline.run_greedy_pass_reconstruction,
    which decides everything from DA3's own output poses, not GPS); the
    node's own lat/lon is reused instead, same as _gather_candidate_pool
    does for a node's primary Google image.
    """
    resolved = []
    for node, node_options in zip(nodes, options):
        node_resolved = {}
        for key, value in node_options.items():
            source, pass_key = key
            try:
                if source == "apple":
                    pano = value
                    path = download_lookaround(pano)
                    node_resolved[key] = Candidate(f"apple:{pano.id}", path, pano.lat, pano.lon)
                else:
                    pano_id = value
                    path = asyncio.run(download_pano_by_id(pano_id))
                    if not path:
                        raise ValueError(f"Panorama {pano_id} not found")
                    node_resolved[key] = Candidate(f"google:{pano_id}", path, node["lat"], node["lon"])
            except Exception as e:
                print(f"Pass candidate download failed for {key} at node {node['id']}: {e}")
        resolved.append(node_resolved)
    return resolved


def reconstruct_chain_greedy(nodes: list[dict], output_dir: str, step_degrees: int = GREEDY_STEP_DEGREES) -> list[tuple[str, str]]:
    """Greedy same-capture-pass sliding-window reconstruction: instead of
    picking candidates by solo score (reconstruct_chain_best4) or windowing
    raw chain nodes with forced-overlap carryover (reconstruct_chain_windowed),
    walks the street node-by-node preferring whichever capture pass (Apple
    build_id or Google historical-date group -- see _pass_options) has the
    best contiguous coverage going forward, and health-checks each 2-node
    window with a real pairwise DA3 call (not solo score -- this session
    found solo score doesn't reliably predict either direction) before
    committing. See Pipeline.run_greedy_pass_reconstruction for the actual
    walk/branch logic, which runs entirely inside one GPU call for the same
    proxy-token-lifetime reason reconstruct_chain_windowed does.

    Unlike every other reconstruct_chain_* function here, this can return
    MULTIPLE disconnected point clouds ("segments") instead of one merged
    result -- a street with no single pass covering it end-to-end is
    expected to break into segments the user pieces together manually, not
    a failure to engineer around.

    Returns [(label, ply_path), ...], one per segment, in street order.
    """
    if len(nodes) < 2:
        raise ValueError("Need at least 2 nodes in the chain for DA3 to have multi-view context.")

    options = _pass_options(nodes)
    resolved = _resolve_pass_candidates(nodes, options)
    for node, node_resolved in zip(nodes, resolved):
        if not node_resolved:
            raise ValueError(f"Node {node['id']} has no usable capture-pass candidates at all.")

    try_order = [_rank_passes_at(options, i) for i in range(len(options))]

    # Alias every candidate to a source+pass+lat/lon filename, same reasoning
    # as _labeled_alias: panoramic-to-3dgs's DA3 log identifies a pano by
    # os.path.basename(path), so this is what actually shows up there. Can't
    # reuse _labeled_alias directly -- its source+lat/lon naming would
    # collide here, since every Google historical date at one node shares
    # that node's own lat/lon (see _resolve_pass_candidates); pass_key is
    # folded into the name too so different dates/build_ids stay distinct.
    with tempfile.TemporaryDirectory() as alias_dir:
        node_candidates = []
        for node_resolved in resolved:
            entries = []
            for (source, pass_key), c in node_resolved.items():
                ext = os.path.splitext(c.path)[1]
                alias_name = f"{source}_{pass_key}_lat{c.lat:.6f}_lon{c.lon:.6f}{ext}".replace("/", "_")
                alias_path = os.path.join(alias_dir, alias_name)
                if not os.path.exists(alias_path):
                    os.symlink(os.path.abspath(c.path), alias_path)
                entries.append((source, pass_key, c.label, alias_path, c.lat, c.lon))
            node_candidates.append(entries)

        segments = run_greedy_pass_reconstruction_gpu(
            node_candidates,
            try_order,
            keep_rate_threshold=GREEDY_KEEP_RATE_THRESHOLD,
            max_attempts_per_position=GREEDY_MAX_ATTEMPTS_PER_POSITION,
            step_degrees=step_degrees,
        )
    if not segments:
        raise RuntimeError("No segment of this street could be reconstructed with any available capture pass.")

    os.makedirs(output_dir, exist_ok=True)
    results = []
    for seg_idx, (pts, cols, node_range, pass_used) in enumerate(segments):
        start, end = node_range
        source, pass_key = pass_used
        label = f"segment {seg_idx} (nodes {start}-{end}, {source}:{pass_key})"
        path = save_pointcloud(pts, cols, os.path.join(output_dir, f"segment_{seg_idx}.ply"))
        results.append((label, path))
    return results


def _labeled_alias(candidate: Candidate, alias_dir: str) -> str:
    """Symlink to candidate.path named by source + lat/lon instead of its
    opaque pano ID (e.g. apple_lat1.234567_lon103.456789.jpg). The real
    cached file (services/lookaround_fetch.py, services/streetview_fetch.py)
    stays ID-named for reuse elsewhere in the app -- this only affects the
    path string handed to panoramic-to-3dgs, whose per-pano DA3 log now uses
    os.path.basename(path) as the pano_id (see DA3Model.py), so this alias is
    what actually shows up in that log: which candidate got how many views
    kept, tagged with the coordinates needed to check whether rejections
    correlate with distance between the reconstructed panos."""
    source = candidate.label.split(":", 1)[0]
    ext = os.path.splitext(candidate.path)[1]
    alias_path = os.path.join(alias_dir, f"{source}_lat{candidate.lat:.6f}_lon{candidate.lon:.6f}{ext}")
    if not os.path.exists(alias_path):
        os.symlink(os.path.abspath(candidate.path), alias_path)
    return alias_path


def _score_and_rank(pool: list[Candidate], step_degrees: int = 20) -> list[Candidate]:
    """Solo-score every candidate in the pool and return them sorted
    best-first. Only caller currently is reconstruct_chain_best4 --
    reconstruct_chain_windowed's scoring happens server-side inside
    Pipeline.run_windowed_reconstruction instead."""
    scores = score_candidates_gpu([c.path for c in pool], step_degrees=step_degrees)
    ranked = [c for c, _ in sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)]
    print(f"Candidate scores (label, keep-count/{360 // step_degrees + (360 % step_degrees > 0)}): {list(zip((c.label for c in pool), scores))}")
    return ranked


def reconstruct_chain_best4(nodes: list[dict], output_dir: str, step_degrees: int = BEST4_STEP_DEGREES) -> str:
    """Instead of trusting the chain's own Google nodes (plus closest-K Apple
    support) by default, builds a candidate pool (the chain's Google nodes +
    nearby Apple panos), scores each candidate SOLO through DA3 -- its own
    view-slices' self-consistency keep-rate, no other pano in the batch
    (see Pipeline.score_candidates) -- and reconstructs using only the
    BEST4_FINAL_COUNT highest-scoring candidates.

    Note this only measures each candidate's own internal coherence, not
    whether it'll correlate well with the others once combined -- two
    individually clean panos that are just too far apart could still both
    score high solo and fail to line up in the final joint DA3 call. That's
    exactly the open question this whole experiment is testing.

    step_degrees is used for BOTH the solo-scoring pass and the final
    reconstruction call, kept in sync deliberately: a candidate's keep-rate
    depends on its own per-pano consensus (median center, mean rotation
    across its own slices), which gets noisier with fewer slices -- scoring
    at a different step than reconstruction risks a mismatch (a candidate
    whose consensus looked robust at one slice count might not be at
    another). Defaults to BEST4_STEP_DEGREES (30); pass 20 explicitly for
    DA3's own original default.
    """
    pool = _gather_candidate_pool(nodes)
    if len(pool) < 2:
        raise ValueError("Need at least 2 candidate panos (chain nodes + Apple support) to score.")

    ranked = _score_and_rank(pool, step_degrees=step_degrees)
    winners = ranked[:BEST4_FINAL_COUNT]
    if len(winners) < 2:
        raise ValueError("Not enough candidates survived scoring for multi-view reconstruction.")

    print(f"Reconstructing with top {len(winners)} (step={step_degrees}): {[c.label for c in winners]}")
    with tempfile.TemporaryDirectory() as alias_dir:
        winner_paths = [_labeled_alias(c, alias_dir) for c in winners]
        ply_path = run_pointcloud_gpu(
            target_depth_path=winner_paths[0],
            output_dir=output_dir,
            support_paths=winner_paths[1:],
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
