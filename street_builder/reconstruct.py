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
    run_editor_gpu,
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

# Yaw step for extract_views_for_da3's slicing. DA3's own default (20 ->
# 18 slices/pano, ~78% overlap) turned out to matter for more than
# redundancy: at step=45 (8 slices/pano), 2 of 4 best-4 winners went from
# partial acceptance to fully rejected -- fewer slices makes a pano's own
# per-slice consensus noisier, not just less redundant. 30 (12 slices, still
# evenly divides 360) is the middle ground being tested now, used for BOTH
# scoring and final reconstruction so the two stay consistent with each
# other (reconstruct_chain_best4's step_degrees default).
BEST4_STEP_DEGREES = 30

# Used only by the full-pool experiments, where a coarser step is needed to
# keep the unfiltered pool's image count down (10 unscored candidates would
# otherwise exceed the crash threshold we've observed).
FULL_POOL_STEP_DEGREES = 45

# _gather_candidate_pool's Apple lookup is nearest-K only, with no cap on how
# far "nearest" might actually be if local Apple coverage is sparse -- this
# bounds it, so a pool built for one window can't reach into territory well
# outside that window's own span. ~2.5x the ~10m consecutive-node spacing
# we've measured on our one real test street; sparser streets may need this
# retuned, but there's no data yet to justify a fancier per-street estimate.
APPLE_CANDIDATE_MAX_DIST_M = 25.0

# One-off diagnostic for reconstruct_chain_best4_car_removal_test: on our one
# real test chain, two of the four best-4 winners (apple:8733854682473071389,
# apple:8733854682473072390) scored well solo (11/12, 10/12) but collapsed to
# near-zero (0/12, 1/12) once reconstructed jointly with the other two -- and
# they're only ~1.3m apart from each other (closer than any other pair in the
# set), which argues against "too far apart" and for a genuine content
# mismatch (moving cars/people) between those two specific captures. Both
# panos visibly contain a car; testing whether Flux's "Remove people &
# vehicles" preset run on just these two (not all four, to save GPU time)
# recovers their keep-rate. Hardcoded to this specific test chain -- delete
# this constant + the function below once the test's been run.
CAR_REMOVAL_TEST_LABELS = {"apple:8733854682473071389", "apple:8733854682473072390"}
CAR_REMOVAL_TEST_PROMPT = "Remove all people and vehicles from the scene."

# Fixed order = the exact best4 winners already confirmed on this test chain,
# pano_0..pano_3 in the same order every diagnostic run below uses -- so
# per-pano keep-counts stay directly comparable across runs. Re-scoring from
# scratch each time (the pool has several candidates tied at 10/12) risked
# silently picking a *different* top-4 set run to run, which would've
# invalidated the comparison; skipping scoring here avoids that AND saves a
# full GPU scoring pass per debug click.
KNOWN_BEST4_LABELS = [
    "apple:2722660790751527800",
    "apple:8733854682473071389",
    "apple:2722660790751529047",
    "apple:8733854682473072390",
]

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


def _known_best4_winners(nodes: list[dict]) -> list[Candidate]:
    """Re-gathers the candidate pool (needed to re-download/re-locate the
    images and their lat/lon) but skips scoring entirely -- filters straight
    to KNOWN_BEST4_LABELS, in that fixed order. See KNOWN_BEST4_LABELS for why."""
    pool = _gather_candidate_pool(nodes)
    by_label = {c.label: c for c in pool}
    missing = [label for label in KNOWN_BEST4_LABELS if label not in by_label]
    if missing:
        raise ValueError(f"Known best-4 winners not found in current candidate pool: {missing}")
    return [by_label[label] for label in KNOWN_BEST4_LABELS]


def _known_best4_pano_objects(nodes: list[dict]) -> dict:
    """Looks up the raw LookaroundPanorama objects (not just the label/path/
    lat/lon Candidate tuple _gather_candidate_pool builds) for the 4 known
    best4 winners -- needed for their heading/pitch/roll metadata, which
    Candidate doesn't carry. Returns {label: pano}, all 4 KNOWN_BEST4_LABELS
    guaranteed present (raises if any are missing)."""
    found = {}
    for node in nodes:
        try:
            candidates = apple_candidates(node["lat"], node["lon"], k=CANDIDATE_POOL_APPLE_PER_NODE)
        except Exception as e:
            print(f"Apple candidate lookup failed for node {node['id']}: {e}")
            continue
        for pano in candidates:
            label = f"apple:{pano.id}"
            if label in KNOWN_BEST4_LABELS:
                found[label] = pano
    missing = [label for label in KNOWN_BEST4_LABELS if label not in found]
    if missing:
        raise ValueError(f"Known best-4 winners not found via Apple lookup: {missing}")
    return found


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


def reconstruct_chain_best4_car_removal_test(nodes: list[dict], output_dir: str, step_degrees: int = BEST4_STEP_DEGREES) -> str:
    """One-off diagnostic, not a permanent feature (see CAR_REMOVAL_TEST_LABELS):
    reconstructs the exact same 4 candidates already confirmed as the best4
    winners on this chain (see _known_best4_winners -- no re-scoring), but
    before the final reconstruction call, runs Flux's "Remove people &
    vehicles" edit on whichever of the 4 match CAR_REMOVAL_TEST_LABELS.
    Everything else is untouched. Compare this run's per-pano keep-counts
    (printed by DA3, see DA3Model.py) against the un-edited run's to check
    whether moving cars/people -- not distance -- explain why those two
    panos collapsed."""
    winners = _known_best4_winners(nodes)
    to_clean = [c for c in winners if c.label in CAR_REMOVAL_TEST_LABELS]
    print(f"Reconstructing with top {len(winners)} (step={step_degrees}), removing cars/people from: {[c.label for c in to_clean]}")

    with tempfile.TemporaryDirectory() as edit_dir, tempfile.TemporaryDirectory() as alias_dir:
        cleaned_winners = []
        for c in winners:
            if c.label in CAR_REMOVAL_TEST_LABELS:
                edited_path = os.path.join(edit_dir, os.path.basename(c.path))
                run_editor_gpu(c.path, CAR_REMOVAL_TEST_PROMPT, "remove_objects", edited_path)
                cleaned_winners.append(Candidate(c.label, edited_path, c.lat, c.lon))
            else:
                cleaned_winners.append(c)

        winner_paths = [_labeled_alias(c, alias_dir) for c in cleaned_winners]
        ply_path = run_pointcloud_gpu(
            target_depth_path=winner_paths[0],
            output_dir=output_dir,
            support_paths=winner_paths[1:],
            step_degrees=step_degrees,
        )
    if not ply_path:
        raise RuntimeError("Pipeline finished but no point cloud was produced.")
    return ply_path


def reconstruct_chain_best4_drop_one_test(nodes: list[dict], output_dir: str, step_degrees: int = BEST4_STEP_DEGREES) -> list[tuple[str, str | None]]:
    """One-off diagnostic, not a permanent feature (see CAR_REMOVAL_TEST_LABELS):
    reconstructs the exact same 4 candidates already confirmed as the best4
    winners on this chain (see _known_best4_winners -- no re-scoring), but
    instead of editing anything, runs two 3-candidate reconstructions --
    winners minus one of the two collapsing panos, then winners minus the
    other -- to check whether each collapsing pano gets accepted fine once
    the *other* collapsing one is out of the batch (pointing at a pairwise
    conflict between those two specifically) or still gets rejected against
    the other two winners alone (pointing at something else). Cheaper than
    the car-removal test since it skips Flux entirely. Per-pano keep-counts
    are only in the server log (DA3Model.py), same as every other debug path
    here.

    Returns [(label, ply_path_or_None), ...], one per dropped candidate.
    """
    winners = _known_best4_winners(nodes)
    if len(winners) < 3:
        raise ValueError("Need at least 3 best-4 winners to drop one and still have multi-view context.")

    results = []
    with tempfile.TemporaryDirectory() as alias_dir:
        for dropped_label in sorted(CAR_REMOVAL_TEST_LABELS):
            remaining = [c for c in winners if c.label != dropped_label]
            if len(remaining) == len(winners):
                print(f"Drop-one test: {dropped_label} wasn't among this run's winners, skipping.")
                results.append((f"without {dropped_label}", None))
                continue
            print(f"Drop-one test: reconstructing without {dropped_label} -- {[c.label for c in remaining]}")
            remaining_paths = [_labeled_alias(c, alias_dir) for c in remaining]
            run_output_dir = os.path.join(output_dir, f"drop_{dropped_label.replace(':', '_')}")
            ply_path = run_pointcloud_gpu(
                target_depth_path=remaining_paths[0],
                output_dir=run_output_dir,
                support_paths=remaining_paths[1:],
                step_degrees=step_degrees,
            )
            results.append((f"without {dropped_label}", ply_path))

    return results


def reconstruct_chain_best4_pairwise_test(nodes: list[dict], output_dir: str, step_degrees: int = BEST4_STEP_DEGREES) -> list[tuple[str, str | None]]:
    """One-off diagnostic, not a permanent feature (see CAR_REMOVAL_TEST_LABELS):
    reconstructs the exact same 4 known best4 winners (see
    _known_best4_winners -- no re-scoring), but as all 6 possible 2-candidate
    pairs instead of the full 4 or a 3-way drop-one. The drop-one test showed
    each collapsing pano (apple:...071389, apple:...072390) fails even when
    the *other* collapsing one is entirely absent, ruling out a two-way
    conflict between just those two -- this narrows further: does each
    collapsing pano fail against literally every partner (pointing at that
    pano being individually flawed), or only against pano_0/pano_2
    specifically (pointing at two separate, mutually-inconsistent capture
    clusters)? The pano_1-vs-pano_3 pair is the most diagnostic single case,
    since they're the closest two candidates of all six (~1.3m apart) --
    good correlation there despite both failing against pano_0/pano_2 would
    be strong evidence for the latter.

    Returns [(label, ply_path_or_None), ...], one per pair, label formatted
    "A vs B".
    """
    from itertools import combinations

    winners = _known_best4_winners(nodes)
    if len(winners) < 2:
        raise ValueError("Need at least 2 best-4 winners to test pairs.")

    results = []
    with tempfile.TemporaryDirectory() as alias_dir:
        for a, b in combinations(winners, 2):
            pair_label = f"{a.label} vs {b.label}"
            print(f"Pairwise test: reconstructing {pair_label}")
            pair_paths = [_labeled_alias(a, alias_dir), _labeled_alias(b, alias_dir)]
            run_output_dir = os.path.join(
                output_dir, f"pair_{a.label.replace(':', '_')}_{b.label.replace(':', '_')}"
            )
            ply_path = run_pointcloud_gpu(
                target_depth_path=pair_paths[0],
                output_dir=run_output_dir,
                support_paths=pair_paths[1:],
                step_degrees=step_degrees,
            )
            results.append((pair_label, ply_path))

    return results


def reconstruct_chain_best4_slope_correction_test(
    nodes: list[dict],
    output_dir: str,
    labels: list[str] | None = None,
    multiplier: float = 1.0,
    step_degrees: int = BEST4_STEP_DEGREES,
) -> str:
    """One-off diagnostic, not a permanent feature (see CAR_REMOVAL_TEST_LABELS):
    de-tilts the given candidates (default: all 4 known best4 winners) by
    their own heading/pitch/roll (services/slope_correction.correct_slope,
    already used for the single-pano Google flow, applied here to Apple
    metadata for the first time) before reconstruction. Tests whether the
    small (~0.1-0.2deg) but systematic pitch difference between the two
    capture passes A/C (build_id 2147485257) and B/D (build_id 2147486516,
    ~7 weeks apart) is enough to explain the cross-pass reconstruction
    failures. `multiplier` scales the correction, in case the reported
    pitch/roll undersells the actual misalignment (see app.py's own
    "Try >1x" note on this same knob for the single-pano flow).

    labels: subset of KNOWN_BEST4_LABELS to reconstruct with, in that fixed
    order (e.g. just the A-vs-B pair, the worst-correlating one, to check
    for a signal cheaply before spending a full 4-way run).
    """
    from services.slope_correction import correct_slope

    labels = labels or KNOWN_BEST4_LABELS
    pano_objs = _known_best4_pano_objects(nodes)
    print(f"Slope-correction test (multiplier={multiplier}): de-tilting {labels}")

    with tempfile.TemporaryDirectory() as alias_dir:
        corrected = []
        for label in labels:
            pano = pano_objs[label]
            raw_path = download_lookaround(pano)
            leveled_path = correct_slope(raw_path, pano.heading, pano.pitch, pano.roll, multiplier=multiplier)
            # "appleLeveled" (not "apple") as the alias source, so _labeled_alias's
            # filename actually differs from the unleveled run's -- otherwise both
            # runs would alias to the identical "apple_lat..lon..jpg" name and the
            # DA3 log couldn't tell which version was reconstructed.
            leveled_label = f"appleLeveled:{label.split(':', 1)[1]}"
            corrected.append(Candidate(leveled_label, leveled_path, pano.lat, pano.lon))

        if len(corrected) < 2:
            raise ValueError("Need at least 2 labels to reconstruct.")

        winner_paths = [_labeled_alias(c, alias_dir) for c in corrected]
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
