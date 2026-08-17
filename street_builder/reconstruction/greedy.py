"""Greedy same-date sliding-window reconstruction: walks the street
node-by-node preferring whichever capture date (Apple or Google) has the
best contiguous coverage, and grades each 2-node window with a real
pairwise DA3 call (not solo score -- unlike best4.py, this session found
solo score doesn't reliably predict either direction).

Can return MULTIPLE disconnected point clouds ("segments") instead of one
merged result -- a street with no single pass covering it end-to-end
breaks into segments the user pieces together manually.
"""
import asyncio
import os
import tempfile

from services.geo import haversine_m
from services.lookaround_fetch import apple_candidates, download_lookaround
from services.pipeline_runner import run_greedy_pass_reconstruction_gpu, save_pointcloud
from services.streetview_fetch import download_pano_by_id, format_date
from street_builder.map_selection.candidates import APPLE_CANDIDATE_MAX_DIST_M
from street_builder.reconstruction.best4 import BEST4_STEP_DEGREES
from street_builder.reconstruction.common import CANDIDATE_POOL_APPLE_PER_NODE, Candidate

# How many of a node's most-recent Google historical dates to try (every
# option here gets downloaded eagerly, so this bounds download cost).
GOOGLE_DATE_OPTIONS_PER_NODE = 3

# Fraction of a candidate's own view-slices that must survive DA3's filter
# for a window to count as "healthy". 0.5 matches this session's clean
# empirical split: healthy same-pass pairs kept 9-12/12, collapsed
# cross-pass pairs kept 0-1/12.
GREEDY_KEEP_RATE_THRESHOLD = 0.5

# How many candidate passes to try at a position before giving up on it.
GREEDY_MAX_ATTEMPTS_PER_POSITION = 3

GREEDY_STEP_DEGREES = BEST4_STEP_DEGREES


def _pass_options(nodes: list[dict]) -> list[dict[str, object]]:
    """Per node, every available capture pass, keyed by date string (same
    format_date both Apple and Google use, so the two are directly
    comparable). Values: an Apple pano object, or a bare Google pano ID.
    Metadata only -- no downloads."""
    options = []
    for node in nodes:
        node_options = {}

        for entry in node.get("dates", [])[:GOOGLE_DATE_OPTIONS_PER_NODE]:
            node_options[entry["label"]] = entry["id"]

        try:
            candidates = apple_candidates(node["lat"], node["lon"], k=CANDIDATE_POOL_APPLE_PER_NODE)
        except Exception as e:
            print(f"Apple candidate lookup failed for node {node['id']}: {e}")
            candidates = []
        for pano in candidates:
            dist = haversine_m(node["lat"], node["lon"], pano.lat, pano.lon)
            if dist > APPLE_CANDIDATE_MAX_DIST_M:
                continue
            date = format_date(pano.date)
            existing = node_options.get(date)
            if isinstance(existing, str):
                continue  # Google's own same-date image already claims this slot
            if existing is None or dist < haversine_m(node["lat"], node["lon"], existing.lat, existing.lon):
                node_options[date] = pano

        options.append(node_options)
    return options


def _rank_passes_at(options: list[dict], start: int, exclude: set | None = None) -> list[str]:
    """Rank dates at node `start` by how many consecutive nodes share them.
    Only picks try-order -- grading is always the real pairwise DA3 call."""
    exclude = exclude or set()
    candidates = [key for key in options[start] if key not in exclude]

    def run_length(key):
        n, i = 0, start
        while i < len(options) and key in options[i]:
            n += 1
            i += 1
        return n

    return sorted(candidates, key=run_length, reverse=True)


def _resolve_pass_candidates(nodes: list[dict], options: list[dict[str, object]]) -> list[dict[str, Candidate]]:
    """Downloads every option eagerly (keeps network I/O outside the GPU call)."""
    resolved = []
    for node, node_options in zip(nodes, options):
        node_resolved = {}
        for date, value in node_options.items():
            try:
                if isinstance(value, str):
                    pano_id = value
                    path = asyncio.run(download_pano_by_id(pano_id))
                    if not path:
                        raise ValueError(f"Panorama {pano_id} not found")
                    node_resolved[date] = Candidate(f"google:{pano_id}", path, node["lat"], node["lon"])
                else:
                    pano = value
                    path = download_lookaround(pano)
                    node_resolved[date] = Candidate(f"apple:{pano.id}", path, pano.lat, pano.lon)
            except Exception as e:
                print(f"Pass candidate download failed for {date} at node {node['id']}: {e}")
        resolved.append(node_resolved)
    return resolved


def reconstruct_chain_greedy(nodes: list[dict], output_dir: str, step_degrees: int = GREEDY_STEP_DEGREES) -> list[tuple[str, str]]:
    """Returns [(label, ply_path), ...], one per segment, in street order."""
    if len(nodes) < 2:
        raise ValueError("Need at least 2 nodes in the chain for DA3 to have multi-view context.")

    options = _pass_options(nodes)
    resolved = _resolve_pass_candidates(nodes, options)
    for node, node_resolved in zip(nodes, resolved):
        if not node_resolved:
            raise ValueError(f"Node {node['id']} has no usable capture-pass candidates at all.")

    try_order = [_rank_passes_at(options, i) for i in range(len(options))]

    # Alias to source+date+lat/lon: panoramic-to-3dgs's DA3 log uses
    # os.path.basename(path) as the pano id, so this is what shows up
    # there. date is folded in since every Google historical date at one
    # node shares that node's own lat/lon.
    with tempfile.TemporaryDirectory() as alias_dir:
        node_candidates = []
        for node_resolved in resolved:
            entries = []
            for date, c in node_resolved.items():
                source = c.label.split(":", 1)[0]
                ext = os.path.splitext(c.path)[1]
                alias_name = f"{source}_{date}_lat{c.lat:.6f}_lon{c.lon:.6f}{ext}".replace("/", "_")
                alias_path = os.path.join(alias_dir, alias_name)
                if not os.path.exists(alias_path):
                    os.symlink(os.path.abspath(c.path), alias_path)
                entries.append((date, c.label, alias_path, c.lat, c.lon))
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
    for seg_idx, (pts, cols, node_range, date_used) in enumerate(segments):
        start, end = node_range
        label = f"segment {seg_idx} (nodes {start}-{end}, date {date_used})"
        path = save_pointcloud(pts, cols, os.path.join(output_dir, f"segment_{seg_idx}.ply"))
        results.append((label, path))
    return results
