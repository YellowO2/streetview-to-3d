"""Search build_graph's candidate graph start -> end, reconstruct the path.

Client half of the pathfinding flow:
- build the candidate graph (build_graph, no GPU)
- download candidates (network, cached; the GPU call can't fetch, so this
  has to happen before it -- see the two-phase note below)
- hand nodes + edges to run_pathfind_reconstruction (one GPU call: the
  real best-first search + DA3 tests + stitching live there)

Two-phase download, not "download everything": a corridor can hold
hundreds of candidates (Apple frames run ~1.2m apart), but the search only
ever tests a few dozen. Phase 1 downloads just the local neighborhood near
the start; only if that doesn't reach the end do we download the rest of
the corridor and retry. Phase 2 re-runs the search from scratch (no
resumable state across GPU calls), so it re-tests phase 1's edges too --
acceptable since it's a rare fallback, not the common path.

v1: single date, single segment. See run_pathfind_reconstruction.
"""
import os

from services.geo import haversine_m
from services.lookaround_fetch import DA3_ONLY_APPLE_ZOOM, download_lookaround
from services.pipeline_runner import run_pathfind_reconstruction_gpu, save_pointcloud
from services.streetview_fetch import DA3_ONLY_ZOOM, run_async, download_pano_by_id
from street_builder.build_graph.build_graph import build_corridor_graph
from street_builder.reconstruction.best4 import BEST4_STEP_DEGREES


def _download(node):
    """Download a node's equirectangular image at DA3-only res, return path (None on failure)."""
    try:
        if node["source"] == "apple":
            return download_lookaround(node["_pano"], zoom=DA3_ONLY_APPLE_ZOOM)
        return run_async(download_pano_by_id(node["id"], zoom=DA3_ONLY_ZOOM))
    except Exception as e:
        print(f"Download failed for {node['key']}: {e}")
        return None


def _local_batch(nodes, google_stops):
    """Nearest candidate of each date to each real Google stop -- spreads
    phase-1 downloads across the whole corridor (google_stops already runs
    its full length), instead of clustering near just one point."""
    by_date = {}
    for n in nodes:
        by_date.setdefault(n["date"], []).append(n)

    keys = set()
    for stop in google_stops:
        for candidates in by_date.values():
            nearest = min(candidates, key=lambda n: haversine_m(stop["lat"], stop["lon"], n["lat"], n["lon"]))
            keys.add(nearest["key"])
    return keys


def _download_and_filter(keys, by_key, edges):
    """Download this batch, return (node_entries, edges restricted to what downloaded)."""
    entries = []
    for key in keys:
        n = by_key[key]
        path = _download(n)
        if path:
            entries.append((key, path, n["lat"], n["lon"], n["date"]))
    have = {e[0] for e in entries}
    filtered = {k: [(o, d) for o, d in v if o in have] for k, v in edges.items() if k in have}
    return entries, filtered


def reconstruct_pathfind(start_lat, start_lon, end_lat, end_lon, output_dir,
                         step_degrees: int = BEST4_STEP_DEGREES) -> list[tuple[str, str]]:
    """Auto-path a street start -> end. Returns [(label, ply_path), ...]."""
    nodes, edges, google_stops = build_corridor_graph(start_lat, start_lon, end_lat, end_lon)
    nodes = [n for n in nodes if edges.get(n["key"])]  # drop isolated (never testable)
    if len(nodes) < 2:
        raise ValueError("Not enough connected candidates along this street.")
    by_key = {n["key"]: n for n in nodes}

    batch_keys = _local_batch(nodes, google_stops)
    print(f"Phase 1: downloading {len(batch_keys)}/{len(nodes)} local candidates...")
    node_entries, batch_edges = _download_and_filter(batch_keys, by_key, edges)

    segments = run_pathfind_reconstruction_gpu(
        node_entries, batch_edges, start_lat, start_lon, end_lat, end_lon, step_degrees=step_degrees,
    )

    if not segments or not segments[0][4]:  # segments[0] = (pts, cols, path_edges, date, reached)
        print("Phase 1 didn't reach the end -- downloading the rest of the corridor...")
        rest_keys = {n["key"] for n in nodes} - batch_keys
        rest_entries, _ = _download_and_filter(rest_keys, by_key, edges)
        all_entries = node_entries + rest_entries
        have = {e[0] for e in all_entries}
        full_edges = {k: [(o, d) for o, d in v if o in have] for k, v in edges.items() if k in have}
        segments = run_pathfind_reconstruction_gpu(
            all_entries, full_edges, start_lat, start_lon, end_lat, end_lon, step_degrees=step_degrees,
        )

    if not segments:
        raise RuntimeError("No connected path found between start and end.")

    os.makedirs(output_dir, exist_ok=True)
    results = []
    for i, (pts, cols, path_edges, date, reached) in enumerate(segments):
        status = "reached end" if reached else "partial (didn't reach end)"
        label = f"path (date {date}, {len(path_edges)} hops, {status})"
        ply = save_pointcloud(pts, cols, os.path.join(output_dir, f"pathfind_{i}.ply"))
        results.append((label, ply))
    return results
