"""Search build_graph's candidate graph start -> end, reconstruct the path.

Client half of the pathfinding flow:
- build the candidate graph (build_graph, no GPU)
- download every candidate image (network, cached; the GPU call can't
  fetch, so this is eager for now -- slow first run, fast after)
- hand nodes + edges to run_pathfind_reconstruction (one GPU call: the
  real best-first search + DA3 tests + stitching live there)

v1: single date, single segment. See run_pathfind_reconstruction.
"""
import os

from services.lookaround_fetch import download_lookaround
from services.pipeline_runner import run_pathfind_reconstruction_gpu, save_pointcloud
from services.streetview_fetch import run_async, download_pano_by_id
from street_builder.build_graph.build_graph import build_corridor_graph
from street_builder.reconstruction.best4 import BEST4_STEP_DEGREES


def _download(node):
    """Download a node's equirectangular image, return path (None on failure)."""
    try:
        if node["source"] == "apple":
            return download_lookaround(node["_pano"])
        return run_async(download_pano_by_id(node["id"]))
    except Exception as e:
        print(f"Download failed for {node['key']}: {e}")
        return None


def reconstruct_pathfind(start_lat, start_lon, end_lat, end_lon, output_dir,
                         step_degrees: int = BEST4_STEP_DEGREES) -> list[tuple[str, str]]:
    """Auto-path a street start -> end. Returns [(label, ply_path), ...]."""
    nodes, edges = build_corridor_graph(start_lat, start_lon, end_lat, end_lon)
    nodes = [n for n in nodes if edges.get(n["key"])]  # drop isolated (never testable)
    if len(nodes) < 2:
        raise ValueError("Not enough connected candidates along this street.")

    print(f"Downloading {len(nodes)} corridor candidates (cached after first run)...")
    node_entries = []
    for i, n in enumerate(nodes):
        path = _download(n)
        if path:
            node_entries.append((n["key"], path, n["lat"], n["lon"], n["date"]))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(nodes)} downloaded")

    have = {e[0] for e in node_entries}
    edges = {k: [(o, d) for o, d in v if o in have] for k, v in edges.items() if k in have}

    segments = run_pathfind_reconstruction_gpu(
        node_entries, edges, start_lat, start_lon, end_lat, end_lon, step_degrees=step_degrees,
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
