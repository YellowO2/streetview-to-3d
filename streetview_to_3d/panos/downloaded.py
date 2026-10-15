"""Download every candidate of the corridor's date graphs (no GPU)."""
import asyncio
import logging

from streetview_to_3d.common.run_async import run_async
from streetview_to_3d.common.streetview_fetch.download_pano_by_id import DA3_ONLY_ZOOM, download_pano_by_id
from streetview_to_3d.panos.Corridor import Corridor

logger = logging.getLogger(__name__)

# Panos downloaded at once: fast enough that the ZeroGPU token (wall-clock) doesn't expire
# before the GPU call, bounded so Google's rate limiter isn't tripped.
DOWNLOAD_CONCURRENCY = 10


async def _download_one(node, sem):
    """A node's pano at DA3's resolution: its path, or None on failure."""
    async with sem:
        try:
            return await download_pano_by_id(node["id"], zoom=DA3_ONLY_ZOOM)
        except Exception as e:
            logger.warning(f"Download failed for {node['key']}: {e}")
            return None


async def _download_all(nodes):
    sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)
    return await asyncio.gather(*[_download_one(n, sem) for n in nodes])


def _download_date_graphs(date_graphs):
    """Download every candidate of every date graph in one batch.

    Returns (ready_graphs, catalog): ready_graphs has each dot's candidates as
    (key, path, lat, lon), dropping what failed to download (a dot left empty goes);
    catalog is {pano key: {dot, source, id, lat, lon, date, heading, pitch, roll}}, which
    matches a reconstructed pano back to its place once the walk picks a date.
    """
    all_nodes = [n for g in date_graphs for bucket in g["dot_candidates"].values() for n in bucket]
    keys = [n["key"] for n in all_nodes]
    catalog = {n["key"]: {"dot": dot, "source": n["source"], "id": n["id"],
                          "lat": n["lat"], "lon": n["lon"], "date": n.get("date"),
                          "heading": n.get("heading"), "pitch": n.get("pitch"),
                          "roll": n.get("roll")}
               for g in date_graphs for dot, bucket in g["dot_candidates"].items()
               for n in bucket}
    paths = run_async(_download_all(all_nodes))
    path_by_key = {key: path for key, path in zip(keys, paths) if path}

    ready_graphs = []
    for g in date_graphs:
        dot_candidates = {}
        for dot_idx, bucket in g["dot_candidates"].items():
            entries = [(n["key"], path_by_key[n["key"]], n["lat"], n["lon"])
                       for n in bucket if n["key"] in path_by_key]
            if entries:
                dot_candidates[dot_idx] = entries
        if dot_candidates:
            ready_graphs.append({"date": g["date"], "dot_candidates": dot_candidates})

    return ready_graphs, catalog


def downloaded(graphs, start, center):
    """The Corridor the GPU step takes (reconstruct.main.reconstruct), from date_graphs'
    graphs: each dot's candidates now (key, path, lat, lon), and what failed to download
    gone (see _download_date_graphs). start/center: (lat, lon). Raises ValueError if nothing downloaded."""
    n_candidates = sum(len(bucket) for g in graphs["date_graphs"] for bucket in g["dot_candidates"].values())
    logger.info(f"Downloading {n_candidates} candidate(s) across {len(graphs['date_graphs'])} date graph(s): "
                f"{[g['date'] for g in graphs['date_graphs']]}")
    ready_graphs, catalog = _download_date_graphs(graphs["date_graphs"])
    if not ready_graphs:
        raise ValueError("Nothing downloaded successfully -- can't reconstruct.")
    return Corridor(ready_graphs, graphs["points"], graphs["adjacency"], graphs["elevations"],
                    catalog, start, center)
