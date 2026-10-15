"""Add a clicked pano's own Street View links to a map state."""
import logging

from streetview_to_3d.gather_panoramas.tools.fetch_pano_by_id import fetch_pano_by_id
from streetview_to_3d.gather_panoramas.tools.google_node import google_node
from streetview_to_3d.tools.run_async import run_async

logger = logging.getLogger(__name__)


def with_real_links(state, key):
    """Merge this node's own per-pano links into state: the tile listing nearby_nodes uses
    can omit a linked pano, and only covers the area first loaded."""
    if not key.startswith("google:"):
        return state  # only Google panos have real link data
    pano_id = key.split(":", 1)[1]
    try:
        meta = run_async(fetch_pano_by_id(pano_id))
    except Exception as e:
        logger.warning(f"Link fetch failed for {pano_id}: {e}")
        return state
    if not meta:
        return state

    nodes = list(state["nodes"])
    edges = list(state["edges"])
    by_key = {n["key"]: n for n in nodes}
    edge_set = {frozenset(e) for e in edges}

    for n in meta["neighbors"]:
        new_node = google_node(n["id"], n["lat"], n["lon"])
        other_key = new_node["key"]
        if other_key not in by_key:
            nodes.append(new_node)
            by_key[other_key] = new_node
        fe = frozenset((key, other_key))
        if fe not in edge_set:
            edges.append((key, other_key))
            edge_set.add(fe)

    return {**state, "nodes": nodes, "edges": edges}
