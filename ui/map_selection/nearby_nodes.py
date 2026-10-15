"""Find the Street View panos and links near a place, for the map to show."""
import logging

from ui.map_selection.tools.google_tile_panos import google_tile_panos
from streetview_to_3d.types.Pano import Pano
from streetview_to_3d.panos.tools.google_node import google_node
from streetview_to_3d.tools.geo import haversine_m as _haversine_m

logger = logging.getLogger(__name__)

DEFAULT_RADIUS_M = 350
MAX_NODES = 200


def nearby_nodes(lat, lon, radius_m=DEFAULT_RADIUS_M, max_nodes=MAX_NODES):
    """(nodes, edges) within radius_m of (lat, lon): google_node dicts, nearest first (no
    date: the tile listing lacks it), and (key_a, key_b) links from the coverage graph."""
    try:
        panos = google_tile_panos(lat, lon)
    except Exception as e:
        logger.warning(f"Google coverage lookup failed: {e}")
        return [], []

    nodes = []
    for p in panos.values():
        if _haversine_m(lat, lon, p.lat, p.lon) > radius_m:
            continue
        nodes.append(google_node(p.id, p.lat, p.lon, p.heading))
    nodes.sort(key=lambda n: _haversine_m(lat, lon, n["lat"], n["lon"]))
    nodes = nodes[:max_nodes]

    kept_keys = {n["key"] for n in nodes}
    edges = set()
    for key in kept_keys:
        pano_id = key.split(":", 1)[1]  # Google ids are strings, matching `panos`' keys directly
        p = panos.get(pano_id)
        if not p:
            continue
        for link in (p.links or []):
            other_key = Pano.key_of("google", link.pano.id)
            if other_key in kept_keys and other_key != key:
                edges.add(tuple(sorted((key, other_key))))

    return nodes, sorted(edges)
