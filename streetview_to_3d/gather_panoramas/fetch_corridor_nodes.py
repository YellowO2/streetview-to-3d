"""Gather every Google panorama along a street corridor (no GPU)."""
from streetview_to_3d.gather_panoramas.tools.corridor_points import corridor_points
from streetview_to_3d.gather_panoramas.tools.fetch_panos_by_id import fetch_panos_by_id
from streetview_to_3d.gather_panoramas.tools.google_node import google_node
from streetview_to_3d.tools.run_async import run_async


def fetch_corridor_nodes(edges):
    """Every capture date of the selected panos folded into each dot, one candidate per date.

    Returns (buckets, points, adjacency, elevations): buckets is {dot: [node dicts with date,
    pitch, roll]}, elevations is metres above sea level per dot (saves placement a re-fetch).
    Raises ValueError without an edge.
    """
    if not edges:
        raise ValueError("Need at least one confirmed edge tracing the route.")
    points, adjacency, members = corridor_points(edges)
    ids = [pano_id for m in members for pano_id in m]
    meta_by_id = dict(zip(ids, run_async(fetch_panos_by_id(ids))))

    buckets = {i: [] for i in range(len(points))}
    elevations = [None] * len(points)
    for i, pano_ids in enumerate(members):
        seen = set()
        for pano_id in pano_ids:
            meta = meta_by_id.get(pano_id)
            if not meta:
                continue
            if elevations[i] is None:
                elevations[i] = meta.get("elevation")
            for entry in meta["dates"]:
                if entry["id"] in seen:
                    continue
                seen.add(entry["id"])
                # each date's own pose: older captures are separate drives
                buckets[i].append(google_node(
                    entry["id"], entry.get("lat", meta["lat"]), entry.get("lon", meta["lon"]),
                    entry.get("heading", meta.get("heading")), date=entry["label"],
                    pitch=entry.get("pitch", meta.get("pitch")), roll=entry.get("roll", meta.get("roll"))))

    return buckets, points, adjacency, elevations
