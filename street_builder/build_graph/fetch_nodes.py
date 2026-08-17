"""Gather every Google + Apple panorama along a street corridor (no GPU)."""
import asyncio

from services.geo import haversine_m
from services.streetview_fetch import fetch_pano_by_id, format_date
from street_builder.map_selection.candidates import (
    APPLE_CANDIDATE_MAX_DIST_M,
    MAX_NODES,
    apple_tile_panos,
    nearby_nodes,
    node_key,
)


def fetch_corridor_nodes(start_lat, start_lon, end_lat, end_lon, apple_radius_m=APPLE_CANDIDATE_MAX_DIST_M):
    """Every Google + Apple pano near the real street between start and end.

    - Uses real Google node positions as the backbone (follows the actual
      street shape, curves included) instead of a straight-line guess.
    - For each Google stop, also fetches its historical dates (one graph
      node per date) and nearby Apple panos within apple_radius_m.

    Returns a flat list of {key, source, id, lat, lon, date} dicts.
    """
    mid_lat, mid_lon = (start_lat + end_lat) / 2, (start_lon + end_lon) / 2
    span_m = haversine_m(start_lat, start_lon, end_lat, end_lon)
    google_stops, _ = nearby_nodes(mid_lat, mid_lon, radius_m=span_m / 2 + apple_radius_m, max_nodes=MAX_NODES)

    nodes = []
    for gs in google_stops:
        try:
            meta = asyncio.run(fetch_pano_by_id(gs["id"]))
        except Exception as e:
            print(f"Google date lookup failed for {gs['id']}: {e}")
            continue
        if not meta:
            continue
        for entry in meta["dates"]:
            nodes.append({
                "key": node_key("google", entry["id"]), "source": "google", "id": entry["id"],
                "lat": gs["lat"], "lon": gs["lon"], "date": entry["label"],
            })

    seen_apple_ids = set()
    for gn in google_stops:
        try:
            candidates = apple_tile_panos(gn["lat"], gn["lon"])
        except Exception as e:
            print(f"Apple corridor lookup failed near {gn['id']}: {e}")
            continue
        for p in candidates.values():
            if p.id in seen_apple_ids:
                continue
            if haversine_m(gn["lat"], gn["lon"], p.lat, p.lon) > apple_radius_m:
                continue
            seen_apple_ids.add(p.id)
            nodes.append({
                "key": node_key("apple", p.id), "source": "apple", "id": p.id,
                "lat": p.lat, "lon": p.lon, "date": format_date(p.date),
                "_pano": p,  # kept for download_lookaround (needs the object, not just the id)
            })

    return nodes
