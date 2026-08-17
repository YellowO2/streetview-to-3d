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


def fetch_corridor_nodes(waypoints, apple_radius_m=APPLE_CANDIDATE_MAX_DIST_M):
    """Every Google + Apple pano near the real street traced by waypoints.

    waypoints: ordered [(lat, lon), ...] -- the user's full clicked chain,
    used as a polyline "spine", not just its first/last point. Fetches
    within a radius of each consecutive pair's own midpoint (a series of
    overlapping circles along the route) instead of one big circle over
    the whole start->end span -- so the corridor traces the actual shape
    the user clicked and excludes a geometrically-nearby branch they
    didn't select (e.g. a fork the route doesn't take).

    - Uses real Google node positions as the backbone (follows the actual
      street shape, curves included) instead of a straight-line guess.
    - For each Google stop, also fetches its historical dates (one graph
      node per date) and nearby Apple panos within apple_radius_m.

    Returns (nodes, google_stops): nodes is the flat {key, source, id, lat,
    lon, date} list; google_stops is the real backbone (one entry per
    physical Google stop, no date/expansion) -- callers that need positions
    spread across the whole corridor (see walk_graph._local_batch) should
    use google_stops, not nodes (which has many same-position date variants).
    """
    stops_by_key = {}
    for (lat1, lon1), (lat2, lon2) in zip(waypoints, waypoints[1:]):
        mid_lat, mid_lon = (lat1 + lat2) / 2, (lon1 + lon2) / 2
        span_m = haversine_m(lat1, lon1, lat2, lon2)
        seg_stops, _ = nearby_nodes(mid_lat, mid_lon, radius_m=span_m / 2 + apple_radius_m, max_nodes=MAX_NODES)
        for s in seg_stops:
            stops_by_key[s["key"]] = s
    google_stops = list(stops_by_key.values())

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

    return nodes, google_stops
