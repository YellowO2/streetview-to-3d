"""Gather every Google + Apple panorama along a street corridor (no GPU)."""
import asyncio

from services.geo import haversine_m
from services.streetview_fetch import fetch_pano_by_id, format_date
from street_builder.map_selection.candidates import MAX_NODES, apple_tile_panos, nearby_nodes, node_key

# Sample points every this many meters along the clicked route -- the one
# spine used for both gathering (this file) and date-coverage ranking
# (walk_graph.py), instead of relying on wherever real nodes happen to be.
POINT_SPACING_M = 5.0

# A candidate only counts as "at" a sample point within this range.
POINT_MAX_DIST_M = 8.0


def interpolate_points(waypoints, spacing_m: float = POINT_SPACING_M) -> list[tuple[float, float]]:
    """Evenly-spaced (lat, lon) points along the waypoint polyline, one
    segment (consecutive waypoint pair) at a time."""
    points = [waypoints[0]]
    for (lat1, lon1), (lat2, lon2) in zip(waypoints, waypoints[1:]):
        seg_len = haversine_m(lat1, lon1, lat2, lon2)
        n_steps = max(1, round(seg_len / spacing_m))
        for i in range(1, n_steps + 1):
            t = i / n_steps
            points.append((lat1 + (lat2 - lat1) * t, lon1 + (lon2 - lon1) * t))
    return points


def fetch_corridor_nodes(waypoints, max_dist_m: float = POINT_MAX_DIST_M):
    """Every Google + Apple pano within max_dist_m of any interpolated
    point along the waypoint polyline (see interpolate_points).

    - For each point: nearby_nodes (Google stops) + apple_tile_panos
      (Apple), both metadata only. A newly-seen Google stop gets one extra
      fetch_pano_by_id call for its real historical dates (one graph node
      per date); already-seen stops/panos aren't re-fetched.

    Returns (nodes, points): nodes is the flat {key, source, id, lat, lon,
    date} list; points is the interpolated point list (needed by
    walk_graph.py for date-coverage ranking).
    """
    points = interpolate_points(waypoints)

    nodes = []
    seen_google_ids = set()
    seen_apple_ids = set()

    for lat, lon in points:
        try:
            google_candidates, _ = nearby_nodes(lat, lon, radius_m=max_dist_m, max_nodes=MAX_NODES)
        except Exception as e:
            print(f"Google lookup failed near ({lat}, {lon}): {e}")
            google_candidates = []
        for gc in google_candidates:
            if gc["id"] in seen_google_ids:
                continue
            seen_google_ids.add(gc["id"])
            try:
                meta = asyncio.run(fetch_pano_by_id(gc["id"]))
            except Exception as e:
                print(f"Google date lookup failed for {gc['id']}: {e}")
                continue
            if not meta:
                continue
            for entry in meta["dates"]:
                nodes.append({
                    "key": node_key("google", entry["id"]), "source": "google", "id": entry["id"],
                    "lat": gc["lat"], "lon": gc["lon"], "date": entry["label"],
                })

        try:
            apple_candidates = apple_tile_panos(lat, lon)
        except Exception as e:
            print(f"Apple lookup failed near ({lat}, {lon}): {e}")
            apple_candidates = {}
        for p in apple_candidates.values():
            if p.id in seen_apple_ids:
                continue
            if haversine_m(lat, lon, p.lat, p.lon) > max_dist_m:
                continue
            seen_apple_ids.add(p.id)
            nodes.append({
                "key": node_key("apple", p.id), "source": "apple", "id": p.id,
                "lat": p.lat, "lon": p.lon, "date": format_date(p.date),
                "_pano": p,  # kept for download_lookaround (needs the object, not just the id)
            })

    return nodes, points
