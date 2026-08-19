"""Gather every Google + Apple panorama along a street corridor (no GPU)."""
import asyncio

from services.geo import haversine_m
from services.streetview_fetch import fetch_pano_by_id, format_date
from street_builder.map_selection.candidates import MAX_NODES, apple_tile_panos, nearby_nodes, node_key

# Sample points every this many meters along the clicked route -- the one
# spine used for both gathering (this file) and date-coverage ranking
# (walk_graph.py), instead of relying on wherever real nodes happen to be.
POINT_SPACING_M = 5.0

# A candidate only counts as "at" a sample point within this range. Half
# of POINT_SPACING_M, deliberately -- each dot's own catchment radius
# shouldn't reach into a neighboring dot's territory, so every real pano
# belongs to exactly one dot (see fetch_corridor_nodes).
POINT_MAX_DIST_M = 2.5


def interpolate_points(edges, spacing_m: float = POINT_SPACING_M) -> list[tuple[float, float]]:
    """Evenly-spaced (lat, lon) points along each given edge -- edges: list
    of ((lat1, lon1), (lat2, lon2)) pairs, each a straight line between two
    real, already-connected points. Not a single ordered polyline: the
    corridor can branch or loop, so each edge is sampled independently
    rather than assuming consecutive list order traces one path."""
    points = []
    seen_starts = set()
    for (lat1, lon1), (lat2, lon2) in edges:
        if (lat1, lon1) not in seen_starts:
            points.append((lat1, lon1))
            seen_starts.add((lat1, lon1))
        seg_len = haversine_m(lat1, lon1, lat2, lon2)
        n_steps = max(1, round(seg_len / spacing_m))
        for i in range(1, n_steps + 1):
            t = i / n_steps
            points.append((lat1 + (lat2 - lat1) * t, lon1 + (lon2 - lon1) * t))
    return points


def fetch_corridor_nodes(edges, max_dist_m: float = POINT_MAX_DIST_M):
    """Every Google + Apple pano within max_dist_m of any interpolated
    point along the given edges (see interpolate_points).

    - For each point: nearby_nodes (Google stops) + apple_tile_panos
      (Apple), both metadata only. A newly-seen Google stop gets one extra
      fetch_pano_by_id call for its real historical dates (one graph node
      per date); already-seen stops/panos aren't re-fetched.

    Each real pano is assigned to exactly one dot -- the first (lowest-
    index) dot it's found within max_dist_m of, tracked globally via
    seen_google_ids/seen_apple_ids so a pano already claimed by an earlier
    dot never gets double-counted by a later one. With max_dist_m at half
    POINT_SPACING_M this is normally unambiguous (a pano can't be in range
    of two dots at once), but the "first dot wins" rule still resolves the
    rare boundary case cleanly.

    Returns (buckets, points): buckets is {point_index: [{key, source, id,
    lat, lon, date}, ...]} -- each dot's own separate set of panos, not one
    shared pool. points is the interpolated point list itself.
    """
    points = interpolate_points(edges)

    buckets = {i: [] for i in range(len(points))}
    seen_google_ids = set()
    seen_apple_ids = set()

    for i, (lat, lon) in enumerate(points):
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
                buckets[i].append({
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
            buckets[i].append({
                "key": node_key("apple", p.id), "source": "apple", "id": p.id,
                "lat": p.lat, "lon": p.lon, "date": format_date(p.date),
                "_pano": p,  # kept for download_lookaround (needs the object, not just the id)
            })

    return buckets, points
