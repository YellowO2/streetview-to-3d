"""Solo mode: DA3 on each Google pano on its own, no links.

The linked walk (walk_graph, join_segments) exists to put panos in one
frame by testing them against each other. Once each pano is placed by its
own GPS, heading and Google's depth instead (google_base), nothing needs
linking, so every pano is reconstructed alone -- with the DA3 that is best
at one pano on its own (config.DA3_SOLO_MODEL_REPO) -- and every place
Google has a pano can take part, not only the dots the walk could reach:

1. prepare: per dot, the Google candidates of the best-ranked date that has
   any there (within google_base's MAX_YEARS of the others); plus Google's official neighbours of those panos
   (google_base.neighbours), each as a place of its own.
2. reconstruct (GPU): rate each place's candidates, keep the best one's
   own cloud. One piece per pano, each in its own DA3 frame.

Apple panos take no part: they have no depth map to be placed by.
"""
import asyncio
import time

import aiohttp

from streetview_to_3d.services.http_headers import BROWSER_HEADERS
from streetview_to_3d.services.streetview_fetch import DA3_ONLY_ZOOM, download_pano_by_id, format_date, run_async

# GPU window: building the solo model from disk inside the call, then one
# DA3 run per candidate (~2 s measured for rating, rounded up).
MODEL_BUILD_S = 45.0
SECONDS_PER_CANDIDATE = 4.0
SAVE_BUFFER_S = 10.0


def _places(date_graphs, catalog):
    """{dot: [(key, path, lat, lon), ...]}: the Google candidates of the
    best-ranked date (date_graphs is in rank order) that has any there,
    within MAX_YEARS of the best-ranked date with Google at all -- a place
    captured ten years apart is a different street."""
    from streetview_to_3d.google_base.fetch import MAX_YEARS
    places, year = {}, None
    for g in date_graphs:
        for dot, bucket in g["dot_candidates"].items():
            google = [c for c in bucket if catalog[c[0]]["source"] == "google"]
            if not google:
                continue
            year = year or int(str(g["date"])[:4])
            if dot not in places and abs(int(str(g["date"])[:4]) - year) <= MAX_YEARS:
                places[dot] = google
    return places


async def _add_neighbours(prep, places):
    """Google's official neighbours of the chosen panos, downloaded and
    added to the catalog, each as a new place (dot) with no road links."""
    from streetview_to_3d.google_base.fetch import neighbours
    from streetview_to_3d.ui.map_selection.candidates import node_key
    catalog = prep["catalog"]
    seeds = [dict(catalog[c[0]]) for bucket in places.values() for c in bucket]
    known = {c["id"] for c in catalog.values()}
    async with aiohttp.ClientSession(headers=BROWSER_HEADERS) as s:
        metas = [m for m in await neighbours(seeds, *prep["center"], s) if m.id not in known]
    paths = await asyncio.gather(*[download_pano_by_id(m.id, zoom=DA3_ONLY_ZOOM) for m in metas])
    dot = max([c["dot"] for c in catalog.values()] + [len(prep["points"]) - 1]) + 1
    for m, path in zip(metas, paths):
        if not path:
            continue
        key = node_key("google", m.id)
        catalog[key] = {"dot": dot, "source": "google", "id": m.id, "lat": m.lat, "lon": m.lon,
                        "date": format_date(m.date), "heading": m.heading, "pitch": m.pitch,
                        "roll": m.roll, "elevation": m.elevation}
        places[dot] = [(key, path, m.lat, m.lon)]
        dot += 1


def prepare(prep):
    """prep (build.prepare_pathfind's) with its solo places: prep["solo"]."""
    t0 = time.monotonic()
    places = _places(prep["date_graphs"], prep["catalog"])
    n_scene = len(places)
    run_async(_add_neighbours(prep, places))
    prep["solo"] = places
    print(f"solo: {n_scene} place(s) on the route + {len(places) - n_scene} neighbour pano(s), "
          f"{sum(map(len, places.values()))} candidate(s)  [{time.monotonic() - t0:.1f}s]")
    return prep


def estimate_gpu_seconds(places):
    return MODEL_BUILD_S + SECONDS_PER_CANDIDATE * sum(map(len, places.values())) + SAVE_BUFFER_S


def reconstruct(places, catalog, rate_pano, deadline):
    """One piece per place: its best-rated candidate's own cloud.
    Returns [(clouds, metadata), ...] as join_segments.pieces_to_output."""
    pieces = []
    for n_done, (dot, bucket) in enumerate(sorted(places.items())):
        if time.monotonic() >= deadline:
            print(f"solo: out of time, {len(places) - n_done} place(s) left")
            break
        rated = [(rate_pano(path), key) for key, path, _, _ in bucket]
        (score, pose, pts, cols, n_kept, n_total), key = max(rated, key=lambda r: r[0][0])
        print(f"solo: dot {dot} {key}: {n_kept}/{n_total} views kept, {len(pts)} points"
              f"{f' (best of {len(bucket)})' if len(bucket) > 1 else ''}", flush=True)
        if pose is None:
            continue
        c = catalog[key]
        pieces.append(({key: (pts, cols)},
                       {key: {"lat": c["lat"], "lon": c["lon"], "date": c["date"],
                              "position": list(map(float, pose[0])), "rotation": [list(map(float, r)) for r in pose[1]],
                              "n_views_kept": n_kept, "n_views_total": n_total}}))
    return pieces
