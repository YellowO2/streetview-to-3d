"""OpenStreetMap around the scene: its buildings and roads, in one request.

The Overpass API needs no key (credit "© OpenStreetMap contributors") but
is a free public service and is sometimes too busy (a 504 on NTU): each
mirror in OVERPASS_URLS is tried in turn, and the answer is kept beside
the scene (CACHE).

OSM is a flat map -- outlines with no ground height, a building's
"height" measured from its own foot -- so what stands on it takes its
ground from terrain.py.
"""
import json
import os
import urllib.parse
import urllib.request

OVERPASS_URLS = ("https://overpass-api.de/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter",
                 "https://overpass.private.coffee/api/interpreter")
TIMEOUT_S = 60
CACHE = "osm.json"        # beside scene.json: a rebuild never asks again


def fetch(lat0, lon0, buildings_m, roads_m, m_per_lat, m_per_lon, scene_dir=None):
    """Every building way and relation within buildings_m and road way
    within roads_m, geometry included, as Overpass returns them; kept in
    scene_dir's CACHE once had."""
    cache = scene_dir and os.path.join(scene_dir, CACHE)
    if cache and os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    box = lambda r: (f"{lat0 - r / m_per_lat},{lon0 - r / m_per_lon},"
                     f"{lat0 + r / m_per_lat},{lon0 + r / m_per_lon}")
    query = (f'[out:json][timeout:90];(way["building"]({box(buildings_m)});'
             f'relation["building"]({box(buildings_m)});way["highway"]({box(roads_m)}););out geom;')
    data = urllib.parse.urlencode({"data": query}).encode()
    error = None
    for url in OVERPASS_URLS:
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "streetview-to-3d"})
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                elements = json.load(r)["elements"]
            if cache:
                with open(cache, "w") as f:
                    json.dump(elements, f)
            return elements
        except (OSError, ValueError) as e:
            error = e
    raise error
