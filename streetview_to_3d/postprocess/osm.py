"""OpenStreetMap around the scene: its buildings (and building parts, how
landmarks are mapped in 3D), roads and water, in one request.

Water is outlines: ways tagged as water, the coastline (the sea's edge),
and the member ways of water relations -- only those crossing the box, not
a lake's thousands of ways (asked whole, Stockholm's timed out). water.py
joins them into areas.

The Overpass API needs no key (credit "© OpenStreetMap contributors") but
is a free public service and is often too busy (a 504 on NTU, every
mirror for an evening): each mirror in OVERPASS_URLS (the wiki's public
instances) is tried in turn, and once all have refused, all again after
BUSY_WAIT_S -- its usage rules ask a client to wait ~30 s after a 429 --
the request saying who asks (USER_AGENT, as they ask too). The answer is
kept beside the scene (CACHE) with the request it answers -- asking
something new (water, building parts) asks again, once.

OSM is a flat map -- outlines with no ground height, a building's
"height" measured from its own foot -- so what stands on it takes its
ground from terrain.py.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

OVERPASS_URLS = ("https://overpass-api.de/api/interpreter",
                 "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
                 "https://overpass.private.coffee/api/interpreter")
TIMEOUT_S = 180         # a busy server took 72 s for a city centre (Lund, 4 MB)
BUSY_WAIT_S, ROUNDS = 30.0, 2
USER_AGENT = "streetview-to-3d (https://github.com/YellowO2/streetview-to-3d)"
CACHE = "osm.json"        # beside scene.json: a rebuild never asks again
WATER_WAYS = ('["natural"="water"]', '["waterway"="riverbank"]', '["natural"="coastline"]',
              '["landuse"~"^(reservoir|basin)$"]')
WATER_RELATIONS = ('["natural"="water"]', '["waterway"="riverbank"]')


def fetch(lat0, lon0, buildings_m, roads_m, m_per_lat, m_per_lon, scene_dir=None, water_m=0):
    """Every building and building part way and relation within buildings_m,
    road ways and tagged street-object nodes within roads_m, and water outlines within
    water_m, geometry included, as Overpass returns them; kept in
    scene_dir's CACHE once had."""
    box = lambda r: (f"{lat0 - r / m_per_lat},{lon0 - r / m_per_lon},"
                     f"{lat0 + r / m_per_lat},{lon0 + r / m_per_lon}")
    water_ways = "".join(f"way{t}({box(water_m)});" for t in WATER_WAYS) if water_m else ""
    # a water relation's members crossing the box, never the relation itself: out
    # geom would print all of them
    water_members = ("(" + "".join(f"rel{t}({box(water_m)});" for t in WATER_RELATIONS)
                     + f");way(r)({box(water_m)});out geom;") if water_m else ""
    query = (f'[out:json][timeout:180];(way["building"]({box(buildings_m)});'
             f'relation["building"]({box(buildings_m)});way["building:part"]({box(buildings_m)});'
             f'relation["building:part"]({box(buildings_m)});way["highway"]({box(roads_m)});'
             f'node["highway"~"^(crossing|traffic_signals|street_lamp)$"]({box(roads_m)});'
             f'node["amenity"~"^(bench|waste_basket)$"]({box(roads_m)});'
             f'{water_ways});out geom;{water_members}')
    cache = scene_dir and os.path.join(scene_dir, CACHE)
    if cache and os.path.exists(cache):
        with open(cache) as f:
            kept = json.load(f)
        if isinstance(kept, dict) and kept.get("query") == query:
            return kept["elements"]
    elements = _ask(query)["elements"]
    if cache:
        with open(cache, "w") as f:
            json.dump({"query": query, "elements": elements}, f)
    return elements


def _ask(query, sleep=time.sleep):
    """Overpass's JSON answer to query from the first mirror that gives one,
    every mirror asked ROUNDS times, BUSY_WAIT_S apart; the last refusal
    raised if none does. One saying the request itself is wrong (a 4xx not
    429) is raised at once: no other server helps."""
    data = urllib.parse.urlencode({"data": query}).encode()
    error = None
    for attempt in range(ROUNDS):
        if attempt:
            sleep(BUSY_WAIT_S)
        for url in OVERPASS_URLS:
            try:
                req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if 400 <= e.code < 500 and e.code != 429:
                    raise
                error = e
            except (OSError, ValueError) as e:
                error = e
    raise error


def is_water(e):
    """Whether an element fetch returned is a water outline: a water way, or
    a water relation's member (untagged, or tagged as nothing else)."""
    if e.get("type") != "way":
        return False
    tags = e.get("tags", {})
    if tags.get("natural") in ("water", "coastline") or tags.get("waterway") == "riverbank" \
            or tags.get("landuse") in ("reservoir", "basin"):
        return True
    return not any(k in tags for k in ("building", "building:part", "highway"))


WATER_AT_CACHE = "osm_water_at.json"


def water_at(latlon, scene_dir=None):
    """Whether each (lat, lon) lies in one of OSM's water areas (a lake, a
    river, a reservoir -- not the sea, which OSM draws only as its
    coastline), in one request; kept in scene_dir's WATER_AT_CACHE with the
    request it answers."""
    if not len(latlon):
        return []
    areas = "".join(f'area.a{t};' for t in WATER_WAYS if "coastline" not in t)
    query = "[out:json][timeout:60];" + "".join(
        f'is_in({lat:.7f},{lon:.7f})->.a;({areas});make p i="{i}",ids=set(id());out;'
        for i, (lat, lon) in enumerate(latlon))
    cache = scene_dir and os.path.join(scene_dir, WATER_AT_CACHE)
    if cache and os.path.exists(cache):
        with open(cache) as f:
            kept = json.load(f)
        if kept.get("query") == query:
            return kept["water"]
    found = {int(e["tags"]["i"]): e["tags"]["ids"] != "" for e in _ask(query)["elements"]}
    answer = [found.get(i, False) for i in range(len(latlon))]
    if cache:
        with open(cache, "w") as f:
            json.dump({"query": query, "water": answer}, f)
    return answer
