"""OpenStreetMap around the scene from Overpass (buildings, roads, street objects, water),
retried across mirrors, OpenFreeMap's tiles standing in on failure, cached beside the scene.
Credit "(c) OpenStreetMap contributors"."""
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request

OVERPASS_URLS = ("https://overpass-api.de/api/interpreter",
                 "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
                 "https://overpass.private.coffee/api/interpreter")
TIMEOUT_S = 20          # per request; answers normally take ~2 s
BUSY_WAIT_S, ROUNDS = 10.0, 3
BUDGET_S = 30.0         # total Overpass time before the tiles stand in
FULL_M = 250.0          # everything this near the cameras
FAR_BUILDING_M = 80.0   # past FULL_M, only buildings with at least this perimeter
FAR_ROADS = "^(motorway|trunk|primary|secondary|tertiary)(_link)?$"   # past FULL_M, only these roads
USER_AGENT = "streetview-to-3d (https://github.com/YellowO2/streetview-to-3d)"
CACHE = "osm.json"
WATER_WAYS = ('["natural"="water"]', '["waterway"="riverbank"]', '["natural"="coastline"]',
              '["landuse"~"^(reservoir|basin)$"]')
WATER_RELATIONS = ('["natural"="water"]', '["waterway"="riverbank"]')


def tag_number(tags, key):
    """A tag's leading number ("12 m", "3,5" -> 3.5), or None."""
    try:
        return float(str(tags[key]).split()[0].replace(",", "."))
    except (KeyError, ValueError, IndexError):
        return None


def building_kind(tags):
    """The building (or building:part) tag, "yes" if neither."""
    return tags.get("building", tags.get("building:part", "yes"))


def _cached(path, query, key):
    """The answer kept at path for query, or None."""
    if path and os.path.exists(path):
        with open(path) as f:
            kept = json.load(f)
        if isinstance(kept, dict) and kept.get("query") == query:
            return kept[key]
    return None


def _keep(path, query, key, answer):
    if path:
        with open(path, "w") as f:
            json.dump({"query": query, key: answer}, f)


def fetch(lat0, lon0, buildings_m, roads_m, m_per_lat, m_per_lon, scene_dir=None, water_m=0, cams=None,
          log=print):
    """Overpass elements (with geometry) around the cameras' box (cams: (lats, lons), else the centre).

    Everything within FULL_M; big buildings to buildings_m, main roads to
    roads_m, water outlines to water_m. Cached in scene_dir. If Overpass
    fails, OpenFreeMap's tiles (not cached); if both fail, Overpass's error."""
    box = lambda r: (f"{lat0 - r / m_per_lat},{lon0 - r / m_per_lon},"
                     f"{lat0 + r / m_per_lat},{lon0 + r / m_per_lon}")
    water_ways = "".join(f"way{t}({box(water_m)});" for t in WATER_WAYS) if water_m else ""
    # a water relation's members crossing the box, not the relation (out geom would print all of it)
    water_members = ("(" + "".join(f"rel{t}({box(water_m)});" for t in WATER_RELATIONS)
                     + f");way(r)({box(water_m)});out geom;") if water_m else ""
    if cams is not None and len(cams[0]):
        r = min(FULL_M, buildings_m, roads_m)
        near = (f"{min(cams[0]) - r / m_per_lat},{min(cams[1]) - r / m_per_lon},"
                f"{max(cams[0]) + r / m_per_lat},{max(cams[1]) + r / m_per_lon}")
    else:
        near = box(min(FULL_M, buildings_m, roads_m))
    query = (f'[out:json][timeout:{TIMEOUT_S:.0f}];(way["building"]({near});'
             f'way["building"]({box(buildings_m)})(if:length()>={FAR_BUILDING_M:.0f});'
             f'relation["building"]({box(buildings_m)});way["building:part"]({near});'
             f'relation["building:part"]({near});way["highway"]({near});'
             f'way["highway"~"{FAR_ROADS}"]({box(roads_m)});'
             f'node["highway"~"^(crossing|traffic_signals|street_lamp)$"]({near});'
             f'node["amenity"~"^(bench|waste_basket)$"]({near});'
             f'{water_ways});out geom;{water_members}')
    cache = scene_dir and os.path.join(scene_dir, CACHE)
    kept = _cached(cache, query, "elements")
    if kept is not None:
        return kept
    from concurrent.futures import ThreadPoolExecutor
    from streetview_to_3d.postprocess.world import openfreemap
    pool = ThreadPoolExecutor(1)
    tiles = pool.submit(openfreemap.elements, lat0, lon0, buildings_m, roads_m, m_per_lat, m_per_lon,
                        water_m, cams=cams)
    pool.shutdown(wait=False)
    try:
        elements = _ask(query, budget=BUDGET_S)["elements"]
    except (OSError, ValueError) as e:
        try:
            elements = tiles.result()
        except (OSError, ValueError) as e2:
            log(f"osm: no OpenFreeMap tiles either ({e2!r})")
            raise e
        log(f"osm: Overpass gave no answer ({e!r}); OpenFreeMap's tiles instead, "
            f"{sum('building' in x['tags'] for x in elements)} buildings")
        return elements
    _keep(cache, query, "elements", elements)
    return elements


def _ask(query, sleep=time.sleep, budget=None, clock=time.monotonic):
    """Overpass's JSON answer from the first mirror that gives one.

    Each mirror is tried ROUNDS times, BUSY_WAIT_S apart, within budget
    seconds; the last error is raised. A 4xx other than 429 is raised at once."""
    data = urllib.parse.urlencode({"data": query}).encode()
    end = clock() + budget if budget else math.inf
    error = TimeoutError(f"Overpass: no answer within {budget} s")
    for attempt in range(ROUNDS):
        if attempt:
            sleep(min(BUSY_WAIT_S, max(end - clock(), 0)))
        for url in OVERPASS_URLS:
            left = end - clock()
            if left < 1:
                raise error
            try:
                req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=min(TIMEOUT_S, left)) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if 400 <= e.code < 500 and e.code != 429:
                    raise
                error = e
            except (OSError, ValueError) as e:
                error = e
    raise error


def is_water(e):
    """Whether an element is a water outline: a water way, or an untagged water relation member."""
    if e.get("type") != "way":
        return False
    tags = e.get("tags", {})
    if tags.get("natural") in ("water", "coastline") or tags.get("waterway") == "riverbank" \
            or tags.get("landuse") in ("reservoir", "basin"):
        return True
    return not any(k in tags for k in ("building", "building:part", "highway"))


WATER_AT_CACHE = "osm_water_at.json"


def water_at(latlon, scene_dir=None, log=print):
    """Whether each (lat, lon) lies in OSM water: OpenFreeMap's areas, else one
    Overpass request (lakes and rivers, not the sea), cached in scene_dir."""
    if not len(latlon):
        return []
    from streetview_to_3d.postprocess.world import openfreemap
    try:
        return openfreemap.water_at(latlon)
    except (OSError, ValueError) as e:
        log(f"osm: no OpenFreeMap water ({e!r}), Overpass's")
    areas = "".join(f'area.a{t};' for t in WATER_WAYS if "coastline" not in t)
    query = "[out:json][timeout:60];" + "".join(
        f'is_in({lat:.7f},{lon:.7f})->.a;({areas});make p i="{i}",ids=set(id());out;'
        for i, (lat, lon) in enumerate(latlon))
    cache = scene_dir and os.path.join(scene_dir, WATER_AT_CACHE)
    kept = _cached(cache, query, "water")
    if kept is not None:
        return kept
    found = {int(e["tags"]["i"]): e["tags"]["ids"] != "" for e in _ask(query, budget=BUDGET_S)["elements"]}
    answer = [found.get(i, False) for i in range(len(latlon))]
    _keep(cache, query, "water", answer)
    return answer
