"""OSM's map data round the scene, from Overpass or OpenFreeMap's tiles.

Buildings, roads, street objects and water from Overpass, retried across mirrors, OpenFreeMap's
tiles standing in on failure; Overpass's answers kept across runs (_cache_path). Credit "(c)
OpenStreetMap contributors"."""
import hashlib
import json
import logging
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from streetview_to_3d.build.postprocess.world.OpenFreeMap import OpenFreeMap
from streetview_to_3d.build.postprocess.world.tools.reach import FULL_M, FAR_BUILDING_M, M_PER_LAT, OSM_M, ROADS_M, USER_AGENT

logger = logging.getLogger(__name__)

OVERPASS_URLS = ("https://overpass-api.de/api/interpreter",
                 "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
                 "https://overpass.private.coffee/api/interpreter")
TIMEOUT_S = 20          # per request; answers normally take ~2 s
BUSY_WAIT_S, ROUNDS = 10.0, 3
BUDGET_S = 30.0         # total Overpass time before the tiles stand in
FAR_ROADS = "^(motorway|trunk|primary|secondary|tertiary)(_link)?$"   # past FULL_M, only these roads
WATER_WAYS = ('["natural"="water"]', '["waterway"="riverbank"]', '["natural"="coastline"]',
              '["landuse"~"^(reservoir|basin)$"]')
WATER_RELATIONS = ('["natural"="water"]', '["waterway"="riverbank"]')


class OpenStreetMap:
    """OSM round site (Site): elements() and water_at(), OpenFreeMap's tiles kept between them."""

    def __init__(self, site):
        self.site, self.tiles = site, OpenFreeMap()

    def elements(self):
        """Overpass elements (with geometry) round the scene's cameras (_fetch's reach), or none
        if neither Overpass nor OpenFreeMap answers."""
        site = self.site
        try:
            return _fetch(*site.origin, OSM_M, ROADS_M, M_PER_LAT, site.m_per_lon, water_m=OSM_M,
                          cams=site.to_ll(site.cameras), tiles=self.tiles)
        except (OSError, ValueError) as e:
            logger.warning(f"terrain: no OpenStreetMap ({e!r})")
            return []

    def water_at(self, latlon):
        """Whether each (lat, lon) lies in OSM water: OpenFreeMap's areas, else one Overpass
        request (lakes and rivers, not the sea), kept across runs (_cache_path)."""
        return _water_at(latlon, self.tiles)


def _cache_path(query):
    """Where the answer to query is kept: one file per query, shared by every run."""
    from streetview_to_3d.tools.paths import DATA_DIR
    return os.path.join(DATA_DIR, "map_cache", hashlib.sha1(query.encode()).hexdigest()[:20] + ".json")


def _cached(path, query, key):
    """The answer kept at path for query, or None."""
    if os.path.exists(path):
        with open(path) as f:
            kept = json.load(f)
        if isinstance(kept, dict) and kept.get("query") == query:
            return kept[key]
    return None


def _keep(path, query, key, answer):
    """Keep answer at path, written whole or not at all, as runs may read it at once."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    partial = f"{path}.{os.getpid()}.part"
    with open(partial, "w") as f:
        json.dump({"query": query, key: answer}, f)
    os.replace(partial, path)


def _fetch(lat0, lon0, buildings_m, roads_m, m_per_lat, m_per_lon, water_m=0, cams=None, tiles=None):
    """Overpass elements (with geometry) around the cameras' box (cams: (lats, lons), else the centre).

    Everything within FULL_M; big buildings to buildings_m, main roads to
    roads_m, water outlines to water_m. Cached across runs (_cache_path). If Overpass
    fails, OpenFreeMap's tiles (tiles: an OpenFreeMap; not cached); if both fail, Overpass's error."""
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
    cache = _cache_path(query)
    kept = _cached(cache, query, "elements")
    if kept is not None:
        return kept
    from concurrent.futures import ThreadPoolExecutor
    tiles = tiles or OpenFreeMap()
    pool = ThreadPoolExecutor(1)
    tiles = pool.submit(tiles.elements, lat0, lon0, buildings_m, roads_m, m_per_lat, m_per_lon,
                        water_m, cams=cams)
    pool.shutdown(wait=False)
    try:
        elements = _ask(query, budget=BUDGET_S)["elements"]
    except (OSError, ValueError) as e:
        try:
            elements = tiles.result()
        except (OSError, ValueError) as e2:
            logger.warning(f"osm: no OpenFreeMap tiles either ({e2!r})")
            raise e
        logger.warning(f"osm: Overpass gave no answer ({e!r}); OpenFreeMap's tiles instead, "
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


def _water_at(latlon, tiles):
    """OpenStreetMap.water_at; tiles: an OpenFreeMap."""
    if not len(latlon):
        return []
    try:
        return tiles.water_at(latlon)
    except (OSError, ValueError) as e:
        logger.warning(f"osm: no OpenFreeMap water ({e!r}), Overpass's")
    areas = "".join(f'area.a{t};' for t in WATER_WAYS if "coastline" not in t)
    query = "[out:json][timeout:60];" + "".join(
        f'is_in({lat:.7f},{lon:.7f})->.a;({areas});make p i="{i}",ids=set(id());out;'
        for i, (lat, lon) in enumerate(latlon))
    cache = _cache_path(query)
    kept = _cached(cache, query, "water")
    if kept is not None:
        return kept
    found = {int(e["tags"]["i"]): e["tags"]["ids"] != "" for e in _ask(query, budget=BUDGET_S)["elements"]}
    answer = [found.get(i, False) for i in range(len(latlon))]
    _keep(cache, query, "water", answer)
    return answer
