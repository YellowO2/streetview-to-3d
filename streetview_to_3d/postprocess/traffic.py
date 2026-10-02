"""Where cars may drive, for the viewer to move a few along (effects/traffic.js).

The car roads (CARS) on the ground and over its bridges within REACH_M of
the cameras: further off, nobody watches a car. Through the scene too, on
its own road at its own ground's height (terrain's road_h meets it there),
in their lanes beside whatever DA3 parked at its kerb.

A car never drives into a building: OSM's widths are guesses, and its
buildings (as fitted onto DA3's walls) can stand nearer a road's line than
its lane. Each point keeps how much room it has to the nearest wall
(room: half the road's width at most); a car keeps to its lane only as far
as that lets it, and where a car's width does not fit (CLEAR_M) the road
is cut.
The ground's roads are split where they cross, so a car can turn there; a
bridge joins only at its ends, as a road under it is no junction. Their
height is the road's own (terrain's road_h; a bridge's deck). Cars keep to
the side the country drives on (its panos' country code: LEFT).

Written to traffic.json beside scene.json (its "traffic"), east/north metres:
    {"side": "left" | "right",
     "colours": {"body": [[r, g, b], ...], "glass": .., "tyre": .., "head": .., "tail": ..},
     "roads": [{"width": m, "points": [[e, n, h], ...], "room": [m, ...]}, ...]}
each road a stretch between junctions or ends, its points STEP_M apart; two
roads meet where an end of one is within JOIN_M of an end of the other
(traffic.js).
"""
import json
import os

import numpy as np

from .roads import KEPT, PATHS, _area, _on_ground, _width

FILENAME = "traffic.json"
CARS = tuple(k for k in KEPT if k not in PATHS)
REACH_M = 100.0
STEP_M = 4.0
MIN_M = 20.0              # a stretch shorter than this is left out
JOIN_M = 1.0
CLEAR_M = 1.1             # a road this near a wall is too narrow for a car (half its width and some)
# the countries that drive on the left (ISO 3166-1 alpha-2)
LEFT = {"GB", "IE", "IM", "JE", "GG", "MT", "CY", "JP", "SG", "MY", "BN", "HK", "MO", "TH", "ID", "TL",
        "IN", "LK", "BD", "BT", "NP", "PK", "MV", "AU", "NZ", "PG", "FJ", "WS", "TO", "SB", "KI", "TV",
        "NR", "CK", "NU", "ZA", "NA", "BW", "ZW", "ZM", "MW", "MZ", "TZ", "KE", "UG", "LS", "SZ", "MU",
        "SC", "JM", "TT", "BB", "BS", "GY", "SR", "AG", "DM", "GD", "KN", "LC", "VC", "VG", "VI", "KY",
        "BM", "FK", "AI", "MS", "TC", "SH"}
# the colours cars come in, each car its own while they last (traffic.js);
# their glass, tyres and lights
BODY = [(0.88, 0.88, 0.86), (0.08, 0.08, 0.09), (0.68, 0.69, 0.70), (0.32, 0.33, 0.35),
        (0.55, 0.09, 0.08), (0.12, 0.20, 0.40), (0.18, 0.30, 0.22), (0.72, 0.66, 0.55),
        (0.42, 0.58, 0.72)]
GLASS = (0.14, 0.17, 0.21)
TYRE = (0.06, 0.06, 0.06)
HEAD = (0.95, 0.93, 0.82)
TAIL = (0.62, 0.08, 0.06)


def side(pano_id):
    """"left" or "right": the side cars keep to where pano_id was taken."""
    import aiohttp
    from streetlevel import streetview
    from streetview_to_3d.services.http_headers import BROWSER_HEADERS
    from streetview_to_3d.services.streetview_fetch import run_async

    async def country():
        async with aiohttp.ClientSession(headers=BROWSER_HEADERS) as session:
            pano = await streetview.find_panorama_by_id_async(pano_id, session=session)
            return pano.country_code if pano else None
    return "left" if run_async(country()) in LEFT else "right"


def roads(elements, to_xy, height, decks, keep, walls=()):
    """[(width m, (n, 3) east/north/height every STEP_M, room (n,) m)]: the
    car roads (elements: osm.fetch's; to_xy their geometry's), on the ground
    at height(xy) and over decks (roads.Deck, those of car roads), kept only
    where keep(xy) (n, 2) is true and CLEAR_M clear of walls (buildings'
    outlines, (m, 2) each); room: how far the nearest wall is, half the
    width at most."""
    import shapely
    wall = shapely.union_all([shapely.make_valid(shapely.Polygon(w)) for w in walls if len(w) >= 3])
    shapely.prepare(wall)

    def clear(xy):
        return shapely.distance(wall, shapely.points(xy)) if not wall.is_empty else np.full(len(xy), np.inf)

    def runs(width, xy, h=None):
        c = clear(xy)
        return [(width, np.c_[xy[i], height(xy[i]) if h is None else h[i]], np.minimum(c[i], width / 2))
                for i in _runs(xy, keep(xy) & (c >= CLEAR_M))]
    ways = [(shapely.LineString(to_xy(e["geometry"])), _width(e["tags"])) for e in elements
            if e.get("type") == "way" and e.get("tags", {}).get("highway") in CARS
            and _on_ground(e["tags"]) and not _area(e) and len(e.get("geometry") or []) >= 2]
    out = []
    if ways:
        tree = shapely.STRtree([w for w, _ in ways])
        for piece in shapely.get_parts(shapely.line_merge(shapely.union_all([w for w, _ in ways]))):
            xy = _every(shapely.get_coordinates(piece))
            width = ways[tree.nearest(shapely.Point(xy[len(xy) // 2]))][1]
            out += runs(width, xy)
    for deck in decks:
        xy = _every(deck.xy)
        out += runs(deck.width, xy, np.interp(_along(xy), deck.at, deck.h))
    return out


def _along(xy):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]


def _every(xy):
    """The line xy (n, 2) as points STEP_M apart, its ends kept."""
    at = _along(xy)
    s = np.linspace(0, at[-1], max(2, int(np.ceil(at[-1] / STEP_M)) + 1))
    return np.c_[np.interp(s, at, xy[:, 0]), np.interp(s, at, xy[:, 1])]


def _runs(xy, ok):
    """The indices of each run of xy (n, 2) where ok (n,), each at least MIN_M long."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(ok, int), 0]))
    return [np.arange(a, b) for a, b in zip(edges[::2], edges[1::2])
            if b - a >= 2 and _along(xy[a:b])[-1] >= MIN_M]


def save(scene_dir, stretches, side_):
    """Write FILENAME (see the module); its name, or None if no road."""
    if not stretches:
        return None
    data = {"side": side_,
            "colours": {"body": [list(c) for c in BODY], "glass": list(GLASS), "tyre": list(TYRE),
                        "head": list(HEAD), "tail": list(TAIL)},
            "roads": [{"width": round(float(w), 2), "points": np.round(p, 2).tolist(),
                       "room": np.round(r, 2).tolist()} for w, p, r in stretches]}
    with open(os.path.join(scene_dir, FILENAME), "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return FILENAME
