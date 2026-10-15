"""How high the roads and their bridges run."""
import numpy as np

from streetview_to_3d.build.postprocess.world.roads.Deck import Deck
from streetview_to_3d.build.postprocess.world.roads.Network import SHOULDER_M
from streetview_to_3d.build.postprocess.world.tools.osm_roads import KEPT, colour, kept_ways, width
from streetview_to_3d.build.postprocess.world.tools.reach import ROADS_M
from streetview_to_3d.build.postprocess.world.tools.seams import meet

MATCH_M = 6.0         # a pano stands on a road whose side is within this (GPS is metres off)
ON_DECK_M = 2.5       # ... or on a bridge if no more than this under its deck
ROAD_MEET_M, ROAD_MEET_MAX_M = 40.0, 10.0  # roads ease to the scene's road height over this; not if this far apart
BRIDGE_CLEAR_M = {"road": 4.5, "water": 2.5}     # min deck clearance over each
FIELD_PAD_M = 50.0    # the roads' height field reaches this far past ROADS_M


class RoadHeight:
    """The roads' height (m) at east/north points, and their bridges' decks (decks).

    The ground (elevation: Elevation) under the roads smoothed into one field (Network.heights);
    the roads the panos stand on, and those joining them near the scene, eased onto the scene's
    road height (site: Site, its ground); decks clear the roads and water (Water) they cross."""

    def __init__(self, network, elements, site, elevation, water):
        import shapely
        to_xy, near = site.to_xy, site.ground.at
        self.raw = network.heights(elevation, (-ROADS_M - FIELD_PAD_M, -ROADS_M - FIELD_PAD_M),
                                   (ROADS_M + FIELD_PAD_M, ROADS_M + FIELD_PAD_M))

        def to_scene(xy, h):
            d, g, _ = near(xy)
            return meet(h, g, d, ROAD_MEET_M, ROAD_MEET_MAX_M)

        def crossed(road_h):
            def low(xy):
                out = np.full(len(xy), -np.inf)
                road = shapely.contains_xy(network.all, *xy.T) if not network.all.is_empty else np.zeros(len(xy), bool)
                out = np.where(road, road_h(xy) + BRIDGE_CLEAR_M["road"], out)
                level = water._at(water.level, xy, -np.inf)
                return np.where(water.inside(xy), np.maximum(out, level + BRIDGE_CLEAR_M["water"]), out)
            return low
        # only the roads the panos stand on, and those joining them, meet the scene's road height
        found = _bridges(elements, to_xy)
        ids = _standing(elements, to_xy, site.cameras, elevation(site.cameras),
                        _decks(found, self.raw, crossed(self.raw)))
        ids = _joined(elements, to_xy, ids, lambda xy: near(xy)[0] < ROAD_MEET_M)
        self._met = _region(elements, to_xy, ids, SHOULDER_M)
        self._to_scene = to_scene
        self.decks = _decks(found, self, crossed(self), to_scene, ids)

    def __call__(self, xy):
        import shapely
        h = self.raw(xy)
        on = shapely.contains_xy(self._met, *xy.T) if len(xy) else np.zeros(0, bool)
        if on.any():
            h[on] = self._to_scene(xy[on], h[on])
        return h


def _layer(tags):
    try:
        return float(str(tags.get("layer", "1")).split(";")[0])
    except ValueError:
        return 1.0


def _bridges(elements, to_xy):
    """[(centreline (n, 2), width, colour, layer, OSM way ids)] of the kept bridges.

    OSM splits a long bridge into ways; they are merged end to end and take
    the widest piece's width."""
    import shapely
    ways = []
    for e in elements:
        tags = e.get("tags", {})
        if e["type"] != "way" or tags.get("highway") not in KEPT:
            continue
        if tags.get("bridge", "no") == "no" or len(e.get("geometry") or []) < 2:
            continue
        ways.append((shapely.LineString(to_xy(e["geometry"])), width(tags), colour(tags), _layer(tags), e["id"]))
    if not ways:
        return []
    merged = shapely.get_parts(shapely.line_merge(shapely.union_all([w[0] for w in ways])))
    tree = shapely.STRtree([w[0] for w in ways])
    out = []
    for line in merged:
        mine = [ways[i] for i in tree.query(line, predicate="covers")] or \
               [ways[i] for i in tree.query(line.buffer(0.5), predicate="intersects")]
        if not mine or line.length < 1:
            continue
        widest = max(mine, key=lambda w: w[1])
        out.append((shapely.get_coordinates(line), widest[1], widest[2], max(w[3] for w in mine),
                    {w[4] for w in mine}))
    return out


def _decks(bridges, ground, under, pull=None, pulled=None):
    """Each bridge as a Deck; pulled onto the scene only if one of its ways is in pulled (or pulled is None)."""
    return [Deck(b, ground, under, pull if pulled is None or b[4] & pulled else None) for b in bridges
            if np.linalg.norm(np.diff(b[0], axis=0), axis=1).sum() >= 1]


def _standing(elements, to_xy, cams, cam_h, decks_):
    """OSM ids of the roads the panos (east/north cams, heights cam_h) stand on.

    A road counts if its side is within MATCH_M; under a bridge, the bridge
    wins when the pano is within ON_DECK_M of its deck."""
    import shapely
    ways = kept_ways(elements, to_xy)
    if not ways or not len(cams):
        return set()
    lines = [shapely.LineString(w[1]) for w in ways]
    tree = shapely.STRtree(lines)
    deck = {i: d for d in decks_ for i in d.ids}
    reach = max(w[2] for w in ways) / 2 + MATCH_M
    out = set()
    for c, h in zip(cams, cam_h):
        p = shapely.Point(c)
        mine = [k for k in tree.query(p, predicate="dwithin", distance=reach)
                if shapely.distance(lines[k], p) <= ways[k][2] / 2 + MATCH_M]
        up = []
        for k in mine:
            d = deck.get(ways[k][0])
            if ways[k][4] and d is not None:
                top = d.h[np.argmin(np.linalg.norm(d.xy - c, axis=1))]
                if h > top - ON_DECK_M:
                    up.append((abs(h - top), ways[k][0]))
        if up:
            out.add(min(up)[1])
        else:
            out.update(ways[k][0] for k in mine if not ways[k][4])
    return out


def _joined(elements, to_xy, ids, near):
    """ids plus every ground road transitively joined to them at a node where near(xy) is true."""
    from collections import defaultdict
    ways = kept_ways(elements, to_xy)
    at = defaultdict(list)
    for w in ways:
        for n in w[3]:
            at[n].append(w)
    out = set(ids)
    todo = [w for w in ways if w[0] in out]
    while todo:
        w = todo.pop()
        close = near(w[1][:len(w[3])])
        for n, ok in zip(w[3], close):
            for v in at[n] if ok else ():
                if v[0] not in out and not v[4]:
                    out.add(v[0])
                    todo.append(v)
    return out


def _region(elements, to_xy, ids, pad_m):
    """The ground roads among ids, widened by pad_m each side, as one prepared shape."""
    import shapely
    shapes = [shapely.LineString(w[1]).buffer(w[2] / 2 + pad_m, quad_segs=2)
              for w in kept_ways(elements, to_xy) if w[0] in ids and not w[4]]
    g = shapely.union_all(shapes) if shapes else shapely.Polygon()
    shapely.prepare(g)
    return g
