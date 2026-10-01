"""The roads around the scene, from OpenStreetMap, laid on the terrain.

OSM draws a road as its centreline, so each is given a width -- its own
"width" tag, else "lanes" x LANE_M, else a guess by kind (WIDTH_M) -- and
those a simple game map keeps -- car roads, pedestrian streets, squares
-- become one surface (Network): each spot one thing, crossings joined.
Only what lies on the open ground (_on_ground): no corridors, metro
passages or walkways. As a game lays roads, the road decides its height
(Network.heights, the ground smoothed) and the land fits itself to the
road (Network.adapt). Laid LIFT_M above that: near the scene as points
(points), spaced as the other map points are there, so the panos can
colour them and they meet DA3's; further out as a flat surface (surface).
Coloured by its "surface" tag (SURFACE), else asphalt for a road and
paving for a path.

Bridges (bridges, bridge_points) are laid apart: each whole (OSM's pieces
joined), not on the ground but from the road at one end to the road at
the other, arched a little (ARCH_PER of their length, ARCH_MAX at most),
clearing what they cross, smooth, with a DECK_M edge down each
side, so seen from low they are a deck, not a sheet. Tunnels are not
drawn.

Called by terrain.build: the points join terrain.ply, the surface is
roads.ply.
"""
import numpy as np

from .geometry import sample_quad

LANE_M = 3.2
WIDTH_M = {"motorway": 14, "trunk": 12, "primary": 10, "secondary": 9, "tertiary": 8,
           "motorway_link": 6, "trunk_link": 6, "primary_link": 6, "secondary_link": 6,
           "tertiary_link": 6, "residential": 6, "unclassified": 6, "living_street": 5,
           "service": 4, "pedestrian": 4, "track": 3}
NARROW_M = 2.0            # footways, paths, cycleways, steps, anything else
SKIP = ("proposed", "construction", "raceway", "bus_stop", "platform", "elevator", "corridor")
LIFT_M = 0.15
ARCH_PER, ARCH_MAX, DECK_M = 0.03, 2.0, 1.0
LAYER_M, GRADE = 5.0, 0.06      # a bridge on a higher layer this much higher per layer; rising no steeper
SIDE = np.array([0.60, 0.60, 0.58]) * 0.85      # a bridge's edge: concrete, in shade
SURFACE = {"asphalt": (0.33, 0.33, 0.35), "concrete": (0.60, 0.60, 0.58), "paving_stones": (0.58, 0.51, 0.45),
           "sett": (0.50, 0.47, 0.44), "tiles": (0.63, 0.54, 0.47), "wood": (0.50, 0.38, 0.26),
           "gravel": (0.58, 0.55, 0.49), "fine_gravel": (0.62, 0.58, 0.50), "compacted": (0.58, 0.54, 0.46),
           "ground": (0.48, 0.41, 0.31), "dirt": (0.48, 0.41, 0.31), "earth": (0.48, 0.41, 0.31),
           "sand": (0.72, 0.66, 0.52), "grass": (0.36, 0.50, 0.30)}
PATHS = ("footway", "path", "pedestrian", "steps", "cycleway", "bridleway")
KEPT = ("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "road",
        "living_street", "pedestrian", "motorway_link", "trunk_link", "primary_link", "secondary_link",
        "tertiary_link")
STRIP_M = 2.5         # a road's surface: corners this far apart along it, as fine as the land near it
PROFILE_M, PROFILE_CELL_M = 12.0, 2.0     # roads' heights: the ground smoothed this much, on this grid
SINK_M, SHOULDER_M = 0.25, 4.0            # the land under a road this far beneath it, easing back over this
MATCH_M = 6.0         # a pano stands on a road whose side is this near it at most: Google's GPS is metres off
ON_DECK_M = 2.5       # ... on a bridge over it if no more than this under the deck (a road under a bridge is
                      # BRIDGE_CLEAR_M 4.5 m down at least)


def _width(tags):
    for key, per in (("width", 1.0), ("lanes", LANE_M)):
        try:
            v = float(str(tags[key]).split()[0].replace(",", "."))
            if v > 0:
                return v * per
        except (KeyError, ValueError):
            pass
    return WIDTH_M.get(tags["highway"], NARROW_M)


def _colour(tags):
    s = tags.get("surface")
    if s in SURFACE:
        return np.array(SURFACE[s])
    return np.array(SURFACE["concrete" if tags["highway"] in PATHS or s == "paving" else "asphalt"])


def _on_ground(tags):
    """Whether a way lies on the open ground: not a tunnel or bridge, not
    indoors (a station's, a mall's corridors), not on another level (layer:
    a metro passage below, a deck or walkway above)."""
    if tags.get("tunnel", "no") != "no" or tags.get("bridge", "no") != "no" or tags.get("indoor", "no") != "no":
        return False
    try:
        return float(str(tags.get("layer", "0")).split(";")[0]) == 0
    except ValueError:
        return True


def _area(e):
    g = e.get("geometry") or []
    return e.get("tags", {}).get("area") == "yes" and len(g) >= 4 and g[0] == g[-1]


def _ways(elements, to_xy):
    """(centreline (n, 2) east/north metres, width m, colour, tags) of the
    roads among osm.fetch's elements that lie on the ground (_on_ground),
    squares (_area) not."""
    for e in elements:
        tags = e.get("tags", {})
        if e["type"] != "way" or "highway" not in tags or tags["highway"] in SKIP:
            continue
        if not _on_ground(tags) or _area(e) or len(e.get("geometry") or []) < 2:
            continue
        yield to_xy(e["geometry"]), _width(tags), _colour(tags), tags


def lines(elements, to_xy):
    """[(centreline (n, 2) east/north metres, width m, colour)] of the roads
    on the ground (_ways)."""
    return [(xy, w, c) for xy, w, c, _ in _ways(elements, to_xy)]


PAVEMENT = (0.64, 0.62, 0.57)
KERB = (0.74, 0.72, 0.67)
PAINT = (0.82, 0.81, 0.73)
KERB_M = .14
PAVING_JOINT = (.43, .42, .39)


def _positive(tags, key, default):
    try:
        value = float(str(tags[key]).split()[0].replace(",", "."))
        return value if np.isfinite(value) and value > 0 else default
    except (KeyError, ValueError, IndexError):
        return default


def _sidewalks(xy, width, tags):
    """Tagged left/right ribbons; 'separate' is handled by mapped footway ways."""
    import shapely
    line = shapely.LineString(xy)
    for side, sign in (("left", 1), ("right", -1)):
        value = tags.get("sidewalk:" + side, tags.get("sidewalk:both", tags.get("sidewalk", "no")))
        if value not in ("yes", "both", side):
            continue
        w = min(6.0, _positive(tags, f"sidewalk:{side}:width",
                            _positive(tags, "sidewalk:both:width", 1.8)))
        # Difference of one-sided buffers keeps turns joined without overlapping the road.
        outer = line.buffer(sign * (width / 2 + w), single_sided=True, join_style="mitre")
        inner = line.buffer(sign * width / 2, single_sided=True, join_style="mitre")
        yield shapely.difference(outer, inner), line.offset_curve(sign * (width / 2 + w / 2)), w


def _lane_dividers(xy, width, tags):
    """Conservative dashed dividers from explicit lane counts, clear of segment ends."""
    import shapely
    if tags.get("lane_markings") == "no" or tags.get("surface", "asphalt") not in ("asphalt", "concrete"):
        return
    lanes = _positive(tags, "lanes", 1)
    if lanes != int(lanes) or not 2 <= lanes <= 8 or width / lanes < 2.4:
        return
    line = shapely.LineString(xy)
    for k in range(1, int(lanes)):
        offset = width * (k / lanes - .5)
        stripe = line if abs(offset) < 1e-6 else line.offset_curve(offset)
        if stripe.geom_type != "LineString":
            continue
        for start in np.arange(width, stripe.length - width - 2.5, 6):
            from shapely.ops import substring
            yield substring(stripe, start, start + 2.5).buffer(.09, cap_style="flat")


def _surface_colours(xy, colour):
    """Quiet, deterministic variation without high-contrast noise."""
    colour = np.asarray(colour)
    cell = np.floor(xy / .8)
    n = np.sin(cell[:, 0] * 12.9898 + cell[:, 1] * 78.233) * 43758.5453
    n -= np.floor(n)
    return np.clip(colour * (1 + (n[:, None] - .5) * .055), 0, 1)


class Network:
    """The roads a simple game map keeps, as one surface: car roads and
    pedestrian streets (KEPT), squares (area=yes), and explicitly tagged
    sidewalks. Each spot belongs to one surface: paint, carriageway,
    pedestrian street, square, then pavement. Pavements never cover roads.

    shapes: {colour: its surface}; strips, fills: what points are laid over
    (points) before they are kept to shapes."""

    def __init__(self, elements, to_xy):
        import shapely
        from collections import defaultdict
        layers = defaultdict(list)                  # priority -> [(shape, colour)]
        self.strips, self.fills = [], []
        self.elements, self.to_xy = elements, to_xy
        self.street_lines = [(xy, w, tags) for xy, w, _, tags in _ways(elements, to_xy)
                             if tags["highway"] in KEPT and tags["highway"] not in PATHS]
        # Junctions from shared OSM nodes: paint stops before a crossing or side street.
        from collections import Counter
        degree, node_xy = Counter(), {}
        for e in elements:
            tags = e.get("tags", {})
            if tags.get("highway") not in KEPT or not _on_ground(tags) or _area(e):
                continue
            nodes = e.get("nodes", [])
            coordinates = to_xy(e.get("geometry", []))
            for k, (node, coordinate) in enumerate(zip(nodes, coordinates)):
                degree[node] += 1 if k in (0, len(nodes) - 1) else 2
                node_xy[node] = coordinate
        junctions = shapely.union_all([shapely.Point(node_xy[node]).buffer(8)
                                      for node, n in degree.items() if n > 2])
        obstacles = shapely.union_all([shapely.make_valid(shapely.Polygon(to_xy(e["geometry"])))
                                      for e in elements if "building" in e.get("tags", {})
                                      and len(e.get("geometry", [])) >= 4])
        for xy, w, c, tags in _ways(elements, to_xy):
            if tags["highway"] == "footway" and tags.get("footway") == "sidewalk":
                shape = shapely.LineString(xy).buffer(w / 2, cap_style="flat", join_style="mitre")
                layers[3].append((shape, np.array(PAVEMENT)))
                self.strips.append((xy, w, np.array(PAVEMENT)))
                continue
            if tags["highway"] not in KEPT:
                continue
            if tags["highway"] not in PATHS:
                for shape, centre, pavement_width in _sidewalks(xy, w, tags):
                    shape = shapely.difference(shape, obstacles)
                    layers[3].append((shape, np.array(PAVEMENT)))
                    for part in shapely.get_parts(centre):
                        if part.geom_type == "LineString":
                            self.strips.append((shapely.get_coordinates(part), pavement_width, np.array(PAVEMENT)))
                for mark in _lane_dividers(xy, w, tags):
                    mark = shapely.difference(mark, junctions)
                    if mark.is_empty:
                        continue
                    layers[-1].append((mark, np.array(PAINT)))
                    for part in shapely.get_parts(mark):
                        if part.geom_type == "Polygon":
                            self.fills.append((shapely.get_coordinates(part.exterior), np.array(PAINT)))
            layers[1 if tags["highway"] in PATHS else 0].append(
                (shapely.LineString(xy).buffer(w / 2, quad_segs=2, cap_style="flat"), c))
            self.strips.append((xy, w, c))
        # Explicit crossing nodes become broad zebra bars, aligned to their road.
        for e in elements:
            tags = e.get("tags", {})
            if e.get("type") != "node" or tags.get("highway") != "crossing":
                continue
            if tags.get("crossing") in ("no", "unmarked") or tags.get("crossing:markings") == "no":
                continue
            if tags.get("crossing:markings") not in ("zebra", "yes") and tags.get("crossing") not in ("zebra", "marked", "traffic_signals"):
                continue
            location = to_xy([e])[0]
            if not self.street_lines:
                continue
            xy, width, _ = min(self.street_lines, key=lambda r: shapely.LineString(r[0]).distance(shapely.Point(location)))
            line = shapely.LineString(xy)
            if line.distance(shapely.Point(location)) > width / 2 + 2:
                continue
            at = line.project(shapely.Point(location))
            a = np.array(line.interpolate(max(0, at - .5)).coords[0])
            b = np.array(line.interpolate(min(line.length, at + .5)).coords[0])
            along = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            side = np.array([-along[1], along[0]])
            centre = np.array(line.interpolate(at).coords[0])
            for offset in np.arange(-width / 2 + .5, width / 2 - .3, 1.1):
                q = np.array([centre + along * u + side * v for u, v in
                              ((-1.8, offset), (1.8, offset), (1.8, offset + .55), (-1.8, offset + .55))])
                layers[-2].append((shapely.Polygon(q), np.array(PAINT)))
                self.fills.append((q, np.array(PAINT)))
        for e in elements:
            tags = e.get("tags", {})
            if e["type"] == "way" and _area(e) and tags.get("highway") in KEPT and _on_ground(tags):
                xy = to_xy(e["geometry"])
                layers[2].append((shapely.make_valid(shapely.Polygon(xy)), _colour(tags)))
                self.fills.append((xy, _colour(tags)))
        self.shapes, taken = {}, shapely.Polygon()
        for p in sorted(layers):
            by = defaultdict(list)
            for g, c in layers[p]:
                by[tuple(c)].append(g)
            for c, gs in by.items():
                g = shapely.difference(shapely.union_all(gs), taken)
                self.shapes[c] = shapely.union_all([self.shapes[c], g]) if c in self.shapes else g
            taken = shapely.union_all([taken] + [g for g, _ in layers[p]])
        pavement = self.shapes.get(PAVEMENT, shapely.Polygon())
        if not pavement.is_empty:
            curb = shapely.difference(pavement, pavement.buffer(-.18, join_style="mitre"))
            self.shapes[PAVEMENT] = shapely.difference(pavement, curb)
            self.shapes[KERB] = curb
            # Same fill candidates cover both colours, with each clipped to its own final shape.
            self.fills += [(xy, np.array(KERB)) for xy, c in list(self.fills) if tuple(c) == PAVEMENT]
            self.strips += [(xy, w, np.array(KERB)) for xy, w, c in list(self.strips) if tuple(c) == PAVEMENT]
        # Real polygon seams survive triangle scattering, unlike tiny vertex noise.
        paving = self.shapes.get(PAVEMENT, shapely.Polygon())
        if not paving.is_empty:
            lo_x, lo_y, hi_x, hi_y = paving.bounds
            seams = []
            for axis, lo, hi, other_lo, other_hi in ((0, lo_x, hi_x, lo_y, hi_y), (1, lo_y, hi_y, lo_x, hi_x)):
                for t in np.arange(np.floor(lo / 1.2) * 1.2, hi, 1.2):
                    coords = [(t, other_lo), (t, other_hi)] if axis == 0 else [(other_lo, t), (other_hi, t)]
                    seam = shapely.intersection(shapely.LineString(coords), paving)
                    if not seam.is_empty:
                        seams.append(seam.buffer(.035, cap_style="flat"))
            joints = shapely.intersection(shapely.union_all(seams), paving)
            self.shapes[PAVEMENT] = shapely.difference(paving, joints)
            self.shapes[PAVING_JOINT] = joints
            self.strips += [(xy, w, np.array(PAVING_JOINT)) for xy, w, c in list(self.strips) if tuple(c) == PAVEMENT]
            self.fills += [(xy, np.array(PAVING_JOINT)) for xy, c in list(self.fills) if tuple(c) == PAVEMENT]
        self.all = taken
        for g in self.shapes.values():
            shapely.prepare(g)
        shapely.prepare(self.all)

    def edges(self, step_m=STRIP_M):
        """(n, 2): points along the surface's outline, step_m apart."""
        import shapely
        return shapely.get_coordinates(shapely.segmentize(shapely.boundary(self.all), step_m)) \
            if not self.all.is_empty else np.zeros((0, 2))

    def heights(self, ground, lo, hi):
        """f(east/north (n, 2)) -> the roads' own height: the ground under
        them (ground(xy)), smoothed over PROFILE_M along and across them --
        a road rises and falls smoothly, not with every bump of a 30 m
        height map that has the buildings and trees in it -- and one height
        field for all, so roads meet at a junction. Within lo..hi (the
        area, east/north)."""
        import shapely
        from scipy.ndimage import gaussian_filter, map_coordinates
        c = PROFILE_CELL_M
        shape = tuple(np.ceil((np.asarray(hi) - lo) / c).astype(int) + 1)
        gx, gy = np.meshgrid(lo[0] + c * np.arange(shape[0]), lo[1] + c * np.arange(shape[1]), indexing="ij")
        cells = np.stack([gx.ravel(), gy.ravel()], 1)
        on = shapely.contains_xy(self.all.buffer(c), *cells.T) if not self.all.is_empty else np.zeros(len(cells), bool)
        total, count = np.zeros(len(cells)), on.astype(float)
        total[on] = ground(cells[on])
        sigma = PROFILE_M / c
        total = gaussian_filter(total.reshape(shape), sigma)
        count = gaussian_filter(count.reshape(shape), sigma)
        field = np.where(count > 1e-3, total / np.maximum(count, 1e-9), np.nan)

        def f(xy):
            if not len(xy):
                return np.zeros(0)
            v = map_coordinates(np.nan_to_num(field), ((xy - lo) / c).T, order=1, mode="nearest")
            w = map_coordinates(np.isfinite(field).astype(float), ((xy - lo) / c).T, order=1, mode="nearest")
            return np.where(w > 0.5, v, ground(xy))
        return f

    def adapt(self, xy, h, road_h):
        """The land's heights h at east/north points xy made to fit the
        roads, as a game's terrain is: under a road SINK_M beneath it, and
        from its edge over SHOULDER_M easing back to the land's own."""
        import shapely
        from scipy.spatial import cKDTree
        edge = self.edges(1.0)
        if not len(edge):
            return h
        d = cKDTree(edge).query(xy, distance_upper_bound=SHOULDER_M)[0]
        d[shapely.contains_xy(self.all, *xy.T)] = 0.0
        near = np.flatnonzero(d < SHOULDER_M)
        if not len(near):
            return h
        t = np.clip(d[near] / SHOULDER_M, 0, 1)
        w = 1 - t * t * (3 - 2 * t)
        h = h.copy()
        h[near] += (road_h(xy[near]) - SINK_M - h[near]) * w
        return h


def _layer(tags):
    try:
        return float(str(tags.get("layer", "1")).split(";")[0])
    except ValueError:
        return 1.0


def bridges(elements, to_xy):
    """[(centreline (n, 2), width, colour, layer, OSM way ids)]: the bridges a game map
    keeps (KEPT), each whole -- OSM breaks a long one into ways end to end
    (a bridge, a viaduct in pieces), and a piece is not a bridge: drawn as
    one, each came down to the ground at every join. Joined where exactly
    two meet (line_merge), each takes the widest of its pieces."""
    import shapely
    ways = []
    for e in elements:
        tags = e.get("tags", {})
        if e["type"] != "way" or tags.get("highway") not in KEPT:
            continue
        if tags.get("bridge", "no") == "no" or len(e.get("geometry") or []) < 2:
            continue
        ways.append((shapely.LineString(to_xy(e["geometry"])), _width(tags), _colour(tags), _layer(tags), e["id"]))
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


class Deck:
    """A bridge's deck (bridges): its centreline every metre (xy, at metres
    along) and height there (h) -- from the road's height at one end to the
    other's (ground), arched a little, but at least under(xy) over its span
    (the lowest it may be there: clear of a road, the water), LAYER_M more
    a layer up, rising to that and back no steeper than GRADE: one smooth
    deck; where it touches the scene, the scene's own (pull(xy, h) -> h).
    The land fits under it (under_decks), as under a road. ids: its OSM
    ways."""

    def __init__(self, bridge, ground, under, pull=None):
        from scipy.ndimage import gaussian_filter1d
        xy, self.width, self.colour, layer, self.ids = bridge
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cum = np.r_[0, np.cumsum(seg)]
        self.length = cum[-1]
        self.at = np.r_[np.arange(0, self.length, 1.0), self.length]
        self.xy = np.stack([np.interp(self.at, cum, xy[:, 0]), np.interp(self.at, cum, xy[:, 1])], 1)
        t = self.at / max(self.length, 1e-9)
        h0, h1 = ground(xy[[0, -1]]) + LIFT_M
        h = h0 + (h1 - h0) * t + min(ARCH_PER * self.length, ARCH_MAX) * 4 * t * (1 - t)
        h = np.maximum(h, under(self.xy) + LAYER_M * max(layer - 1, 0.0))
        h[0], h[-1] = h0, h1
        for order in (slice(None), slice(None, None, -1)):         # no steeper than GRADE from either end
            hh = h[order]
            for i in range(1, len(hh)):
                hh[i] = min(hh[i], hh[i - 1] + GRADE * 1.0)
        if pull is not None:
            h = pull(self.xy, h)
        if len(h) > 4:
            h[1:-1] = gaussian_filter1d(h, 2.0)[1:-1]
        self.h = h

    def sample(self, s):
        """(centreline every s m, its height, sideways unit vectors)."""
        at = np.r_[np.arange(0, self.length, s), self.length]
        centre = np.stack([np.interp(at, self.at, self.xy[:, 0]), np.interp(at, self.at, self.xy[:, 1])], 1)
        ahead = np.gradient(centre, axis=0) if len(centre) > 1 else np.array([[1.0, 0.0]])
        side = np.stack([-ahead[:, 1], ahead[:, 0]], 1) / np.maximum(np.linalg.norm(ahead, axis=1), 1e-9)[:, None]
        return centre, np.interp(at, self.at, self.h), side

    def footprint(self):
        import shapely
        return shapely.LineString(self.xy).buffer(self.width / 2, quad_segs=2, cap_style="flat")


def decks(bridges, ground, under, pull=None, pulled=None):
    """Each bridge a Deck, pulled onto the scene (pull) only if one of its
    OSM ways is in pulled (every one, with pulled None)."""
    return [Deck(b, ground, under, pull if pulled is None or b[4] & pulled else None) for b in bridges
            if np.linalg.norm(np.diff(b[0], axis=0), axis=1).sum() >= 1]


def _kept(elements, to_xy):
    """[(OSM id, centreline (n, 2), width, node ids, is a bridge)]: the
    ways a game map keeps (KEPT) on the ground, and its bridges."""
    out = []
    for e in elements:
        tags = e.get("tags", {})
        if e["type"] != "way" or tags.get("highway") not in KEPT or _area(e) or len(e.get("geometry") or []) < 2:
            continue
        bridge = tags.get("bridge", "no") != "no"
        if bridge or _on_ground(tags):
            out.append((e["id"], to_xy(e["geometry"]), _width(tags), e.get("nodes") or [], bridge))
    return out


def standing(elements, to_xy, cams, cam_h, decks_):
    """The OSM ids of the roads the panos stand on (east/north cams, their
    heights cam_h): each pano's roads, those whose side is within MATCH_M
    of it -- where a bridge (decks_) passes over one, the bridge if the
    pano is up on its deck (within ON_DECK_M under it), else the roads
    below. Only these, and those joining them (joined), meet the scene's
    road: a road passing near it, over it or under it is another road."""
    import shapely
    ways = _kept(elements, to_xy)
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


def joined(elements, to_xy, ids, near):
    """ids and the roads on the ground joining them, again and again, at
    a node near the scene (near(east/north (n, 2)) -> bool): a side street
    meets the scene's road as the road it joins does, not a step lower."""
    from collections import defaultdict
    ways = _kept(elements, to_xy)
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


def region(elements, to_xy, ids, pad_m):
    """The ground roads among ids, each pad_m wider each side, as one
    prepared shape (empty if none)."""
    import shapely
    shapes = [shapely.LineString(w[1]).buffer(w[2] / 2 + pad_m, quad_segs=2)
              for w in _kept(elements, to_xy) if w[0] in ids and not w[4]]
    g = shapely.union_all(shapes) if shapes else shapely.Polygon()
    shapely.prepare(g)
    return g


def deck_edges(decks_, step_m=STRIP_M):
    """(n, 2): points along every deck's outline, step_m apart."""
    import shapely
    if not decks_:
        return np.zeros((0, 2))
    return shapely.get_coordinates(shapely.segmentize(shapely.boundary(
        shapely.union_all([d.footprint() for d in decks_])), step_m))


def under_decks(decks_, xy, h):
    """The land's heights h at east/north points xy, fitted under the
    bridges: never above SINK_M beneath a deck over it, and beside one
    no higher than easing up from that over SHOULDER_M -- the height map
    has the bridge in it (it sees tops), so the land bulged over it."""
    from scipy.spatial import cKDTree
    if not decks_:
        return h
    centre = np.concatenate([d.xy for d in decks_])
    height = np.concatenate([d.h for d in decks_])
    half = np.concatenate([np.full(len(d.xy), d.width / 2) for d in decks_])
    dist, k = cKDTree(centre).query(xy, distance_upper_bound=float(half.max()) + SHOULDER_M)
    near = np.flatnonzero(np.isfinite(dist))
    near = near[dist[near] < half[k[near]] + SHOULDER_M]
    if not len(near):
        return h
    out = np.clip((dist[near] - half[k[near]]) / SHOULDER_M, 0, 1)      # 0 under the deck, 1 at its shoulder's edge
    top = height[k[near]] - SINK_M
    h = h.copy()
    h[near] = np.minimum(h[near], top + (h[near] - top) * out * out * (3 - 2 * out))
    return h


def bridge_surface(decks_):
    """(points (n, 3) world, colours (n, 3), triangles (m, 3)): each deck,
    and its edge DECK_M down each side."""
    pts, cols, faces, n = [], [], [], 0
    for deck in decks_:
        centre, h, side = deck.sample(STRIP_M)
        width, colour = deck.width, deck.colour
        k = len(centre)
        rows = []            # each a line of corners along the bridge: deck left, right; the sides' top and bottom
        for edge in (-width / 2, width / 2):
            rows.append((centre + edge * side, h, colour))
        for edge in (-width / 2, width / 2):
            rows += [(centre + edge * side, h, SIDE), (centre + edge * side, h - DECK_M, SIDE)]
        for xy_, h_, c in rows:
            pts.append(np.column_stack([xy_[:, 0], -h_, xy_[:, 1]]))
            cols.append(np.tile(c, (k, 1)))
        i = np.arange(k - 1)
        for a in (0, 2, 4):               # the deck, then each side: a strip between rows a and a + 1
            A, B = n + a * k + i, n + (a + 1) * k + i
            faces += [np.stack([A, B, B + 1], 1), np.stack([A, B + 1, A + 1], 1)]
        n += 6 * k
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), int)
    return np.concatenate(pts), np.concatenate(cols), np.concatenate(faces)


def bridge_points(decks_, step):
    """(points (n, 3) world, colours (n, 3)): each deck and its edge down
    each side, step(xy) apart."""
    pts, cols = [], []
    for deck in decks_:
        s = float(step(deck.xy.mean(0)[None])[0])
        if s > deck.width:
            continue
        centre, h, side = deck.sample(s)
        across = np.arange(-deck.width / 2, deck.width / 2 + 1e-9, s)
        top = (centre[:, None] + across[None, :, None] * side[:, None]).reshape(-1, 2)
        pts.append(np.column_stack([top[:, 0], -np.repeat(h, len(across)), top[:, 1]]))
        cols.append(np.tile(deck.colour, (len(top), 1)))
        down = np.arange(s, DECK_M + 1e-9, s)
        for edge in (-deck.width / 2, deck.width / 2):
            e = centre + edge * side
            pts.append(np.column_stack([np.repeat(e[:, 0], len(down)),
                                        -(h[:, None] - down[None]).ravel(), np.repeat(e[:, 1], len(down))]))
            cols.append(np.tile(SIDE, (len(e) * len(down), 1)))
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(pts), np.concatenate(cols)


def coverage(roads, step_m=0.5):
    """f(outline (n, 2)) -> the share of its edges lying on a road (within
    the road's half width, less half a metre: a pavement edge is not in it)."""
    from scipy.spatial import cKDTree
    pts, half = [], []
    for xy, width, _ in roads:
        for a, b in zip(xy[:-1], xy[1:]):
            n = max(2, int(np.linalg.norm(b - a) / step_m))
            pts.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
            half.append(np.full(n, width / 2 - 0.5))
    if not pts:
        return lambda xy: 0.0
    tree, half = cKDTree(np.concatenate(pts)), np.concatenate(half)

    def f(xy):
        e = np.concatenate([a + (c - a) * np.linspace(0, 1, max(2, int(np.linalg.norm(c - a) / step_m)))[:, None]
                            for a, c in zip(xy[:-1], xy[1:])])
        d, k = tree.query(e)
        return float((d < half[k]).mean())
    return f


def near(roads, margin_m, step_m=1.0):
    """f(east/north (n, 2)) -> whether within margin_m of a road's side."""
    from scipy.spatial import cKDTree
    pts, half = [], []
    for xy, width, _ in roads:
        for a, b in zip(xy[:-1], xy[1:]):
            n = max(2, int(np.linalg.norm(b - a) / step_m))
            pts.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
            half.append(np.full(n, width / 2 + margin_m))
    if not pts:
        return lambda xy: np.zeros(len(xy), bool)
    tree, half = cKDTree(np.concatenate(pts)), np.concatenate(half)
    reach = float(half.max())

    def f(xy):
        d, k = tree.query(xy, distance_upper_bound=reach)
        return d < half[np.minimum(k, len(half) - 1)]
    return f


BELOW_M = 0.05        # the surface this far under the points, so where both are, the points show


def _curb_faces(net, ground, where):
    """Visible vertical edge of raised pavements; clipped before segmentizing."""
    import shapely
    pavement = shapely.union_all([net.shapes.get(PAVEMENT, shapely.Polygon()),
                                  net.shapes.get(KERB, shapely.Polygon()), net.shapes.get(PAVING_JOINT, shapely.Polygon())])
    boundary = shapely.intersection(shapely.boundary(pavement), where)
    for part in shapely.get_parts(shapely.segmentize(boundary, STRIP_M)):
        if part.geom_type not in ("LineString", "LinearRing"):
            continue
        xy = shapely.get_coordinates(part)
        for a, b in zip(xy[:-1], xy[1:]):
            en = np.array([a, b])
            bottom = ground(en) + LIFT_M
            yield np.r_[np.c_[en[:, 0], -bottom, en[:, 1]],
                        np.c_[en[::-1, 0], -(bottom[::-1] + KERB_M), en[::-1, 1]]]


def surface(net, ground, away):
    """(points (n, 3) world, colours (n, 3), triangles (m, 3)): the
    Network's surface on the ground (the land's own height: ground), LIFT_M
    up, where away (a shapely geometry: what is not near the scene)."""
    import shapely
    pts, cols, faces, n = [], [], [], 0
    for colour, shape in net.shapes.items():
        area = shapely.segmentize(shapely.intersection(shape, away), STRIP_M)
        tris = shapely.get_parts(shapely.constrained_delaunay_triangles(area)) if not area.is_empty else []
        if not len(tris):
            continue
        corners = shapely.get_coordinates(tris).reshape(-1, 4, 2)[:, :3]
        xy, idx = np.unique(np.round(corners.reshape(-1, 2), 3), axis=0, return_inverse=True)
        pts.append(np.column_stack([xy[:, 0], -(ground(xy) + LIFT_M - BELOW_M + (KERB_M if colour in (PAVEMENT, KERB, PAVING_JOINT) else 0)), xy[:, 1]]))
        cols.append(_surface_colours(xy, colour))
        faces.append(idx.reshape(-1, 3) + n)
        n += len(xy)
    for quad in _curb_faces(net, ground, away):
        pts.append(quad)
        cols.append(np.tile(np.array(KERB) * .82, (4, 1)))
        faces.append(np.array([[0, 1, 2], [0, 2, 3]]) + n)
        n += 4
    from .street_details import quads
    for quad, colour in quads(net, ground):
        if not shapely.intersects(shapely.Polygon(quad[:, [0, 2]]).convex_hull, away):
            continue
        pts.append(quad)
        cols.append(np.tile(colour, (4, 1)))
        faces.append(np.array([[0, 1, 2], [0, 2, 3]]) + n)
        n += 4
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), int)
    return np.concatenate(pts), np.concatenate(cols), np.concatenate(faces)


def points(net, step, ground, where):
    """(points (n, 3) world, colours (n, 3)): the Network's surface as
    points, LIFT_M over the ground, only where(xy) (near the scene).

    step(xy): point spacing at east/north points; ground(xy): height of the
    ground there. Laid along each way and over each outline, then each kept
    only where its colour's surface is (Network: one thing a spot)."""
    import shapely
    xys, cols = [], []
    for xy, width, colour in net.strips:
        for a, b in zip(xy[:-1], xy[1:]):
            length = np.linalg.norm(b - a)
            if length < 1e-6:
                continue
            s = float(step(((a + b) / 2)[None])[0])
            if s > width:
                continue
            along = a + (b - a) * (np.arange(max(1, int(length / s)) + 1) / max(1, int(length / s)))[:, None]
            side = np.array([-(b - a)[1], (b - a)[0]]) / length
            across = np.arange(-width / 2, width / 2 + 1e-9, s)
            xys.append((along[:, None, :] + across[None, :, None] * side).reshape(-1, 2))
            cols.append(np.tile(colour, (len(xys[-1]), 1)))
    for xy, colour in net.fills:
        s = float(step(xy.mean(0)[None])[0])
        lo, hi = xy.min(0), xy.max(0)
        counts = np.maximum(1, np.ceil((hi - lo) / s).astype(int))
        gx, gy = np.meshgrid(lo[0] + (np.arange(counts[0]) + .5) * (hi[0] - lo[0]) / counts[0],
                             lo[1] + (np.arange(counts[1]) + .5) * (hi[1] - lo[1]) / counts[1])
        xys.append(np.stack([gx.ravel(), gy.ravel()], 1))
        cols.append(np.tile(colour, (len(xys[-1]), 1)))
    if not xys:
        return np.zeros((0, 3)), np.zeros((0, 3))
    xy, col = np.concatenate(xys), np.concatenate(cols)
    keep = where(xy)
    xy, col = xy[keep], col[keep]
    mine = np.zeros(len(xy), bool)
    for colour, shape in net.shapes.items():
        of = np.flatnonzero((col == colour).all(1))
        if len(of):
            mine[of] = shapely.contains_xy(shape, *xy[of].T)
    xy, col = xy[mine], col[mine]
    raised = np.all(col == PAVEMENT, axis=1) | np.all(col == KERB, axis=1) | np.all(col == PAVING_JOINT, axis=1)
    varied = col.copy()
    for colour in net.shapes:
        mine = np.all(col == colour, axis=1)
        varied[mine] = _surface_colours(xy[mine], colour)
    pts = np.column_stack([xy[:, 0], -(ground(xy) + LIFT_M + raised * KERB_M), xy[:, 1]])
    curb_pts = []
    for quad in _curb_faces(net, ground, net.all.envelope):
        a, b, _, d = quad
        s = max(.1, float(step(((a + b)[[0, 2]] / 2)[None])[0]))
        count = max(1, int(np.ceil(np.linalg.norm(b - a) / s)))
        q = a + ((np.arange(count) + .5) / count)[:, None] * (b - a) + .5 * (d - a)
        curb_pts.append(q[where(q[:, [0, 2]])])
    if curb_pts:
        curb_pts = np.concatenate(curb_pts)
        pts = np.r_[pts, curb_pts]
        varied = np.r_[varied, np.tile(np.array(KERB) * .82, (len(curb_pts), 1))]
    from .street_details import quads
    detail_pts, detail_cols = [], []
    for q, colour in quads(net, ground):
        centre = q.mean(0)[[0, 2]]
        if not where(centre[None])[0]:
            continue
        s = max(.08, float(step(centre[None])[0]))
        samples = sample_quad(q, s)
        detail_pts.append(samples)
        detail_cols.append(np.tile(colour, (len(samples), 1)))
    if detail_pts:
        pts = np.concatenate([pts, *detail_pts])
        varied = np.concatenate([varied, *detail_cols])
    return pts, varied
