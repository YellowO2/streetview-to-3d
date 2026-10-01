"""Gather every Google panorama along a street corridor (no GPU)."""
from streetview_to_3d.services.geo import haversine_m
from streetview_to_3d.services.streetview_fetch import fetch_panos_by_id, run_async
from streetview_to_3d.ui.map_selection.candidates import node_key

# Real selection-graph nodes within this distance of each other collapse
# into ONE dot (see corridor_points) -- real coverage is frequently
# double/triple-sampled at the same real spot (a road captured in both
# directions, a pedestrian path running alongside a road, a backpack
# capture re-walking the same stretch), which otherwise shows up as
# several near-duplicate dots each independently competing for a date/
# candidates instead of one dot with the union of everyone's real
# candidates. 8.0, not 5.0 -- checked on real NTU data after a first pass
# at 5.0: normal real node spacing along a single path is itself often
# 8-11m, so 5.0 already correctly avoided merging genuinely distinct
# waypoints; 8.0 is a deliberate small step up.
MERGE_DIST_M = 8.0

# Separate graphs closer than this get one bridge edge (see
# bridge_components) -- about the reach of a real link between neighbouring
# panos, so DA3 has as good a chance there as along a street.
BRIDGE_DIST_M = 15.0


def corridor_points(edges):
    """Real (lat, lon) dots + structural adjacency straight from the
    corridor's own already-confirmed edges -- edges: list of ((lat1,
    lon1, pano_id1), (lat2, lon2, pano_id2)) pairs, each a real,
    already-connected pair of selected panos (not a single ordered
    polyline: the corridor can branch or loop, so edges aren't assumed
    to trace one path in list order).

    No synthetic in-between sampling -- a dot starts as exactly one real
    selection-graph node, not an interpolated point along a straight
    line between two of them. Real selection-graph nodes within
    MERGE_DIST_M of each other then collapse into the SAME dot (see
    MERGE_DIST_M's own docstring). NOT transitive: merging is a single
    greedy pass over yet-unclaimed raw nodes -- each still-unclaimed node
    becomes a new dot's seed and claims every other still-unclaimed node
    within MERGE_DIST_M of ITSELF, but an already-claimed node never goes
    on to claim its own further-out neighbors. A real transitive union-
    find (A-B close, B-C close => A,B,C all one dot even if A-C alone
    is 70m+ apart) was tried and confirmed BROKEN on real NTU data: a
    single, ordinary, densely-sampled 72m stretch chain-collapsed into
    ONE dot, since consecutive real nodes along it were each individually
    under threshold. The greedy version caps that: a claimed node can
    still anchor a real connection out to whatever's left over, but can't
    silently drag its own whole neighborhood in behind it. A merged dot's
    own position is the centroid of everything folded into it.

    Returns (points, adjacency, members). adjacency: {dot_index:
    [neighbor_dot_index, ...]} -- the corridor's own real dot-to-dot
    structure, independent of which real panos end up at either dot. This
    is what the pathfind algorithm walks dot-by-dot over (see
    reconstruct/walk_graph.py). members: per dot, the ids of the selected
    panos folded into it -- its candidates, with their older dates (see
    fetch_corridor_nodes).
    """
    raw_points: list[tuple[float, float]] = []
    raw_ids: list[str] = []
    raw_index_by_id: dict[str, int] = {}

    def raw_index_for(node):
        lat, lon, pano_id = node
        idx = raw_index_by_id.get(pano_id)
        if idx is None:
            idx = len(raw_points)
            raw_points.append((lat, lon))
            raw_ids.append(pano_id)
            raw_index_by_id[pano_id] = idx
        return idx

    raw_edges = [(raw_index_for(a), raw_index_for(b)) for a, b in edges]

    # Greedy, non-transitive merge -- see this function's own docstring
    # for why NOT union-find. O(n^2) distance checks -- fine at real-
    # world selection-graph sizes (a few thousand nodes at most).
    dot_index_by_raw: dict[int, int] = {}
    points: list[tuple[float, float]] = []
    members: list[list[str]] = []
    claimed = [False] * len(raw_points)

    for i in range(len(raw_points)):
        if claimed[i]:
            continue
        lat_i, lon_i = raw_points[i]
        cluster = [i]
        claimed[i] = True
        for j in range(len(raw_points)):
            if claimed[j]:
                continue
            if haversine_m(lat_i, lon_i, *raw_points[j]) <= MERGE_DIST_M:
                cluster.append(j)
                claimed[j] = True

        lat = sum(raw_points[m][0] for m in cluster) / len(cluster)
        lon = sum(raw_points[m][1] for m in cluster) / len(cluster)
        dot_idx = len(points)
        points.append((lat, lon))
        members.append([raw_ids[m] for m in cluster])
        for m in cluster:
            dot_index_by_raw[m] = dot_idx

    adjacency: dict[int, list[int]] = {i: [] for i in range(len(points))}

    def connect(i, j):
        if i == j:
            return
        if j not in adjacency[i]:
            adjacency[i].append(j)
        if i not in adjacency[j]:
            adjacency[j].append(i)

    for a, b in raw_edges:
        connect(dot_index_by_raw[a], dot_index_by_raw[b])

    bridge_components(points, adjacency, connect)
    return points, adjacency, members


def bridge_components(points, adjacency, connect, max_gap_m: float = BRIDGE_DIST_M):
    """Join graphs Google never linked where they come within max_gap_m.

    A walked (scout) path often meets a road a few metres from it without
    either pano linking the other. A bridge edge here lets the walk try the
    DA3 pair test there: if it holds they become one piece, if not the walk
    restarts past it as it does after any failed test. Kruskal over the
    closest dot pairs, so each two groups get their nearest crossing and no
    more -- one good join is all a piece needs.
    """
    comp = list(range(len(points)))

    def find(i):
        while comp[i] != i:
            comp[i] = comp[comp[i]]
            i = comp[i]
        return i

    for i, ns in adjacency.items():
        for j in ns:
            comp[find(i)] = find(j)

    pairs = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if find(i) != find(j):
                d = haversine_m(*points[i], *points[j])
                if d <= max_gap_m:
                    pairs.append((d, i, j))
    for _, i, j in sorted(pairs):
        if find(i) != find(j):
            comp[find(i)] = find(j)
            connect(i, j)


def fetch_corridor_nodes(edges):
    """Every Google pano at each real corridor dot (see corridor_points):
    the selected panos folded into it, each with its real historical dates
    (one candidate per date). The selection already holds them, so nothing
    is searched for again -- one fetch_pano_by_id each, all at once, for
    the dates. (Apple Look Around was here too and needed a search around
    each dot; its GPS sits 1-1.6 m off Google's and its depth is poor, so
    it is gone, and the search with it.)

    A pano belongs to one dot (corridor_points folds each into exactly
    one); a date shared by two of a dot's panos is kept once.

    Returns (buckets, points, adjacency, elevations): buckets is
    {point_index: [{key, source, id, lat, lon, date}, ...]} -- each dot's
    own separate set of panos, not one shared pool. points is the
    corridor's own real node list. adjacency is the dot-to-dot structural
    graph (see corridor_points). elevations is metres above sea level per
    dot, which every pano lookup already returns -- taking it here is what
    saves placement from re-fetching every panorama later just to read it.
    """
    points, adjacency, members = corridor_points(edges)
    ids = [pano_id for m in members for pano_id in m]
    meta_by_id = dict(zip(ids, run_async(fetch_panos_by_id(ids))))

    buckets = {i: [] for i in range(len(points))}
    elevations = [None] * len(points)
    for i, pano_ids in enumerate(members):
        seen = set()
        for pano_id in pano_ids:
            meta = meta_by_id.get(pano_id)
            if not meta:
                continue
            if elevations[i] is None:
                elevations[i] = meta.get("elevation")
            for entry in meta["dates"]:
                if entry["id"] in seen:
                    continue
                seen.add(entry["id"])
                # each date's own pose: older captures are separate drives,
                # metres away and often facing another way
                buckets[i].append({
                    "key": node_key("google", entry["id"]), "source": "google", "id": entry["id"],
                    "lat": entry.get("lat", meta["lat"]), "lon": entry.get("lon", meta["lon"]),
                    "date": entry["label"],
                    "heading": entry.get("heading", meta.get("heading")),
                    "pitch": entry.get("pitch", meta.get("pitch")),
                    "roll": entry.get("roll", meta.get("roll")),
                })

    return buckets, points, adjacency, elevations
