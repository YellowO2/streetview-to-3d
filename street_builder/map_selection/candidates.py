"""Low-level fetch of real Street View / Look Around panoramas near a
location. Used by the map picker and by street_builder/build_graph/.
"""
from streetlevel import streetview
from streetlevel.geo import wgs84_to_tile_coord
from streetlevel.lookaround import lookaround as apple_lookaround

from services.geo import haversine_m as _haversine_m
from services.streetview_fetch import fetch_pano_by_id, run_async

# Both services publish coverage on zoom-17 Slippy Map tiles.
_TILE_ZOOM = 17


def _tile_neighborhood(lat, lon):
    tx, ty = wgs84_to_tile_coord(lat, lon, _TILE_ZOOM)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            yield tx + dx, ty + dy


def google_tile_panos(lat, lon):
    """All Street View panos on the 3x3 tile neighborhood around (lat, lon), keyed by id."""
    seen = {}
    for tx, ty in _tile_neighborhood(lat, lon):
        for p in streetview.get_coverage_tile(tx, ty):
            seen[p.id] = p
    return seen


def apple_tile_panos(lat, lon):
    """All Look Around panos on the 3x3 tile neighborhood around (lat, lon), keyed by id."""
    seen = {}
    for tx, ty in _tile_neighborhood(lat, lon):
        tile = apple_lookaround.get_coverage_tile(tx, ty)
        for p in tile.panos:
            seen[p.id] = p
    return seen


def node_key(source, pano_id):
    return f"{source}:{pano_id}"


DEFAULT_RADIUS_M = 350
MAX_NODES = 200


def nearby_nodes(lat, lon, radius_m=DEFAULT_RADIUS_M, max_nodes=MAX_NODES):
    """Google Street View nodes within radius_m of (lat, lon), distance-sorted,
    plus edges from Street View's own coverage graph.

    Returns (nodes, edges). Node: {key, source, id, lat, lon, heading} --
    no date (tile listing doesn't carry it; see build_graph/fetch_nodes.py
    for the full per-pano fetch that does). Edge: (key_a, key_b).
    """
    try:
        panos = google_tile_panos(lat, lon)
    except Exception as e:
        print(f"Google coverage lookup failed: {e}")
        return [], []

    nodes = []
    for p in panos.values():
        if _haversine_m(lat, lon, p.lat, p.lon) > radius_m:
            continue
        nodes.append({
            "key": node_key("google", p.id),
            "source": "google",
            "id": p.id,
            "lat": p.lat,
            "lon": p.lon,
            "heading": p.heading,
        })
    nodes.sort(key=lambda n: _haversine_m(lat, lon, n["lat"], n["lon"]))
    nodes = nodes[:max_nodes]

    kept_keys = {n["key"] for n in nodes}
    edges = set()
    for key in kept_keys:
        pano_id = key.split(":", 1)[1]  # Google ids are strings, matching `panos`' keys directly
        p = panos.get(pano_id)
        if not p:
            continue
        for link in (p.links or []):
            other_key = node_key("google", link.pano.id)
            if other_key in kept_keys and other_key != key:
                edges.add(tuple(sorted((key, other_key))))

    return nodes, sorted(edges)


def expand_area(center_lat, center_lon, radius_m, max_nodes=2000):
    """Auto-discover the real Street View graph within radius_m of
    (center_lat, center_lon) -- the same real-link expansion
    map_selection/tab.py's _augment_real_links does for one clicked node,
    just driven by a BFS loop instead of a person clicking node by node.
    Lets a whole area (a campus, a district) be selected without clicking
    every node by hand -- feed the result straight in as corridor_edges,
    same shape a manually-built selection already produces.

    This is a PURE single-source BFS: nearby_nodes is used only to locate
    the one real pano nearest to the center point (a geometric tile scan
    can't tell us that without first knowing a real pano id to start
    from), and every other node in the result is discovered strictly by
    walking real per-pano links (fetch_pano_by_id) outward from that one
    seed. A node only ever gets ADDED because it was found as a real-link
    neighbor of an already-visited, in-radius node -- never because it
    happened to be geometrically nearby. That guarantees the whole
    output is one connected component by construction, exactly like
    a person standing at the seed and clicking outward one linked node
    at a time, radius_m capping how far they walk.

    This matters: dumping every geometrically-nearby node in as extra
    seeds (the earlier version of this function did) can silently return
    MULTIPLE disconnected components -- e.g. a center point sitting in
    the middle of a triangular block, equidistant from 3 unconnected
    streets, would have a tile scan grab panos from all 3 (all within
    radius) even though they share no real link. A pure BFS from one
    start node instead correctly returns just the one street the start
    node actually belongs to.

    Only ever expanding FROM a node that's still within radius_m --
    anything found just past the boundary is kept as a leaf in the
    result but never itself expanded further.

    Apple has no real link data (same limitation _augment_real_links
    documents) -- this only ever discovers Google coverage. Not a
    problem in practice: Apple candidates still get pulled in later,
    per corridor dot, by fetch_corridor_nodes's own radius lookup --
    exactly how a manually-clicked selection already works today, since
    clicks were always Google-link-driven too.

    Returns (nodes, edges) -- same shape nearby_nodes/tab.py's
    state["nodes"]/state["edges"] already use.
    """
    seed_nodes, _ = nearby_nodes(center_lat, center_lon, radius_m=min(radius_m, DEFAULT_RADIUS_M))
    google_seeds = [n for n in seed_nodes if n["key"].startswith("google:")]
    if not google_seeds:
        # Nothing to walk real links from -- either nothing nearby at
        # all, or only Apple coverage nearby (no real link data to BFS).
        return (seed_nodes[:1], []) if seed_nodes else ([], [])

    start_node = min(google_seeds, key=lambda n: _haversine_m(center_lat, center_lon, n["lat"], n["lon"]))

    nodes = [start_node]
    edges = []
    by_key = {start_node["key"]: start_node}
    edge_set = set()

    visited = set()
    queue = [start_node["key"]]
    while queue and len(nodes) < max_nodes:
        key = queue.pop(0)
        if key in visited:
            continue
        visited.add(key)

        pano_id = key.split(":", 1)[1]
        try:
            meta = run_async(fetch_pano_by_id(pano_id))
        except Exception as e:
            print(f"expand_area: link fetch failed for {pano_id}: {e}")
            continue
        if not meta:
            continue

        for n in meta["neighbors"]:
            other_key = node_key("google", n["id"])
            if other_key not in by_key:
                new_node = {"key": other_key, "source": "google", "id": n["id"],
                            "lat": n["lat"], "lon": n["lon"], "heading": None}
                nodes.append(new_node)
                by_key[other_key] = new_node
            fe = frozenset((key, other_key))
            if fe not in edge_set:
                edges.append((key, other_key))
                edge_set.add(fe)
            if (other_key not in visited
                    and _haversine_m(center_lat, center_lon, by_key[other_key]["lat"], by_key[other_key]["lon"]) <= radius_m):
                queue.append(other_key)

    return nodes, edges


DEFAULT_CHUNK_SIZE = 30


def split_into_chunks(nodes, edges, chunk_size=DEFAULT_CHUNK_SIZE):
    """Split a node/edge graph (e.g. expand_area's output) into connected
    chunks of roughly chunk_size nodes each -- so a large-scale
    selection (a whole campus) can be processed as many independent,
    GPU-call-sized pieces (see street_builder.main.prepare_pathfind)
    instead of one huge corridor that would blow past a single GPU
    call's own constraints.

    Each chunk is grown via its OWN local BFS (seeded from whichever
    unassigned node comes first, restricted to only-still-unassigned
    neighbors) rather than slicing one global traversal order -- that
    distinction matters: a flat slice of a whole-graph BFS order can
    easily contain nodes that aren't actually connected to each other at
    all (two leaves from different branches, visited back-to-back by
    coincidence of queue timing). Growing each chunk from its own seed,
    only ever stepping to neighbors that are still unclaimed, guarantees
    every chunk is a genuinely connected subgraph -- reachable from its
    own seed via real edges that are themselves entirely within that
    chunk. Naturally handles disconnected components and dead-ends too:
    a chunk just ends up smaller than chunk_size if its local
    unassigned neighborhood runs out first.

    Returns (chunks, known_adjacent_chunk_pairs). chunks: [{"chunk_id":
    str, "nodes": [...], "corridor_edges": [...], "start": (lat, lon),
    "goals": [(lat, lon), ...]}, ...] -- each chunk's own start/goals/
    corridor_edges are exactly what
    street_builder.main.prepare_pathfind(start, goals, corridor_edges)
    needs. known_adjacent_chunk_pairs: [(chunk_id_a, chunk_id_b), ...]
    -- every pair of chunks connected by at least one real edge in the
    original graph (see join_segments.bridge_pieces, which this feeds
    directly) -- usually just consecutive chunks, but a branch or loop
    can connect a chunk to more than one other.
    """
    by_key = {n["key"]: n for n in nodes}
    adjacency: dict[str, list[str]] = {n["key"]: [] for n in nodes}
    for a, b in edges:
        if a in adjacency and b in adjacency:
            adjacency[a].append(b)
            adjacency[b].append(a)

    unassigned = {n["key"] for n in nodes}
    raw_chunks: list[list[str]] = []
    for n in nodes:
        while n["key"] in unassigned:
            # Only reachable here for the FIRST node of a new chunk --
            # subsequent iterations of the outer for-loop hit nodes
            # already claimed by an earlier chunk's own BFS and skip
            # straight past (the `while` condition catches that).
            seed = n["key"]
            chunk_keys = []
            queue = [seed]
            queued = {seed}
            while queue and len(chunk_keys) < chunk_size:
                key = queue.pop(0)
                if key not in unassigned:
                    continue
                chunk_keys.append(key)
                unassigned.discard(key)
                for other in adjacency[key]:
                    if other in unassigned and other not in queued:
                        queued.add(other)
                        queue.append(other)
            if chunk_keys:
                raw_chunks.append(chunk_keys)
            break  # this node is now assigned (or was never reachable) -- move on

    # A chunk with <2 nodes can't stand alone (prepare_pathfind needs a
    # start + at least 1 goal) -- fold it into any chunk it shares a
    # real edge with. Rare (only ever an isolated leftover fragment or a
    # single unconnected node); if truly isolated (no real edge to
    # anything), it's dropped with a warning -- not useful for a
    # corridor reconstruction on its own either way.
    key_to_chunk_idx = {k: i for i, keys in enumerate(raw_chunks) for k in keys}
    for i, keys in enumerate(raw_chunks):
        if len(keys) >= 2 or not keys:
            continue
        target = None
        for k in keys:
            for other in adjacency[k]:
                j = key_to_chunk_idx.get(other)
                if j is not None and j != i:
                    target = j
                    break
            if target is not None:
                break
        if target is not None:
            raw_chunks[target].extend(keys)
            for k in keys:
                key_to_chunk_idx[k] = target
            raw_chunks[i] = []
        else:
            print(f"split_into_chunks: dropping isolated node(s) with no real edge to anything: {keys}")
            raw_chunks[i] = []
    raw_chunks = [keys for keys in raw_chunks if keys]

    key_to_chunk_id = {}
    for i, keys in enumerate(raw_chunks):
        chunk_id = f"chunk{i}"
        for k in keys:
            key_to_chunk_id[k] = chunk_id

    chunks = []
    for i, keys in enumerate(raw_chunks):
        chunk_id = f"chunk{i}"
        chunk_nodes = [by_key[k] for k in keys]
        chunk_edges = [(a, b) for a, b in edges
                       if key_to_chunk_id.get(a) == chunk_id and key_to_chunk_id.get(b) == chunk_id]
        chunks.append({
            "chunk_id": chunk_id,
            "nodes": chunk_nodes,
            "corridor_edges": [((by_key[a]["lat"], by_key[a]["lon"]), (by_key[b]["lat"], by_key[b]["lon"]))
                                for a, b in chunk_edges],
            "start": (chunk_nodes[0]["lat"], chunk_nodes[0]["lon"]),
            "goals": [(n["lat"], n["lon"]) for n in chunk_nodes[1:]],
        })

    adjacent_pairs = set()
    for a, b in edges:
        ca, cb = key_to_chunk_id.get(a), key_to_chunk_id.get(b)
        if ca and cb and ca != cb:
            adjacent_pairs.add(frozenset((ca, cb)))
    known_adjacent_chunk_pairs = [tuple(sorted(pair)) for pair in adjacent_pairs]

    return chunks, known_adjacent_chunk_pairs
