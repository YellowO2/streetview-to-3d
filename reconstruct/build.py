"""Orchestrator for the pathfind flow -- wires the pipeline stages
together for the UI (tab.py calls into this):

1. build_graph.build_corridor_graphs: gather candidate panos along the
   real click-graph and split into up to DATE_TOP_N isolated, capped
   per-date graphs (no GPU) -- see build_street_graph/.
2. Download every node referenced by any of those graphs (network, cached).
3. run_pathfind_and_join_gpu: ONE GPU call -- the corridor-search
   algorithm (reconstruct/walk_graph.py) runs entirely
   inside it, producing possibly-several disconnected segments.
4. join_segments_gpu: a SEPARATE GPU call -- bridges segments together
   with real DA3 tests where possible, then GPS-fits + merges whatever's
   still separate into one final point cloud (see
   reconstruct/join_segments.py). Split from step 3
   deliberately: bridging only needs each segment's own already-
   confirmed nodes, nothing from the corridor search itself, so keeping
   it separate lets Join (and bridging behavior) be re-run/re-tuned
   against an already-computed step 3 result without re-paying for the
   whole corridor search each time.

Each of steps 3 and 4 is its own single GPU call, not split further: an
earlier version fell back to a second call within one step (download
everything, retry) if the first didn't reach the end. That's exactly the
pattern that causes 'Expired ZeroGPU proxy token' -- each @spaces.GPU
call requests a fresh session credential, and a second request can
arrive after the first one's already aged out. The top-N date filter
already keeps each step's own work bounded, so there's no real need for
a fallback call within either one.
"""
import asyncio
import json
import os
import time

from services.lookaround_fetch import DA3_ONLY_APPLE_ZOOM, download_lookaround
from services.pipeline_runner import save_pointcloud
from services.streetview_fetch import DA3_ONLY_ZOOM, run_async, download_pano_by_id
from build_street_graph.build_graph import TOP_PANOS_PER_DOT, _cap_bucket_for_date, build_corridor_graphs
from ui.map_selection.candidates import apple_tile_panos

# Where prepare_pathfind_from_cover_chunk downloads the whole-NTU metadata +
# date cover from (see tests/fetch_ntu_metadata.py,
# tests/inspect_global_date_cover.py, build_street_graph/
# global_dates.py -- these were produced ONCE, offline, not something a
# real chunk run recomputes). Same dataset repo the CLI checkpoint flow
# already uses (see tab.py's CLI_JOIN_DATASET_REPO).
GLOBAL_DATASET_REPO = "potato-bug/ntu-reconstruction"
_global_metadata = None
_global_cover = None

# Yaw step for DA3's view slicing. 30 (12 slices) is the tested middle
# ground between DA3's own default 20 (18 slices) and the too-coarse 45
# (8 slices, caused 2/4 winners to go from partial acceptance to fully
# rejected in an earlier scoring experiment).
DEFAULT_STEP_DEGREES = 30

# How many panos download at once. Downloads used to run one at a time
# (each its own fresh event loop) -- for a large batch (100+ candidates on
# a real branching selection) that alone can take long enough to let the
# ZeroGPU proxy token expire before the GPU call ever fires, since the
# token's lifetime is wall-clock, not "how many GPU calls made". Bounded
# rather than unlimited for the same reason download_panorama_image caps
# its own per-pano tile connections -- don't burst past what Google's rate
# limiter tolerates.
DOWNLOAD_CONCURRENCY = 10


async def _download_one(node, sem):
    """Download a node's equirectangular image at DA3-only res, return path (None on failure)."""
    async with sem:
        try:
            if node["source"] == "apple":
                # download_lookaround is a blocking call (unlike the Google
                # path) -- off the event loop so it doesn't stall the other
                # concurrent downloads while it runs.
                return await asyncio.to_thread(download_lookaround, node["_pano"], DA3_ONLY_APPLE_ZOOM)
            return await download_pano_by_id(node["id"], zoom=DA3_ONLY_ZOOM)
        except Exception as e:
            print(f"Download failed for {node['key']}: {e}")
            return None


async def _download_all(nodes):
    sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)
    return await asyncio.gather(*[_download_one(n, sem) for n in nodes])


def _download_date_graphs(date_graphs):
    """Download every node referenced by any of the given date graphs' dot
    buckets, in one combined batch (concurrently, DOWNLOAD_CONCURRENCY at a
    time -- node keys are unique across graphs, since each graph only ever
    holds its own date's own real panos). Returns (ready_graphs,
    node_entries): ready_graphs -- each date graph with dot_candidates
    values replaced by (key, path, lat, lon) tuples for whatever actually
    downloaded (a dot that loses every candidate to a failed download is
    dropped entirely -- the walk algorithm treats it exactly like a dot
    that was never populated, same skip-one handling either way);
    node_entries -- flat (key, path, lat, lon, date) list across ALL
    graphs, for join_segments' GPS lookup (see join_segments.join_segments);
    catalog -- {pano key: {dot, source, id, lat, lon, date, heading, pitch,
    roll}} for every candidate, whichever date it belongs to. It is what
    lets a reconstructed pano be matched back to the place it came from,
    since which date wins is not decided until the walk is over."""
    all_nodes = [n for g in date_graphs for bucket in g["dot_candidates"].values() for n in bucket]
    keys = [n["key"] for n in all_nodes]
    catalog = {n["key"]: {"dot": dot, "source": n["source"], "id": n["id"],
                          "lat": n["lat"], "lon": n["lon"], "date": n.get("date"),
                          "heading": n.get("heading"), "pitch": n.get("pitch"),
                          "roll": n.get("roll")}
               for g in date_graphs for dot, bucket in g["dot_candidates"].items()
               for n in bucket}
    paths = run_async(_download_all(all_nodes))
    path_by_key = {key: path for key, path in zip(keys, paths) if path}

    ready_graphs = []
    node_entries = []
    for g in date_graphs:
        dot_candidates = {}
        for dot_idx, bucket in g["dot_candidates"].items():
            entries = [(n["key"], path_by_key[n["key"]], n["lat"], n["lon"])
                       for n in bucket if n["key"] in path_by_key]
            if entries:
                dot_candidates[dot_idx] = entries
                node_entries.extend((key, path, lat, lon, g["date"]) for key, path, lat, lon in entries)
        if dot_candidates:
            ready_graphs.append({"date": g["date"], "dot_candidates": dot_candidates})

    return ready_graphs, node_entries, catalog


def prepare_pathfind(start, goals, corridor_edges, center) -> dict:
    """CPU/network only, no GPU -- gathers candidates along the corridor,
    splits them into isolated per-date graphs, and downloads every node
    any of them reference. Split out from the GPU step specifically so
    the GPU-triggering click (run_prepared_pathfind) can happen as its
    own fresh, minimal-latency user interaction right before the
    @spaces.GPU call, instead of that call being buried at the end of a
    long download inside one combined request -- the ZeroGPU proxy token's
    validity is wall-clock, and a long blocking step ahead of it is exactly
    what can let it go stale before schedule() is ever reached.

    start: (lat, lon) -- the fixed start node's real position.
    center: (lat, lon) -- the searched coordinate that defined this area.
    Carried through untouched; postprocess measures every position from it.
    goals: [(lat, lon), ...] -- every other selected node.
    corridor_edges: [((lat1, lon1), (lat2, lon2)), ...] -- the REAL,
    already-confirmed edges of the clicked selection graph (from Street
    View's own pano.links, see map_selection/candidates.py and
    map_selection/tab.py's handle_bridge_message) -- not inferred from
    click order or proximity, since these can branch or loop. Used only to
    shape *where* to sample candidate panos (fetch_corridor_nodes); the
    search is still free to use different nodes than exactly these.

    Returns a dict to pass straight to run_prepared_pathfind."""
    t0 = time.monotonic()
    if not goals:
        raise ValueError("Need at least one goal (a second selected node).")
    if not corridor_edges:
        raise ValueError("Need at least one confirmed edge tracing the route.")
    start_lat, start_lon = start

    date_graphs, points, adjacency, elevations = build_corridor_graphs(corridor_edges, start_lat, start_lon, goals)
    if not date_graphs:
        raise ValueError("No date reaches from the start toward any goal -- not enough connected candidates.")

    n_candidates = sum(len(bucket) for g in date_graphs for bucket in g["dot_candidates"].values())
    print(f"Downloading {n_candidates} candidate(s) across {len(date_graphs)} date graph(s): "
          f"{[g['date'] for g in date_graphs]}")
    for g in date_graphs:
        for dot, bucket in g["dot_candidates"].items():
            print(f"  [candidates] date={g['date']} dot={dot}: {[n['key'] for n in bucket]}")
    ready_graphs, node_entries, catalog = _download_date_graphs(date_graphs)
    if not ready_graphs:
        raise ValueError("Nothing downloaded successfully -- can't reconstruct.")

    print(f"prepare_pathfind: done in {time.monotonic() - t0:.1f}s")
    return {
        "date_graphs": ready_graphs,
        "node_entries": node_entries,
        "points": points,
        "adjacency": adjacency,
        "elevations": elevations,
        "catalog": catalog,
        "start": start,
        "center": center,
        "goals": goals,
        "top_dates": [g["date"] for g in ready_graphs],
    }






def run_prepared_pathfind(prep: dict, output_dir, step_degrees: int = DEFAULT_STEP_DEGREES):
    """Convenience one-shot: corridor search + join/bridging in ONE GPU
    session (see pipeline_runner.run_pathfind_and_join_gpu) -- avoids
    paying for two separate DA3 model loads when you just want the final
    result end-to-end and don't care about re-testing join/bridging
    separately. UI callers doing the 3-step Prepare/Run/Join flow (see
    tab.py) should call run_prepared_pathfind_segments,
    save_pathfind_segments, and save_joined_pathfind instead -- that
    split lets join/bridging be re-tested without re-running the much
    more expensive corridor search each time; this one-shot call always
    redoes both together.

    Returns (results, segments, bundle_path): results is [(label,
    ply_path), ...] -- one per segment (see
    reconstruct/walk_graph.py for what a "segment" is),
    plus one "joined" entry per still-separate piece (see
    join_segments.join_segments -- multiple pieces means bridging left
    some genuinely unconnected, not an error) when there's more than one
    segment to actually combine. segments/bundle_path are the same as
    save_segments_bundle produces, still saved here so join/bridging can
    be re-tuned later (via the separate Join button) without redoing
    this whole call."""
    from services.pipeline_runner import run_pathfind_and_join_gpu
    t0 = time.monotonic()
    start_lat, start_lon = prep["start"]
    segments, pieces = run_pathfind_and_join_gpu(
        prep["date_graphs"], prep["points"], prep["adjacency"], start_lat, start_lon,
        step_degrees=step_degrees,
    )
    if not segments:
        raise RuntimeError("No connected path found from start toward any goal.")

    results = []
    if pieces is None:
        # a lone segment has nothing to bridge TO, but it is still a piece,
        # and the scene is only filled in by saving one
        from reconstruct.join_segments import pieces_to_output
        pieces = pieces_to_output(segments)
    results.extend(_save_joined_pieces(pieces, output_dir, prep["catalog"]))
    print(f"run_prepared_pathfind: done in {time.monotonic() - t0:.1f}s")
    return results








def open_scene(prep, output_dir):
    """A scene holding every place this run will try to reconstruct.

    One node per dot that has a candidate, carrying the best-ranked pano
    we know of there -- so the nodes, their adjacency and the road lines
    all exist before the GPU runs. Reconstruction fills in the rest.
    """
    import scene as scene_mod
    best, elevations = {}, prep.get("elevations") or []
    for key, c in prep["catalog"].items():
        best.setdefault(c["dot"], (key, c))

    order = sorted(best)                       # dot index -> node index
    index = {dot: i for i, dot in enumerate(order)}
    nodes = []
    for dot in order:
        _, c = best[dot]
        nodes.append(scene_mod.Node(pano=scene_mod.Pano(
            source=c["source"], id=c["id"], lat=c["lat"], lon=c["lon"],
            date=c["date"], heading=c["heading"], pitch=c["pitch"], roll=c["roll"],
            elevation=elevations[dot] if dot < len(elevations) else None)))

    adjacency = {str(index[d]): sorted(index[n] for n in ns if n in index)
                 for d, ns in prep["adjacency"].items() if d in index}
    sc = scene_mod.Scene(center=list(prep["center"]), nodes=nodes, adjacency=adjacency)
    sc.save(output_dir)
    return sc


def _save_joined_pieces(pieces, output_dir, catalog) -> list[str]:
    """Fill the scene's nodes in with what the reconstruction produced.

    A node already exists for every place; this writes each one's points,
    the pano that actually filled it, and DA3's camera pose. One .ply per
    NODE: DA3 reconstructs one or two panoramas at a time and a node's
    points enter exactly once, so a panorama is the smallest thing ever
    independently produced.

    Edges are recorded by node index, which is what makes a piece a
    connected component rather than something stored.
    """
    import scene as scene_mod
    from reconstruct.join_segments import _piece_edges
    sc = scene_mod.Scene.load(output_dir)
    node_of_dot = {catalog[n.key]["dot"]: i
                   for i, n in enumerate(sc.nodes) if n.key in catalog}

    results = []
    for p_i, (clouds, metadata) in enumerate(pieces):
        placed = {}
        for key, m in metadata.items():
            c = catalog.get(key)
            if c is None or c["dot"] not in node_of_dot:
                continue
            i = node_of_dot[c["dot"]]
            node = sc.nodes[i]
            node.pano = scene_mod.Pano(
                source=c["source"], id=c["id"], lat=m["lat"], lon=m["lon"],
                date=m.get("date"), elevation=node.pano.elevation,
                heading=c["heading"], pitch=c["pitch"], roll=c["roll"],
                views_kept=m.get("n_views_kept"), views_total=m.get("n_views_total"))
            node.position = list(m["position"])
            node.rotation = m.get("rotation")
            node.ply = f"node_{i}.ply"
            save_pointcloud(*clouds[key], os.path.join(output_dir, node.ply))
            placed[key] = i

        for a, b, keep_a, keep_b in _piece_edges(metadata):
            if a in placed and b in placed:
                sc.edges.append(scene_mod.Edge(a=placed[a], b=placed[b],
                                               keep_a=keep_a, keep_b=keep_b))
        results.append(f"piece {p_i}: {len(placed)} node(s)")

    sc.save(output_dir)
    return results






