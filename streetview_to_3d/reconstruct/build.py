"""Orchestrator for the pathfind flow, called from ui/tab.py:

1. prepare_pathfind (no GPU): build_corridor_graphs gathers candidate
   panos along the clicked graph and splits them into isolated per-date
   graphs (see build_street_graph/), then every candidate is downloaded.
2. run_prepared_pathfind: ONE GPU call (pipeline_runner.
   run_pathfind_and_join_gpu) that walks the date graphs
   (reconstruct/walk_graph.py) and then bridges the pieces it left
   (reconstruct/join_segments.py). One call, not two: each @spaces.GPU call
   requests a fresh session credential, and a second one can arrive after
   the first has already expired.
3. The result is written into the run's scene (open_scene,
   _save_joined_pieces); placement happens afterwards, in postprocess/.
"""
import asyncio
import os
import time

import numpy as np

from streetview_to_3d.services.da3_ops import VIEW_STEP_DEGREES
from streetview_to_3d.services.lookaround_fetch import DA3_ONLY_APPLE_ZOOM, download_lookaround
from streetview_to_3d.services.pipeline_runner import save_pointcloud
from streetview_to_3d.services.streetview_fetch import DA3_ONLY_ZOOM, run_async, download_pano_by_id, fetch_depth
from streetview_to_3d.build_street_graph.build_graph import build_corridor_graphs

# How many panos download at once. Downloads used to run one at a time
# (each its own fresh event loop) -- for a large batch (100+ candidates on
# a real branching selection) that alone can take long enough to let the
# ZeroGPU proxy token expire before the GPU call ever fires, since the
# token's lifetime is wall-clock, not "how many GPU calls made". Bounded
# rather than unlimited for the same reason download_panorama_image caps
# its own per-pano tile connections -- don't burst past what Google's rate
# limiter tolerates.
DOWNLOAD_CONCURRENCY = 10


def _fill_ground(clouds, metadata, catalog):
    """clouds with each Google pano's floor hole filled from Google's own
    ground (see reconstruct.ground_fill). Never fails a run: a pano with no
    depth map, or any error, just keeps its cloud as it was."""
    from streetview_to_3d.reconstruct import ground_fill
    if not ground_fill.FILL_GROUND:
        return clouds
    try:
        panos = {}
        for key, m in metadata.items():
            c = catalog.get(key)
            if c is None or c["source"] != "google" or key not in clouds or m.get("rotation") is None:
                continue
            depth = run_async(fetch_depth(c["id"]))
            if depth is None:
                continue
            panos[key] = ground_fill.Pano(
                np.asarray(clouds[key][0]), np.asarray(m["position"], float),
                np.asarray(m["rotation"], float), depth,
                run_async(download_pano_by_id(c["id"], zoom=DA3_ONLY_ZOOM)))
        ready = ground_fill.prepare_piece(list(panos.values()))
        if not ready:
            return clouds
        existing = np.concatenate([np.asarray(p) for p, _ in clouds.values() if len(p)])
        filled = dict(clouds)
        for key, p in panos.items():
            if p not in ready:
                continue
            extra = ground_fill.fill(p, ready, existing)
            if extra is not None:
                pts, cols = clouds[key]
                filled[key] = (np.concatenate([pts, extra[0]]), np.concatenate([cols, extra[1]]))
        print(f"ground fill: {len(ready)} pano(s), +{sum(len(filled[k][0]) - len(clouds[k][0]) for k in filled)} "
              f"point(s), {ready[0].scale:.2f} m per DA3 unit", flush=True)
        return filled
    except Exception as e:
        print(f"ground fill skipped: {e}", flush=True)
        return clouds


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


def prepare_pathfind(start, goals, corridor_edges, center, link=True) -> dict:
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
    link: False prepares solo mode instead (reconstruct.solo): Google panos
    only, plus Google's own neighbours of them.
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

    prep = {
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
    if not link:
        from streetview_to_3d.reconstruct import solo
        solo.prepare(prep)
    print(f"prepare_pathfind: done in {time.monotonic() - t0:.1f}s")
    return prep


def run_prepared_pathfind(prep: dict, output_dir, step_degrees: int = VIEW_STEP_DEGREES,
                          conf_lower_percentile: float | None = None,
                          gpu_seconds: float | None = None, hfov=None, ring_pitches=None):
    """Walk and join in one GPU call, then write the pieces into the scene
    at output_dir. Returns one "piece i: n node(s)" line per piece. A prep
    made for solo mode (prepare_pathfind(link=False)) runs that instead.

    conf_lower_percentile: how much of each view's own weakest pixels DA3
    drops before backprojection -- see services.da3_ops.CONF_LOWER_PERCENTILE.
    None keeps that module's own default.

    gpu_seconds: the ZeroGPU window to ask for. None sizes it from the dot
    count -- see services.pipeline_runner.estimate_gpu_seconds.

    hfov, ring_pitches: solo mode's views (see reconstruct.solo); None keeps
    its defaults. The walk ignores them.
    """
    from streetview_to_3d.services.pipeline_runner import run_pathfind_and_join_gpu, run_solo_gpu
    t0 = time.monotonic()
    if "solo" in prep:
        pieces = run_solo_gpu(prep["solo"], prep["catalog"], conf_lower_percentile=conf_lower_percentile,
                              gpu_seconds=gpu_seconds, hfov=hfov, ring_pitches=ring_pitches)
        if not pieces:
            raise RuntimeError("DA3 reconstructed none of the panos.")
        results = _save_joined_pieces(pieces, output_dir, prep["catalog"])
        print(f"run_prepared_pathfind (solo): done in {time.monotonic() - t0:.1f}s")
        return results
    start_lat, start_lon = prep["start"]
    segments, pieces = run_pathfind_and_join_gpu(
        prep["date_graphs"], prep["points"], prep["adjacency"], start_lat, start_lon,
        step_degrees=step_degrees, conf_lower_percentile=conf_lower_percentile,
        gpu_seconds=gpu_seconds,
    )
    if not segments:
        raise RuntimeError("No connected path found from start toward any goal.")

    results = []
    if pieces is None:
        # a lone segment has nothing to bridge TO, but it is still a piece,
        # and the scene is only filled in by saving one
        from streetview_to_3d.reconstruct.join_segments import pieces_to_output
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
    from streetview_to_3d import scene as scene_mod
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
            elevation=elevations[dot] if dot < len(elevations) else c.get("elevation"))))

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

    Two pieces can both hold a panorama for the same place: patching walks
    overlapping stretches on purpose, and set_cover keeps a piece for any
    place it adds. A place is one node, so the bigger piece keeps it and the
    other's panorama there is dropped with its links. Writing both let the
    second overwrite the first while the first's links survived, gluing two
    DA3 frames into one piece.
    """
    from streetview_to_3d import scene as scene_mod
    from streetview_to_3d.reconstruct.join_segments import _piece_edges
    sc = scene_mod.Scene.load(output_dir)
    node_of_dot = {catalog[n.key]["dot"]: i
                   for i, n in enumerate(sc.nodes) if n.key in catalog}

    results, taken = [], set()
    for p_i in sorted(range(len(pieces)), key=lambda k: -len(pieces[k][1])):
        clouds, metadata = pieces[p_i]
        clouds = _fill_ground(clouds, metadata, catalog)
        placed = {}
        for key, m in metadata.items():
            c = catalog.get(key)
            if c is None or c["dot"] not in node_of_dot:
                continue
            i = node_of_dot[c["dot"]]
            if i in taken:
                print(f"save: piece {p_i}'s {key} is at node {i}, already held by a "
                      f"bigger piece -- dropped with its links", flush=True)
                continue
            taken.add(i)
            node = sc.nodes[i]
            node.pano = scene_mod.Pano(
                source=c["source"], id=c["id"], lat=m["lat"], lon=m["lon"],
                date=m.get("date"), elevation=node.pano.elevation,
                heading=c["heading"], pitch=c["pitch"], roll=c["roll"],
                views_kept=m.get("n_views_kept"), views_total=m.get("n_views_total"))
            node.position = list(m["position"])
            node.rotation = m.get("rotation")
            # A pano DA3 kept none of the views of has no points. Its
            # camera still places the piece, but an empty .ply would
            # stop the viewer opening the scene at all.
            if len(clouds[key][0]):
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


