"""Orchestrator for the pathfind flow -- wires the pipeline stages
together for the UI (tab.py calls into this):

1. build_graph.build_corridor_graphs: gather candidate panos along the
   real click-graph and split into up to DATE_TOP_N isolated, capped
   per-date graphs (no GPU) -- see build_street_graph/.
2. Download every node referenced by any of those graphs (network, cached).
3. run_pathfind_reconstruction_gpu: ONE GPU call -- the corridor-search
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
from services.pipeline_runner import run_pathfind_reconstruction_gpu, save_pointcloud
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


def _load_global_cover():
    """Downloads (once per process, cached module-level) the whole-NTU
    metadata + date cover produced offline by tests/fetch_ntu_metadata.py
    + tests/inspect_global_date_cover.py. Real network fetch only the
    first time this call lands on a given worker; every later chunk in
    the same worker reuses the cached copy."""
    global _global_metadata, _global_cover
    if _global_metadata is None:
        from huggingface_hub import hf_hub_download
        # Deliberately still fetch_metadata.json: this is the name in the
        # HuggingFace dataset, which a local rename does not change. It is
        # the same file local code calls downsampled_and_fetched_graph.json.
        meta_path = hf_hub_download(repo_id=GLOBAL_DATASET_REPO, repo_type="dataset", filename="global/fetch_metadata.json")
        cover_path = hf_hub_download(repo_id=GLOBAL_DATASET_REPO, repo_type="dataset", filename="global/date_cover.json")
        with open(meta_path) as f:
            _global_metadata = json.load(f)
        with open(cover_path) as f:
            _global_cover = {int(k): v for k, v in json.load(f).items()}
    return _global_metadata, _global_cover


def prepare_pathfind_from_cover_chunk(dots, date, top_per_dot=TOP_PANOS_PER_DOT) -> dict:
    """Same job as prepare_pathfind (gather candidates, download, return
    a dict ready for run_prepared_pathfind_segments) but for a chunk that
    was already cut FROM the pre-computed whole-NTU date cover (see
    global_dates.split_cover_into_chunks) instead of independently
    fetching/ranking its own dates off the raw selection graph. dots are
    global dot indices (into _load_global_cover's own points/adjacency/
    buckets) and date is the single date split_cover_into_chunks already
    assigned this whole chunk -- no per-dot date lookup or lat/lon
    matching needed here, dot index IS the identity.

    This is the fix for why cross-chunk bridging kept failing on a real,
    recurring pattern this session: two adjacent chunks ranking their OWN
    local dates independently could land on different "best" dates even
    when both had real data on a shared date that was merely their
    second- or third-best locally. Sourcing every chunk from the SAME
    global cover, cut along the cover's own region boundaries, removes
    that mismatch by construction -- a cross-date seam only ever happens
    at a chunk boundary now, never buried inside one chunk's own walk."""
    t0 = time.monotonic()
    if len(dots) < 2:
        raise ValueError("Need at least 2 dots (a start + a goal) in this chunk.")

    metadata, _ = _load_global_cover()
    global_points = metadata["points"]
    global_buckets = metadata["buckets"]
    global_adjacency = metadata["adjacency"]

    dot_set = set(dots)
    local_index = {d: i for i, d in enumerate(dots)}
    local_points = [tuple(global_points[d]) for d in dots]
    local_adjacency = {
        local_index[d]: [local_index[n] for n in global_adjacency.get(str(d), []) if n in dot_set]
        for d in dots
    }

    dot_candidates = {}
    for d in dots:
        lat, lon = global_points[d]
        capped = _cap_bucket_for_date(global_buckets.get(str(d), []), date, lat, lon, top_per_dot)
        if capped:
            dot_candidates[local_index[d]] = capped
    if not dot_candidates:
        raise ValueError(f"No dot in this chunk has a real candidate on date {date}.")

    # Apple candidates need their live _pano object to actually download
    # (see _download_one) -- the cached global metadata dropped it (not
    # JSON-serializable, and not needed for date ranking/covering). Cheap
    # per-chunk re-fetch, only for however many Apple candidates this
    # specific chunk's cover actually picked.
    for bucket in dot_candidates.values():
        for n in bucket:
            if n["source"] == "apple" and "_pano" not in n:
                try:
                    tile_panos = apple_tile_panos(n["lat"], n["lon"])
                    n["_pano"] = tile_panos[n["id"]]
                except Exception as e:
                    print(f"Apple re-fetch failed for {n['key']}: {e}")

    date_graphs = [{"date": date, "dot_candidates": dot_candidates}]
    n_candidates = sum(len(bucket) for bucket in dot_candidates.values())
    print(f"prepare_pathfind_from_cover_chunk: {n_candidates} candidate(s) across {len(dot_candidates)} dot(s), date={date}")
    ready_graphs, node_entries, catalog = _download_date_graphs(date_graphs)
    if not ready_graphs:
        raise ValueError("Nothing downloaded successfully -- can't reconstruct.")

    start = tuple(local_points[0])
    goals = [tuple(p) for p in local_points[1:]]
    center = tuple(metadata["points"][0])
    print(f"prepare_pathfind_from_cover_chunk: done in {time.monotonic() - t0:.1f}s")
    return {
        "date_graphs": ready_graphs,
        "node_entries": node_entries,
        "points": local_points,
        "adjacency": local_adjacency,
        "start": start,
        "center": center,
        "catalog": catalog,
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

    results = save_pathfind_segments(segments, output_dir)
    bundle_path = save_segments_bundle(segments, output_dir)
    if pieces is None:
        # a lone segment has nothing to bridge TO, but it is still a piece,
        # and the scene is only filled in by saving one
        from reconstruct.join_segments import pieces_to_output
        pieces = pieces_to_output(segments)
    results.extend(_save_joined_pieces(pieces, output_dir, prep["catalog"]))
    print(f"run_prepared_pathfind: done in {time.monotonic() - t0:.1f}s")
    return results, segments, bundle_path


def _stack(clouds):
    """One (points, colors) array pair from a piece's per-node clouds."""
    import numpy as np
    pts = [c[0] for c in clouds.values()]
    cols = [c[1] for c in clouds.values()]
    return (np.concatenate(pts) if pts else np.zeros((0, 3)),
            np.concatenate(cols) if cols else np.zeros((0, 3)))


def save_pathfind_segments(segments, output_dir) -> list[tuple[str, str]]:
    """Saves each segment as one preview .ply. No GPU, no fitting/joining.
    A preview only -- the real output keeps every node's points apart, see
    _save_joined_pieces. Returns [(label, ply_path), ...]."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for i, (clouds, path_edges, date, reached, node_positions, frame_poses) in enumerate(segments):
        status = "full corridor covered" if reached else "partial"
        label = f"path (date {date}, {len(path_edges)} hops, {status})"
        pts, cols = _stack(clouds)
        ply = save_pointcloud(pts, cols, os.path.join(output_dir, f"pathfind_{i}.ply"))
        results.append((label, ply))
    return results


def save_joined_pathfind(segments, output_dir, catalog, chunk_ids=None, known_adjacent_chunk_pairs=None) -> list[tuple[str, str]]:
    """Bridges (real DA3 tests between segment boundaries) via join_segments.py
    and records each still-separate piece in the scene (see
    _save_joined_pieces). Usually one piece; more means bridging genuinely
    couldn't connect everything -- see join_segments.join_segments. Its
    own GPU call (join_segments_gpu), separate from the corridor
    search's -- safe to call repeatedly against the same already-computed
    segments while tuning the join/bridging step, without re-running the
    corridor search.

    chunk_ids/known_adjacent_chunk_pairs: passed straight through to
    join_segments_gpu -- see its own docstring. For a large-scale multi-
    chunk reconstruction (many segments, one per chunk), pass these so
    bridging only attempts pairs known to be structurally adjacent
    instead of a blind O(n^2) scan over every segment."""
    from services.pipeline_runner import join_segments_gpu
    t0 = time.monotonic()
    os.makedirs(output_dir, exist_ok=True)
    pieces = join_segments_gpu(segments, chunk_ids=chunk_ids, known_adjacent_chunk_pairs=known_adjacent_chunk_pairs)
    results = _save_joined_pieces(pieces, output_dir, catalog)
    print(f"save_joined_pathfind: done in {time.monotonic() - t0:.1f}s")
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


def _save_joined_pieces(pieces, output_dir, catalog) -> list[tuple[str, str]]:
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
        results.append((f"piece {p_i} ({len(placed)} node(s))", None))

    sc.save(output_dir)
    return results


def save_segments_bundle(segments, output_dir) -> str:
    """Serializes only what Join actually needs (segments -- frame_poses
    already carries each node's own lat/lon, see join_segments.py) to one
    file, so Join can be re-run later -- a different session, or after
    tweaking join_segments.py -- without re-running Prepare or the
    expensive GPU search. Deliberately drops the rest of prep
    (date_graphs especially -- the full downloaded candidate pool across
    every date considered, unused by Join and by far the biggest part of
    prep) since it's dead weight for this file's one purpose; Prepare/Run
    themselves are cheap enough to redo from scratch if ever needed, so
    there's no reason to pay to store or re-download it. Plain pickle:
    numpy arrays, tuples, dicts all round-trip natively, and this file is
    only ever produced and consumed by this same codebase, not a public
    interchange format."""
    import pickle
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "pathfind_segments.pkl")
    with open(path, "wb") as f:
        pickle.dump({"segments": segments}, f)
    return path


def load_segments_bundle(path: str) -> list:
    """Inverse of save_segments_bundle. Returns segments, ready to feed
    straight into save_joined_pathfind (or save_pathfind_segments)."""
    import pickle
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["segments"]


def run_prepared_pathfind_segments(prep: dict, step_degrees: int = DEFAULT_STEP_DEGREES, protected_positions=None):
    """Same GPU call as run_prepared_pathfind, but returns the raw segment
    list (pts, cols, path_edges, date, reached, node_positions per segment)
    instead of saved .ply paths -- what join_segments.py needs to fit and
    merge segments, rather than just preview them individually.

    protected_positions: passed straight through to run_pathfind_reconstruction_gpu
    -- see walk_graph.run_pathfind_reconstruction's own docstring. For a
    chunked large-area reconstruction, pass the chunk's own real boundary
    node COORDINATES (known from the chunking step) so a location needed
    for cross-chunk bridging later doesn't get dropped as redundant
    coverage within this chunk alone."""
    t0 = time.monotonic()
    start_lat, start_lon = prep["start"]
    segments = run_pathfind_reconstruction_gpu(
        prep["date_graphs"], prep["points"], prep["adjacency"], start_lat, start_lon, step_degrees=step_degrees,
        protected_positions=protected_positions,
    )
    print(f"run_prepared_pathfind_segments: done in {time.monotonic() - t0:.1f}s")
    if not segments:
        raise RuntimeError("No connected path found from start toward any goal.")
    return segments
