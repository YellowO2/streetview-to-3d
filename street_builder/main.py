"""Orchestrator for the pathfind flow -- wires the pipeline stages
together for the UI (map_selection/tab.py calls into this):

1. build_graph.build_corridor_graphs: gather candidate panos along the
   real click-graph and split into up to DATE_TOP_N isolated, capped
   per-date graphs (no GPU) -- see street_builder/build_graph/.
2. Download every node referenced by any of those graphs (network, cached).
3. run_pathfind_reconstruction_gpu: ONE GPU call -- the corridor-search
   algorithm (street_builder/reconstruction/walk_graph.py) runs entirely
   inside it, producing possibly-several disconnected segments.
4. join_segments_gpu: a SEPARATE GPU call -- bridges segments together
   with real DA3 tests where possible, then GPS-fits + merges whatever's
   still separate into one final point cloud (see
   street_builder/reconstruction/join_segments.py). Split from step 3
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
import os
import time

from services.lookaround_fetch import DA3_ONLY_APPLE_ZOOM, download_lookaround
from services.pipeline_runner import run_pathfind_reconstruction_gpu, save_pointcloud
from services.streetview_fetch import DA3_ONLY_ZOOM, run_async, download_pano_by_id
from street_builder.build_graph.build_graph import build_corridor_graphs

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
    graphs, for join_segments' GPS lookup (see join_segments.join_segments)."""
    all_nodes = [n for g in date_graphs for bucket in g["dot_candidates"].values() for n in bucket]
    keys = [n["key"] for n in all_nodes]
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

    return ready_graphs, node_entries


def prepare_pathfind(start, goals, corridor_edges) -> dict:
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

    date_graphs, points, adjacency = build_corridor_graphs(corridor_edges, start_lat, start_lon, goals)
    if not date_graphs:
        raise ValueError("No date reaches from the start toward any goal -- not enough connected candidates.")

    n_candidates = sum(len(bucket) for g in date_graphs for bucket in g["dot_candidates"].values())
    print(f"Downloading {n_candidates} candidate(s) across {len(date_graphs)} date graph(s): "
          f"{[g['date'] for g in date_graphs]}")
    ready_graphs, node_entries = _download_date_graphs(date_graphs)
    if not ready_graphs:
        raise ValueError("Nothing downloaded successfully -- can't reconstruct.")

    print(f"prepare_pathfind: done in {time.monotonic() - t0:.1f}s")
    return {
        "date_graphs": ready_graphs,
        "node_entries": node_entries,
        "points": points,
        "adjacency": adjacency,
        "start": start,
        "goals": goals,
        "top_dates": [g["date"] for g in ready_graphs],
    }


def run_prepared_pathfind(prep: dict, output_dir,
                          step_degrees: int = DEFAULT_STEP_DEGREES) -> list[tuple[str, str]]:
    """Convenience one-shot: GPU search + save per-segment previews + join
    (if there's more than one segment) in a single call. UI callers doing
    the 3-step Prepare/Run/Join flow (see tab.py) should call
    run_prepared_pathfind_segments, save_pathfind_segments, and
    save_joined_pathfind separately instead -- join is its own separate
    GPU call (join_segments_gpu), so splitting it out means re-testing/
    tuning it doesn't require re-running the much more expensive corridor
    search each time.

    Returns [(label, ply_path), ...] -- one per segment (see
    street_builder/reconstruction/walk_graph.py for what a "segment" is),
    plus one more "joined" entry (see join_segments.join_segments) when
    there's more than one segment to actually combine."""
    t0 = time.monotonic()
    segments = run_prepared_pathfind_segments(prep, step_degrees=step_degrees)
    results = save_pathfind_segments(segments, output_dir)
    if len(segments) > 1:
        try:
            results.append(save_joined_pathfind(prep, segments, output_dir))
        except Exception as e:
            print(f"join_segments failed: {e}")
            results.append((f"path (joined) -- failed: {e}", None))
    print(f"run_prepared_pathfind: done in {time.monotonic() - t0:.1f}s")
    return results


def save_pathfind_segments(segments, output_dir) -> list[tuple[str, str]]:
    """Saves each segment's own point cloud as its own .ply, no GPU, no
    fitting/joining. Returns [(label, ply_path), ...] previews."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for i, (pts, cols, path_edges, date, reached, node_positions, frame_poses) in enumerate(segments):
        status = "full corridor covered" if reached else "partial"
        label = f"path (date {date}, {len(path_edges)} hops, {status})"
        ply = save_pointcloud(pts, cols, os.path.join(output_dir, f"pathfind_{i}.ply"))
        results.append((label, ply))
    return results


def save_joined_pathfind(prep: dict, segments, output_dir) -> tuple[str, str]:
    """Bridges (real DA3 tests between segment boundaries) + fits/merges
    whatever's left (see join_segments.py), saves the result, returns
    one (label, ply_path). Its own GPU call (join_segments_gpu),
    separate from the corridor search's -- safe to call repeatedly
    against the same already-computed segments while tuning the
    join/bridging step, without re-running the corridor search."""
    from services.pipeline_runner import join_segments_gpu
    t0 = time.monotonic()
    os.makedirs(output_dir, exist_ok=True)
    pts, cols = join_segments_gpu(segments, prep["node_entries"])
    ply = save_pointcloud(pts, cols, os.path.join(output_dir, "pathfind_joined.ply"))
    print(f"save_joined_pathfind: done in {time.monotonic() - t0:.1f}s")
    return f"path (joined, {len(segments)} segments)", ply


def save_segments_bundle(prep: dict, segments, output_dir) -> str:
    """Serializes prep + segments (everything Join needs) to one file, so
    Join can be re-run later -- a different session, or after tweaking
    join_segments.py -- without re-running Prepare or the expensive GPU
    search. Plain pickle: numpy arrays, tuples, dicts all round-trip
    natively, and this file is only ever produced and consumed by this
    same codebase, not a public interchange format."""
    import pickle
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "pathfind_segments.pkl")
    with open(path, "wb") as f:
        pickle.dump({"prep": prep, "segments": segments}, f)
    return path


def load_segments_bundle(path: str) -> tuple[dict, list]:
    """Inverse of save_segments_bundle. Returns (prep, segments), ready to
    feed straight into save_joined_pathfind (or save_pathfind_segments)."""
    import pickle
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["prep"], bundle["segments"]


def run_prepared_pathfind_segments(prep: dict, step_degrees: int = DEFAULT_STEP_DEGREES):
    """Same GPU call as run_prepared_pathfind, but returns the raw segment
    list (pts, cols, path_edges, date, reached, node_positions per segment)
    instead of saved .ply paths -- what join_segments.py needs to fit and
    merge segments, rather than just preview them individually."""
    t0 = time.monotonic()
    start_lat, start_lon = prep["start"]
    segments = run_pathfind_reconstruction_gpu(
        prep["date_graphs"], prep["points"], prep["adjacency"], start_lat, start_lon, step_degrees=step_degrees,
    )
    print(f"run_prepared_pathfind_segments: done in {time.monotonic() - t0:.1f}s")
    if not segments:
        raise RuntimeError("No connected path found from start toward any goal.")
    return segments
