"""Search build_graph's candidate graph start -> goals, reconstruct the path.

Client half of the pathfinding flow:
- build the candidate graph (build_graph, no GPU) from the real click-graph
  edges (map_selection/tab.py's handle_bridge_message) -- not inferred from
  click order or proximity, since the selection can branch or loop
- download the top-N-date candidates (network, cached; the GPU call can't
  fetch, so this has to happen before it -- see _local_batch)
- hand nodes + edges + goals to run_pathfind_reconstruction in ONE GPU
  call: the real multi-goal best-first search + DA3 tests + stitching +
  multi-segment retry all live there

One GPU call, not two: an earlier version fell back to a second call
(download everything, retry) if the first didn't reach the end. That's
exactly the pattern that causes 'Expired ZeroGPU proxy token' -- each
@spaces.GPU call requests a fresh session credential, and a second
request can arrive after the first one's already aged out. The top-N
date filter already keeps the single download bounded, so there's no real
need for a fallback call.
"""
import asyncio
import os

from services.geo import haversine_m
from services.lookaround_fetch import DA3_ONLY_APPLE_ZOOM, download_lookaround
from services.pipeline_runner import run_pathfind_reconstruction_gpu, save_pointcloud
from services.streetview_fetch import DA3_ONLY_ZOOM, run_async, download_pano_by_id
from street_builder.build_graph.build_graph import build_corridor_graph
from street_builder.build_graph.fetch_nodes import POINT_MAX_DIST_M
from street_builder.reconstruction.best4 import BEST4_STEP_DEGREES

# How many dates to keep, ranked by coverage span (does this date's
# coverage reach from near-start to near-end at all) -- a date confined to
# one stretch of the route can't connect it on its own.
DATE_TOP_N = 8

# Mirrors run_pathfind_reconstruction's own start_zone_m/goal_tolerance_m
# defaults -- used here only to pre-check whether a date's own same-date
# edges can even reach from a start-zone root to a goal-zone node, before
# spending a download (let alone a GPU test) on it.
START_ZONE_M = 5.0
GOAL_TOLERANCE_M = 15.0

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


def _covered_indices(candidates, points, max_dist_m):
    """Indices of sample points with >=1 candidate within max_dist_m."""
    covered = set()
    for i, (lat, lon) in enumerate(points):
        if any(haversine_m(lat, lon, c["lat"], c["lon"]) <= max_dist_m for c in candidates):
            covered.add(i)
    return covered


def _date_connects(date_nodes, edges, start_lat, start_lon, goals):
    """Whether this date's own same-date edges can reach from a start-zone
    root to at least one goal-zone at all -- a structural check (graph
    reachability), not a real DA3 test. Dates that fail this can't connect
    anything no matter what, so there's no point downloading or GPU-testing
    them. Doesn't need to reach EVERY goal to be worth trying -- the
    multi-goal search itself handles a date covering only some of them."""
    by_key = {n["key"]: n for n in date_nodes}
    roots = [k for k, n in by_key.items()
             if haversine_m(n["lat"], n["lon"], start_lat, start_lon) <= START_ZONE_M]
    if not roots:
        return False
    seen = set(roots)
    stack = list(roots)
    while stack:
        key = stack.pop()
        n = by_key[key]
        if any(haversine_m(n["lat"], n["lon"], g[0], g[1]) <= GOAL_TOLERANCE_M for g in goals):
            return True
        for other_key, _ in edges.get(key, []):
            if other_key in by_key and other_key not in seen:
                seen.add(other_key)
                stack.append(other_key)
    return False


def _rank_dates(by_date, points, max_dist_m, top_n):
    """Dates ranked by span (earliest to latest covered point -- does
    coverage reach start to end) then total coverage count as tiebreaker."""
    scored = []
    for date, candidates in by_date.items():
        covered = _covered_indices(candidates, points, max_dist_m)
        span = (max(covered) - min(covered)) if covered else 0
        scored.append((date, span, len(covered)))
    scored.sort(key=lambda t: (t[1], t[2]), reverse=True)
    return [date for date, _, _ in scored[:top_n]]


def _local_batch(nodes, edges, points, start_lat, start_lon, goals):
    """Keep every already-gathered candidate whose date both (a) structurally
    connects start-zone to at least one goal-zone via its own same-date
    edges, and (b) ranks in the top-N by coverage span among the dates that
    pass (a). This is the whole download batch (single GPU call, see module
    docstring for why there's no second-pass fallback).

    Returns (keys, top_dates) -- keys as a list (not a set) specifically so
    the caller can preserve top_dates' rank order downstream. A set's
    iteration order depends on Python's per-process string hash seed, which
    was silently discarding this ranking (and making the whole pathfind
    result non-reproducible run to run) even though top_dates itself is a
    real, deterministic ranking."""
    by_date = {}
    for n in nodes:
        by_date.setdefault(n["date"], []).append(n)

    connectable = {date: ns for date, ns in by_date.items()
                   if _date_connects(ns, edges, start_lat, start_lon, goals)}
    dropped = len(by_date) - len(connectable)
    if dropped:
        print(f"Dropped {dropped}/{len(by_date)} dates: no same-date edge path from start toward any goal.")

    top_dates = _rank_dates(connectable, points, POINT_MAX_DIST_M, DATE_TOP_N)
    print(f"Top dates by coverage span: {top_dates}")

    keys = [n["key"] for n in nodes if n["date"] in top_dates]
    return keys, top_dates


def _download_and_filter(keys, by_key, edges):
    """Download this batch (concurrently, DOWNLOAD_CONCURRENCY at a time),
    return (node_entries, edges restricted to what downloaded). keys must
    be a list (or otherwise order-preserving) -- node_entries' order is
    what determines the pathfind date-try order downstream, and dict.fromkeys
    (not a set) is what dedupes it while preserving that order."""
    unique_keys = list(dict.fromkeys(keys))
    nodes = [by_key[k] for k in unique_keys]
    paths = run_async(_download_all(nodes))

    entries = []
    for key, n, path in zip(unique_keys, nodes, paths):
        if path:
            entries.append((key, path, n["lat"], n["lon"], n["date"]))
    have = {e[0] for e in entries}
    filtered = {k: [(o, d) for o, d in v if o in have] for k, v in edges.items() if k in have}
    return entries, filtered


def prepare_pathfind(start, goals, corridor_edges) -> dict:
    """CPU/network only, no GPU -- gathers candidates along the corridor and
    downloads the top-date batch. Split out from the GPU step specifically
    so the GPU-triggering click (run_prepared_pathfind) can happen as its
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
    if not goals:
        raise ValueError("Need at least one goal (a second selected node).")
    if not corridor_edges:
        raise ValueError("Need at least one confirmed edge tracing the route.")
    start_lat, start_lon = start

    nodes, edges, points = build_corridor_graph(corridor_edges)
    nodes = [n for n in nodes if edges.get(n["key"])]  # drop isolated (never testable)
    if len(nodes) < 2:
        raise ValueError("Not enough connected candidates along this street.")
    by_key = {n["key"]: n for n in nodes}

    batch_keys, top_dates = _local_batch(nodes, edges, points, start_lat, start_lon, goals)
    print(f"Downloading {len(batch_keys)}/{len(nodes)} top-date candidates...")
    node_entries, batch_edges = _download_and_filter(batch_keys, by_key, edges)

    return {
        "node_entries": node_entries,
        "batch_edges": batch_edges,
        "start": start,
        "goals": goals,
        "top_dates": top_dates,
    }


def run_prepared_pathfind(prep: dict, output_dir,
                          step_degrees: int = BEST4_STEP_DEGREES) -> list[tuple[str, str]]:
    """Convenience one-shot: GPU search + save per-segment previews + join
    (if there's more than one segment) in a single call. UI callers doing
    the 3-step Prepare/Run/Join flow (see tab.py) should call
    run_prepared_pathfind_segments, save_pathfind_segments, and
    save_joined_pathfind separately instead -- join doesn't need the GPU at
    all, so splitting it out means re-testing/tuning it doesn't require
    re-running the expensive DA3 search each time.

    Returns [(label, ply_path), ...] -- one per segment (see
    run_pathfind_reconstruction for what a "segment" is), plus one more
    "joined" entry (see join_segments.join_segments) when there's more
    than one segment to actually combine."""
    segments = run_prepared_pathfind_segments(prep, step_degrees=step_degrees)
    results = save_pathfind_segments(segments, output_dir)
    if len(segments) > 1:
        try:
            results.append(save_joined_pathfind(prep, segments, output_dir))
        except Exception as e:
            print(f"join_segments failed: {e}")
            results.append((f"path (joined) -- failed: {e}", None))
    return results


def save_pathfind_segments(segments, output_dir) -> list[tuple[str, str]]:
    """Saves each segment's own point cloud as its own .ply, no GPU, no
    fitting/joining. Returns [(label, ply_path), ...] previews."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for i, (pts, cols, path_edges, date, reached, node_positions) in enumerate(segments):
        status = "reached all goals" if reached else "partial"
        label = f"path (date {date}, {len(path_edges)} hops, {status})"
        ply = save_pointcloud(pts, cols, os.path.join(output_dir, f"pathfind_{i}.ply"))
        results.append((label, ply))
    return results


def save_joined_pathfind(prep: dict, segments, output_dir) -> tuple[str, str]:
    """Fits + merges every segment (see join_segments.join_segments), saves
    the result, returns one (label, ply_path). No GPU -- pure linear
    algebra, safe to call repeatedly against the same already-computed
    segments while tuning the join step."""
    from street_builder.reconstruction.join_segments import join_segments
    os.makedirs(output_dir, exist_ok=True)
    pts, cols = join_segments(segments, prep["node_entries"])
    ply = save_pointcloud(pts, cols, os.path.join(output_dir, "pathfind_joined.ply"))
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


def run_prepared_pathfind_segments(prep: dict, step_degrees: int = BEST4_STEP_DEGREES):
    """Same GPU call as run_prepared_pathfind, but returns the raw segment
    list (pts, cols, path_edges, date, reached, node_positions per segment)
    instead of saved .ply paths -- what join_segments.py needs to fit and
    merge segments, rather than just preview them individually."""
    start_lat, start_lon = prep["start"]
    segments = run_pathfind_reconstruction_gpu(
        prep["node_entries"], prep["batch_edges"], start_lat, start_lon, prep["goals"],
        step_degrees=step_degrees, date_order=prep["top_dates"],
    )
    if not segments:
        raise RuntimeError("No connected path found from start toward any goal.")
    return segments
