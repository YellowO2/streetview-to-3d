"""The reconstruction's orchestrator, called from ui/tab.py: prepare_pathfind gathers and
downloads candidates (no GPU), run_prepared_pathfind walks them in one GPU call and writes
the pieces into the run's scene. Placement happens afterwards, in postprocess/.
"""
import asyncio
import os
import time

from streetview_to_3d.reconstruct.runner import WalkSettings, run_walk_gpu, save_pointcloud
from streetview_to_3d.common.streetview_fetch import DA3_ONLY_ZOOM, download_pano_by_id, fetch_da3_pano, run_async
from streetview_to_3d.panos.build_graph import build_corridor_graphs

# Panos downloaded at once: fast enough that the ZeroGPU token (wall-clock) doesn't expire
# before the GPU call, bounded so Google's rate limiter isn't tripped.
DOWNLOAD_CONCURRENCY = 10


async def _download_one(node, sem):
    """A node's pano at DA3's resolution: its path, or None on failure."""
    async with sem:
        try:
            return await download_pano_by_id(node["id"], zoom=DA3_ONLY_ZOOM)
        except Exception as e:
            print(f"Download failed for {node['key']}: {e}")
            return None


async def _download_all(nodes):
    sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)
    return await asyncio.gather(*[_download_one(n, sem) for n in nodes])


def _download_date_graphs(date_graphs):
    """Download every candidate of every date graph in one batch.

    Returns (ready_graphs, catalog): ready_graphs has each dot's candidates as
    (key, path, lat, lon), dropping what failed to download (a dot left empty goes);
    catalog is {pano key: {dot, source, id, lat, lon, date, heading, pitch, roll}}, which
    matches a reconstructed pano back to its place once the walk picks a date.
    """
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
    for g in date_graphs:
        dot_candidates = {}
        for dot_idx, bucket in g["dot_candidates"].items():
            entries = [(n["key"], path_by_key[n["key"]], n["lat"], n["lon"])
                       for n in bucket if n["key"] in path_by_key]
            if entries:
                dot_candidates[dot_idx] = entries
        if dot_candidates:
            ready_graphs.append({"date": g["date"], "dot_candidates": dot_candidates})

    return ready_graphs, catalog


def prepare_pathfind(start, goals, corridor_edges, center) -> dict:
    """Gather the corridor's candidates into per-date graphs and download them (no GPU).

    Kept apart from the GPU step so the GPU click is a fresh interaction: the ZeroGPU
    token expires on wall-clock time. start/center: (lat, lon); goals: [(lat, lon)], the
    other selected nodes; corridor_edges: the selection's real Street View links,
    [((lat, lon, id), (lat, lon, id))]. Returns the dict run_prepared_pathfind takes.
    """
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
    ready_graphs, catalog = _download_date_graphs(date_graphs)
    if not ready_graphs:
        raise ValueError("Nothing downloaded successfully -- can't reconstruct.")

    prep = {
        "date_graphs": ready_graphs,
        "points": points,
        "adjacency": adjacency,
        "elevations": elevations,
        "catalog": catalog,
        "start": start,
        "center": center,
    }
    print(f"prepare_pathfind: done in {time.monotonic() - t0:.1f}s")
    return prep


def run_prepared_pathfind(prep: dict, output_dir, settings=WalkSettings()):
    """Walk in one GPU call, then write the pieces into the scene at output_dir.
    Returns one "piece i: n node(s)" line per piece."""
    t0 = time.monotonic()
    start_lat, start_lon = prep["start"]
    pieces = run_walk_gpu(prep["date_graphs"], prep["points"], prep["adjacency"], start_lat, start_lon, settings)
    if not pieces:
        raise RuntimeError("No connected path found from start toward any goal.")
    results = _save_joined_pieces(pieces, output_dir, prep["catalog"])
    print(f"run_prepared_pathfind: done in {time.monotonic() - t0:.1f}s")
    return results


def open_scene(prep, output_dir):
    """A scene with one node per dot that has a candidate (its best-ranked pano), saved
    before the GPU runs; reconstruction fills in the rest."""
    from streetview_to_3d.common import scene as scene_mod
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


def _keep_labels(pano, output_dir):
    """Copy the pano's class map (segment.pano_labels) into the scene, if the run made one."""
    import glob
    import shutil
    from streetview_to_3d.models.segment import labels_path
    if pano.source != "google":
        return
    path = fetch_da3_pano(pano.id)
    made = sorted(glob.glob(f"{glob.escape(path)}.*.labels.png"), key=os.path.getmtime) if path else []
    if made:
        os.makedirs(os.path.dirname(labels_path(output_dir, pano.id)), exist_ok=True)
        shutil.copyfile(made[-1], labels_path(output_dir, pano.id))


def _save_joined_pieces(pieces, output_dir, catalog) -> list[str]:
    """Write each node's points (one .ply per node), the pano that filled it, DA3's pose,
    and the edges by node index.

    Patches overlap, so two pieces can hold the same place: the bigger keeps it and the
    other's pano there is dropped with its links (keeping both glued two frames together).
    """
    from streetview_to_3d.common import scene as scene_mod
    from streetview_to_3d.reconstruct.pieces import _piece_edges
    sc = scene_mod.Scene.load(output_dir)
    node_of_dot = {catalog[n.key]["dot"]: i
                   for i, n in enumerate(sc.nodes) if n.key in catalog}

    results, taken = [], set()
    for p_i in sorted(range(len(pieces)), key=lambda k: -len(pieces[k][1])):
        clouds, metadata = pieces[p_i]
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
            # no points: the camera still places the piece, but an empty .ply breaks the viewer
            pts, cols = clouds[key]
            if len(pts):
                node.ply = f"node_{i}.ply"
                save_pointcloud(pts, cols, os.path.join(output_dir, node.ply))
            _keep_labels(node.pano, output_dir)
            placed[key] = i

        for a, b, keep_a, keep_b in _piece_edges(metadata):
            if a in placed and b in placed:
                sc.edges.append(scene_mod.Edge(a=placed[a], b=placed[b],
                                               keep_a=keep_a, keep_b=keep_b))
        results.append(f"piece {p_i}: {len(placed)} node(s)")

    sc.save(output_dir)
    return results
