"""Fetch NTU chunks off HuggingFace as a directory road alignment can read.

NTU's reconstruction is ~900 MB and lives in a dataset repo, so it is not
on disk. This pulls down a chosen subset and writes it in the layout
`postprocess.gps_fit.load_pieces` expects.

    # pieces picked in build/gap_selector.html, pasted straight in
    python -m tools.fetch_ntu_pieces --pieces 12,13,14,GAP_9 --out /tmp/ntu_bit

    # what is there, and which road each chunk sits on (metadata only, fast)
    python -m tools.fetch_ntu_pieces --list

    # everything along two roads, into a directory ready to align
    python -m tools.fetch_ntu_pieces --roads 58,68 --out /tmp/ntu_road58

    python -m postprocess.road_align.run --dir /tmp/ntu_road58 \
        --out ~/Downloads/road58.ply

NTU-specific on purpose: it knows this one dataset repo and this one graph.
The alignment pipeline itself stays unaware of where pieces came from.

A chunk is NOT a piece. `discover_pieces` splits a chunk wherever its own
nodes stop fitting their own GPS, so one chunk can hold several pieces --
chunk22 holds three, and only 2 of its 19 nodes belong to the one that was
asked for. Fetching the chunk whole and treating it as a piece forces all
three to share a single rigid transform, which is exactly what the split
existed to prevent: it sprawled 20 m off the road and reported a 22.9 m
GPS residual.

So --pieces writes one output piece per (selected piece, chunk) pair,
carrying only that piece's nodes.

The cloud cannot be split that precisely: a chunk's .ply is one merged
reconstruction with no per-point labels, so which node a point came from
is not recorded anywhere. Points are assigned to their NEAREST CAMERA
instead, in the chunk's own DA3 frame, which is where the cameras and the
points already share coordinates. That is an approximation of a boundary
that was never stored -- but the pieces it separates are ones that do not
belong in a common frame anyway, so a few metres of misattribution at the
seam costs far less than keeping them welded together.
"""
import argparse
import gzip
import json
import os
import shutil

import numpy as np
from scipy.spatial import cKDTree

from postprocess.corridors import _metres, roads
from postprocess.gps_fit.fit import real_en, use_origin
from postprocess.road_align.road_frames import smooth
from paths import DATA_DIR, FETCHED_GRAPH
import scene as scene_mod

REPO = "potato-bug/ntu-reconstruction"
RAW_PREFIX = "cli_raw"
NEAR_M = 12.0            # a camera this close to a road line is on that road


def _api():
    from huggingface_hub import HfApi
    return HfApi()


def chunk_files(api=None):
    """{chunk id: (ply path in repo, metadata path in repo)}."""
    api = api or _api()
    files = api.list_repo_files(repo_id=REPO, repo_type="dataset")
    out = {}
    for f in files:
        if not f.startswith(RAW_PREFIX + "/") or not f.endswith(".ply.gz"):
            continue
        d, base = os.path.split(f)
        if not base.startswith("pathfind_joined"):
            continue
        suffix = base[len("pathfind_joined"):-len(".ply.gz")]
        meta = os.path.join(d, f"pathfind_metadata{suffix}.json")
        if meta in files:
            out[d.split("/", 1)[1] + suffix] = (f, meta)
    return out


def chunks_for_pieces(piece_ids):
    """Which chunks hold the nodes of these selector pieces.

    The selector page deals in pieces, which are groupings of NODES; the
    repo stores chunks. data/selector_nodes.json records both for every
    node, so it is the map between them.

    A piece can need more than one chunk, and a chunk can serve more than
    one piece -- so this returns the union, and the fetched set may cover
    more ground than the pieces asked for.
    """
    with open(os.path.join(DATA_DIR, "selector_nodes.json")) as f:
        nodes = json.load(f)
    want = {str(p) for p in piece_ids}
    known = {str(n["group"]) for n in nodes}
    missing = want - known
    if missing:
        raise SystemExit(f"no such piece(s): {', '.join(sorted(missing))}")
    return {n["chunk_id"] for n in nodes if str(n["group"]) in want}


def _download(rel):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=REPO, repo_type="dataset", filename=rel)


def _dequantize(path):
    """Inverse of the upload's quantisation, in place; no-op if not quantised.

    Kept here rather than imported from ui.tab, which pulls in
    Gradio and the whole reconstruction UI just to reach one function.
    """
    with open(path, "rb") as f:
        data = f.read()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii")
    lines = header.splitlines()
    if not any(l.startswith("property short x") for l in lines):
        return
    scale = float(next(l for l in lines
                       if l.startswith("comment quant_scale_m")).split()[-1])
    n = int(next(l for l in lines if l.startswith("element vertex")).split()[-1])
    colour = "red" in header
    fields = [("x", "<i2"), ("y", "<i2"), ("z", "<i2")]
    if colour:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    verts = np.frombuffer(data[end:], dtype=np.dtype(fields), count=n)
    pts = np.stack([verts["x"], verts["y"], verts["z"]], 1).astype(np.float32) * scale

    out = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if colour:
        out += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    s = np.empty(n, dtype=np.dtype(out))
    s["x"], s["y"], s["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    if colour:
        for c in ("red", "green", "blue"):
            s[c] = verts[c]
    new_header = header.replace("property short x", "property float x") \
                       .replace("property short y", "property float y") \
                       .replace("property short z", "property float z")
    new_header = "\n".join(l for l in new_header.splitlines()
                           if not l.startswith("comment quant_scale_m")) + "\n"
    with open(path, "wb") as f:
        f.write(new_header.encode("ascii"))
        f.write(s.tobytes())


def ntu_center():
    """NTU's first dot. Every NTU transform ever solved is measured from it,
    so it is the area centre for anything cut out of this dataset."""
    return tuple(json.load(open(FETCHED_GRAPH))["points"][0])


def index(chunks=None, api=None):
    """{chunk id: (metadata dict, [road ids it lies on])}.

    Metadata only -- kilobytes per chunk against tens of megabytes for a
    cloud, so the whole campus can be indexed before deciding what to pull.
    """
    use_origin(*ntu_center())
    graph = scene_mod.Graph.read(FETCHED_GRAPH)
    xy = np.array(_metres(graph.points))
    lines = {i: smooth(xy[w]) for i, w in enumerate(roads(graph))}
    trees = {i: cKDTree(c) for i, c in lines.items() if len(c) >= 4}

    api = api or _api()
    out = {}
    for cid, (_, meta_rel) in sorted((chunks or chunk_files(api)).items()):
        meta = json.load(open(_download(meta_rel)))
        cams = np.array([real_en(m["lat"], m["lon"]) for m in meta.values()])
        on = sorted(r for r, t in trees.items() if (t.query(cams)[0] <= NEAR_M).any())
        out[cid] = (meta, on)
    return out


def _split(pts, cols, meta, groups):
    """[(piece id, meta subset, point mask), ...] for one chunk.

    `groups` maps node key -> the piece it belongs to. Points follow their
    nearest camera, in the chunk's own DA3 frame.
    """
    keys = list(meta)
    if not any(k in groups for k in keys):
        return []
    # the tree must hold EVERY camera in the chunk, not just the wanted
    # ones -- otherwise a point beside a discarded node still finds a
    # wanted node as its nearest and the whole cloud comes through
    cam = np.array([meta[k]["position"] for k in keys])
    nearest = cKDTree(cam).query(pts)[1]
    out = []
    for pid in sorted({groups[k] for k in keys if k in groups}):
        idx = [j for j, k in enumerate(keys) if groups.get(k) == pid]
        mask = np.isin(nearest, idx)
        out.append((pid, {keys[j]: meta[keys[j]] for j in idx}, mask))
    return out


def fetch(chunk_ids, out_dir, api=None, groups=None):
    """Write chosen chunks into `out_dir` as a scene.

    This is the adapter from the Hub's own layout (a .ply.gz plus a
    per-node metadata JSON per chunk) into a scene postprocess can read.
    """
    from postprocess.ply_io import write_ply
    from reconstruct.join_segments import _read_ply_points

    api = api or _api()
    files = chunk_files(api)
    os.makedirs(out_dir, exist_ok=True)
    ntu_graph = json.load(open(FETCHED_GRAPH))
    sc = scene_mod.Scene(
        center=list(ntu_center()),
        graph=scene_mod.Graph(points=ntu_graph["points"],
                              adjacency=ntu_graph["adjacency"]))
    written, total, n = {}, 0, 0
    for cid in sorted(chunk_ids):
        if cid not in files:
            print(f"  {cid}: not in the repo, skipped")
            continue
        ply_rel, meta_rel = files[cid]
        raw = os.path.join(out_dir, f".{cid}.ply")
        with gzip.open(_download(ply_rel), "rb") as f_in, open(raw, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        _dequantize(raw)
        meta = json.load(open(_download(meta_rel)))

        if groups is None:                       # whole chunk, unsplit
            name = f"piece_{n}.ply"
            os.replace(raw, os.path.join(out_dir, name))
            nodes, edges = scene_mod.from_metadata(meta)
            sc.pieces.append(scene_mod.Piece(ply=name, nodes=nodes, edges=edges))
            written[n] = cid
            total += os.path.getsize(os.path.join(out_dir, name))
            print(f"  piece_{n} <- {cid}  ({len(meta)} node(s))")
            n += 1
            continue

        pts, cols = _read_ply_points(raw)
        parts = _split(pts, cols, meta, groups)
        for pid, sub, mask in parts:
            name = f"piece_{n}.ply"
            out = os.path.join(out_dir, name)
            write_ply(out, pts[mask], cols[mask])
            nodes, edges = scene_mod.from_metadata(sub)
            sc.pieces.append(scene_mod.Piece(ply=name, nodes=nodes, edges=edges))
            written[n] = f"{cid}:{pid}"
            total += os.path.getsize(out)
            print(f"  piece_{n} <- {cid} piece {pid}  ({len(sub)} of {len(meta)} "
                  f"node(s), {int(mask.sum()):,} of {len(pts):,} points, "
                  f"{os.path.getsize(out) / 1e6:.0f} MB)")
            n += 1
        os.remove(raw)
    sc.save(out_dir)
    print(f"  {total / 1e6:.0f} MB written")
    with open(os.path.join(out_dir, "chunk_ids.json"), "w") as f:
        json.dump(written, f, indent=2)      # which chunk each piece came from
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true",
                    help="index every chunk and print which roads it is on")
    ap.add_argument("--roads", help="comma-separated road ids to fetch")
    ap.add_argument("--chunks", help="comma-separated chunk ids to fetch")
    ap.add_argument("--pieces",
                    help="comma-separated selector piece ids, as copied out of "
                         "build/gap_selector.html")
    ap.add_argument("--out", default="/tmp/ntu_pieces")
    args = ap.parse_args()

    api = _api()
    groups = None            # None = take each chunk whole, no splitting
    if args.list or args.roads:
        idx = index(api=api)
        by_road = {}
        for cid, (meta, on) in idx.items():
            for r in on:
                by_road.setdefault(r, []).append(cid)
        if args.list:
            print(f"{len(idx)} chunk(s) on {len(by_road)} road(s)\n")
            for r in sorted(by_road, key=lambda r: -len(by_road[r])):
                print(f"  road{r:<5} {len(by_road[r]):>3} chunk(s): "
                      + ", ".join(sorted(by_road[r])[:8])
                      + (" ..." if len(by_road[r]) > 8 else ""))
            return
        want = {c for r in (int(x) for x in args.roads.split(","))
                for c in by_road.get(r, [])}
    elif args.chunks:
        want = set(args.chunks.split(","))
    elif args.pieces:
        ids = [x.strip() for x in args.pieces.split(",") if x.strip()]
        want = chunks_for_pieces(ids)
        with open(os.path.join(DATA_DIR, "selector_nodes.json")) as f:
            groups = {n["key"]: str(n["group"]) for n in json.load(f)
                      if str(n["group"]) in set(ids)}
        print(f"{len(ids)} piece(s) -> {len(want)} chunk(s), "
              f"{len(groups)} node(s) kept")
    else:
        raise SystemExit("give --list, --roads, --chunks or --pieces")

    print(f"fetching {len(want)} chunk(s) into {args.out}")
    fetch(want, args.out, api=api, groups=groups)
    print(f"\nnow:  python -m postprocess.road_align.run --dir {args.out}")


if __name__ == "__main__":
    main()
