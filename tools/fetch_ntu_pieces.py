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

A chunk needs no splitting to become a piece. `pathfind_metadata.json` is
already exactly the {key: {position, rotation, lat, lon, ...}} shape a
piece_*_meta.json has, and the chunk's cloud is already one reconstruction
in one DA3 frame -- which is the unit load_pieces GPS-fits. So a piece here
IS a chunk, renamed. (`discover_pieces` regroups chunks by GPS quality;
that is a separate concern and nothing downstream requires it.)
"""
import argparse
import gzip
import json
import os
import shutil

import numpy as np
from scipy.spatial import cKDTree

from postprocess.corridors import _metres, roads
from postprocess.gps_fit.fit import real_en
from postprocess.road_align.road_frames import smooth
from paths import DATA_DIR, FETCHED_GRAPH

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

    Kept here rather than imported from street_builder.tab, which pulls in
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


def index(chunks=None, api=None):
    """{chunk id: (metadata dict, [road ids it lies on])}.

    Metadata only -- kilobytes per chunk against tens of megabytes for a
    cloud, so the whole campus can be indexed before deciding what to pull.
    """
    graph = json.load(open(FETCHED_GRAPH))
    xy = np.array(_metres(graph["points"]))
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


def fetch(chunk_ids, out_dir, api=None):
    """Write chosen chunks as piece_N.ply + piece_N_meta.json in `out_dir`.

    The metadata is symlinked and only the cloud is written for real: a
    .ply.gz has to be decompressed and dequantised before anything can read
    it, so it cannot be shared with the download cache, but the metadata is
    already exactly what we want and copying it would just be a second copy
    of every file on a disk that has to hold the clouds too.
    """
    api = api or _api()
    files = chunk_files(api)
    os.makedirs(out_dir, exist_ok=True)
    written, total = {}, 0
    for n, cid in enumerate(sorted(chunk_ids)):
        if cid not in files:
            print(f"  {cid}: not in the repo, skipped")
            continue
        ply_rel, meta_rel = files[cid]
        ply = os.path.join(out_dir, f"piece_{n}.ply")
        with gzip.open(_download(ply_rel), "rb") as f_in, open(ply, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        _dequantize(ply)
        meta_dst = os.path.join(out_dir, f"piece_{n}_meta.json")
        if os.path.lexists(meta_dst):
            os.unlink(meta_dst)
        os.symlink(os.path.realpath(_download(meta_rel)), meta_dst)
        meta = json.load(open(meta_dst))
        written[n] = cid
        total += os.path.getsize(ply)
        print(f"  piece_{n} <- {cid}  ({len(meta)} node(s), "
              f"{os.path.getsize(ply) / 1e6:.0f} MB)")
    print(f"  {total / 1e6:.0f} MB of cloud written; metadata symlinked")
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
        print(f"{len(ids)} piece(s) -> {len(want)} chunk(s)")
    else:
        raise SystemExit("give --list, --roads, --chunks or --pieces")

    print(f"fetching {len(want)} chunk(s) into {args.out}")
    fetch(want, args.out, api=api)
    print(f"\nnow:  python -m postprocess.road_align.run --dir {args.out}")


if __name__ == "__main__":
    main()
