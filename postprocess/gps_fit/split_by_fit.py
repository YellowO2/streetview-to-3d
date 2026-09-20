"""Break pieces where the reconstruction stops agreeing with its own GPS.

A piece is placed by ONE rotation, scale and shift, so it can only be put
in the right place if it is internally the right shape. DA3 drifts along a
long run: measured on NTU chunk40, error climbed smoothly from 4 m at one
end to 18 m at the other, and no rigid transform straightens that.

Cutting the piece where the drift exceeds a threshold turns the problem
into one we can solve -- a bend inside a piece becomes a gap between two
pieces, and road alignment closes gaps.

    python -m postprocess.gps_fit.split_by_fit --dir DIR --out DIR_SPLIT

Points follow their nearest camera, which is the best attribution
available: a merged cloud has no per-point record of which panorama it
came from.
"""
import argparse
import json
import os
import shutil

import numpy as np
from scipy.spatial import cKDTree

import area
from postprocess.gps_fit.discover_pieces import FIT_THRESHOLD_M, resolve, residuals_for
from postprocess.gps_fit.fit import load_origin, real_en
from postprocess.ply_io import write_ply
from street_builder.reconstruction.join_segments import _read_ply_points

# Street View dots sit ~10 m apart, so cameras further apart than this are
# not neighbours and a cut between them is free.
ADJACENT_M = 25.0


def _nodes(meta):
    """{key: node} in the shape discover_pieces.resolve expects."""
    return {k: {"da3_xz": [v["position"][0], v["position"][2]],
                "real_en": list(real_en(v["lat"], v["lon"]))}
            for k, v in meta.items()}


def _adjacency(nodes):
    """Who walks next to whom, from GPS alone.

    discover_pieces normally takes this from the dataset's dot graph, but a
    piece's own cameras already trace the path they were captured along, so
    proximity recovers it without needing the graph.
    """
    keys = list(nodes)
    pos = np.array([nodes[k]["real_en"] for k in keys])
    pairs = cKDTree(pos).query_pairs(ADJACENT_M)
    adj = {k: set() for k in keys}
    for i, j in pairs:
        adj[keys[i]].add(keys[j])
        adj[keys[j]].add(keys[i])
    return adj


def split_piece(meta, threshold):
    """[{key: node}, ...] -- the piece, broken where it stops fitting."""
    nodes = _nodes(meta)
    return resolve(nodes, _adjacency(nodes), threshold=threshold)


def _residual(meta):
    nodes = list(_nodes(meta).values())
    if len(nodes) < 2:
        return None
    res, _ = residuals_for(nodes)
    return float(np.median(res))


def split(directory, out_dir, threshold=FIT_THRESHOLD_M, log=print):
    """Rewrite a pieces directory with every piece broken where it must be."""
    load_origin(directory)
    os.makedirs(out_dir, exist_ok=True)
    for name in (area.FILENAME, area.GRAPH):
        shutil.copy(os.path.join(directory, name), out_dir)

    metas = {}
    for name in sorted(os.listdir(directory)):
        if name.endswith("_meta.json"):
            i = int(name.split("_")[1])
            metas[i] = json.load(open(os.path.join(directory, name)))

    n = 0
    for i, meta in sorted(metas.items()):
        parts = split_piece(meta, threshold)
        before = _residual(meta)
        pts, cols = _read_ply_points(os.path.join(directory, f"piece_{i}.ply"))
        keys = list(meta)
        cams = np.array([meta[k]["position"] for k in keys])
        nearest = cKDTree(cams).query(pts)[1]

        log(f"piece_{i}: {len(meta)} node(s), residual "
            f"{'n/a' if before is None else f'{before:.1f} m'} -> {len(parts)} piece(s)")
        for part in sorted(parts, key=lambda p: -len(p)):
            sub = {k: meta[k] for k in part}
            idx = [j for j, k in enumerate(keys) if k in part]
            mask = np.isin(nearest, idx)
            write_ply(os.path.join(out_dir, f"piece_{n}.ply"), pts[mask], cols[mask])
            with open(os.path.join(out_dir, f"piece_{n}_meta.json"), "w") as f:
                json.dump(sub, f)
            after = _residual(sub)
            log(f"   piece_{n}: {len(sub):>3} node(s), residual "
                f"{'n/a' if after is None else f'{after:.2f} m'}, "
                f"{int(mask.sum()):,} point(s)")
            n += 1
    log(f"{len(metas)} piece(s) -> {n}")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="directory of piece_*.ply")
    ap.add_argument("--out", required=True, help="where to write the split set")
    ap.add_argument("--threshold", type=float, default=FIT_THRESHOLD_M,
                    help="metres a node may sit from its GPS before it is cut off")
    args = ap.parse_args()
    split(os.path.expanduser(args.dir), os.path.expanduser(args.out), args.threshold)


if __name__ == "__main__":
    main()
