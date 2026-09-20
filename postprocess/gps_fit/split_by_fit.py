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
import os

import numpy as np
from scipy.spatial import cKDTree

import scene as scene_mod
from postprocess.gps_fit.discover_pieces import FIT_THRESHOLD_M, resolve, residuals_for
from postprocess.gps_fit.fit import real_en, use_origin
from postprocess.ply_io import write_ply
from street_builder.reconstruction.join_segments import _read_ply_points

# Street View dots sit ~10 m apart, so cameras further apart than this are
# not neighbours and a cut between them is free.
ADJACENT_M = 25.0


def _for_resolve(nodes):
    """{key: node} in the shape discover_pieces.resolve expects."""
    return {n.key: {"da3_xz": [n.position[0], n.position[2]],
                    "real_en": list(real_en(n.lat, n.lon))}
            for n in nodes}


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


def split_piece(nodes, threshold):
    """[set of keys, ...] -- the piece, broken where it stops fitting."""
    prepared = _for_resolve(nodes)
    return [set(part) for part in
            resolve(prepared, _adjacency(prepared), threshold=threshold)]


def _residual(nodes):
    prepared = list(_for_resolve(nodes).values())
    if len(prepared) < 2:
        return None
    res, _ = residuals_for(prepared)
    return float(np.median(res))


def split(directory, out_dir, threshold=FIT_THRESHOLD_M, log=print):
    """Write a copy of a scene with every piece broken where it must be."""
    sc = scene_mod.Scene.load(directory)
    use_origin(*sc.origin)
    os.makedirs(out_dir, exist_ok=True)
    out = scene_mod.Scene(center=sc.center, graph=sc.graph)

    for i, piece in enumerate(sc.pieces):
        parts = split_piece(piece.nodes, threshold)
        before = _residual(piece.nodes)
        pts, cols = _read_ply_points(os.path.join(directory, piece.ply))
        cams = np.array([n.position for n in piece.nodes])
        nearest = cKDTree(cams).query(pts)[1]

        log(f"{piece.ply}: {len(piece)} node(s), residual "
            f"{'n/a' if before is None else f'{before:.1f} m'} -> {len(parts)} piece(s)")
        for part in sorted(parts, key=len, reverse=True):
            nodes = [n for n in piece.nodes if n.key in part]
            keep = [j for j, n in enumerate(piece.nodes) if n.key in part]
            mask = np.isin(nearest, keep)
            name = f"piece_{len(out.pieces)}.ply"
            write_ply(os.path.join(out_dir, name), pts[mask], cols[mask])
            out.pieces.append(scene_mod.Piece(ply=name, nodes=nodes))
            after = _residual(nodes)
            log(f"   {name}: {len(nodes):>3} node(s), residual "
                f"{'n/a' if after is None else f'{after:.2f} m'}, "
                f"{int(mask.sum()):,} point(s)")
    out.save(out_dir)
    log(f"{len(sc.pieces)} piece(s) -> {len(out.pieces)}")
    return len(out.pieces)


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
