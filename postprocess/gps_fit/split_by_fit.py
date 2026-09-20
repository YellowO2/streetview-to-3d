"""Break pieces where the reconstruction stops agreeing with its own GPS.

A piece is placed by ONE rotation, scale and shift, so it can only be put
in the right place if it is internally the right shape. DA3 drifts along a
long run: measured on NTU chunk40, error climbed smoothly from 4 m at one
end to 18 m at the other, and no rigid transform straightens that.

Cutting the piece where the drift exceeds a threshold turns the problem
into one we can solve -- a bend inside a piece becomes a gap between two
pieces, and road alignment closes gaps.

    python -m postprocess.gps_fit.split_by_fit --dir DIR
"""
import argparse
import os

import numpy as np
from scipy.spatial import cKDTree

import scene as scene_mod
from postprocess.gps_fit.discover_pieces import FIT_THRESHOLD_M, resolve, residuals_for
from postprocess.gps_fit.fit import real_en, use_origin

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


def split(directory, threshold=FIT_THRESHOLD_M, log=print):
    """Regroup a scene's nodes so no piece spans a break in the fit.

    Nothing is copied or rewritten: points belong to nodes, so breaking a
    piece is entirely a matter of which nodes are listed together. The
    scene is saved back over itself with the new grouping.
    """
    sc = scene_mod.Scene.load(directory)
    use_origin(*sc.origin)
    out = scene_mod.Scene(center=sc.center, graph=sc.graph)

    for piece in sc.pieces:
        parts = split_piece(piece.nodes, threshold)
        before = _residual(piece.nodes)
        log(f"{len(piece)} node(s), residual "
            f"{'n/a' if before is None else f'{before:.1f} m'} -> {len(parts)} piece(s)")
        for part in sorted(parts, key=len, reverse=True):
            nodes = [n for n in piece.nodes if n.key in part]
            # an edge leaving the part is exactly the link being cut
            inside = [e for e in piece.edges if e.a in part and e.b in part]
            out.pieces.append(scene_mod.Piece(nodes=nodes, edges=inside))
            after = _residual(nodes)
            log(f"   piece {len(out.pieces) - 1}: {len(nodes):>3} node(s), residual "
                f"{'n/a' if after is None else f'{after:.2f} m'}")
    out.save(directory)
    log(f"{len(sc.pieces)} piece(s) -> {len(out.pieces)}")
    return len(out.pieces)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="a scene directory")
    ap.add_argument("--threshold", type=float, default=FIT_THRESHOLD_M,
                    help="metres a node may sit from its GPS before it is cut off")
    args = ap.parse_args()
    split(os.path.expanduser(args.dir), args.threshold)


if __name__ == "__main__":
    main()
