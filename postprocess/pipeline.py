"""Everything after the GPU, as one call.

A reconstruction leaves a scene whose nodes each hold their own points, in
whatever frame DA3 built them. This seats every piece on its road and on
the real ground, writing each node's own transform back into the scene --
scene.json plus the node .plys already written by the GPU stage are the
whole result. A viewer reads transform to place each node, so nothing
merges the points into one extra file by default.

    python -m postprocess.pipeline --dir splats/<run id>
    python -m postprocess.pipeline --dir splats/<run id> --merge out.ply
"""
import argparse
import os

import numpy as np

from postprocess.road_align.run import align


def process(run_dir, min_nodes=2, log=print, merge_ply=None):
    """Place a reconstruction. merge_ply: also write one combined .ply
    there, for local inspection outside the viewer -- not needed by the
    Space, which reads scene.json's per-node transforms directly."""
    run_dir = os.path.expanduser(run_dir)
    transforms, clouds, _, _ = align(run_dir, min_nodes=min_nodes, log=log)
    if not merge_ply:
        return None

    pts, cols = [], []
    for i, T in transforms.items():
        xz, y, col = clouds[i]
        pts.append(np.column_stack([xz[:, 0], y, xz[:, 1]]) @ T[:3, :3].T + T[:3, 3])
        cols.append(col)
    from postprocess.ply_io import write_ply
    write_ply(merge_ply, np.concatenate(pts), np.concatenate(cols))
    log(f"\nwrote {sum(len(p) for p in pts):,} points to {merge_ply}")
    return merge_ply


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="a reconstruction run's directory")
    ap.add_argument("--min-nodes", type=int, default=2)
    ap.add_argument("--merge", help="also write one combined .ply here")
    args = ap.parse_args()
    process(args.dir, args.min_nodes, merge_ply=args.merge)


if __name__ == "__main__":
    main()
