"""Everything after the GPU, as one call.

A reconstruction leaves a scene whose nodes each hold their own points, in
whatever frame DA3 built them. This works out where those points belong:

  ALIGN   seat every piece on its road and on the real ground, writing
          each node's own transform back into the scene
  WRITE   one .ply of the result

    python -m postprocess.pipeline --dir splats/<run id>
"""
import argparse
import os

import numpy as np

from postprocess.ply_io import write_ply
from postprocess.road_align.run import align

OUT_PLY = "aligned.ply"


def process(run_dir, min_nodes=2, log=print):
    """Place a reconstruction. Returns the path of the written cloud."""
    run_dir = os.path.expanduser(run_dir)
    transforms, clouds, _, _ = align(run_dir, min_nodes=min_nodes, log=log)

    pts, cols = [], []
    for i, T in transforms.items():
        xz, y, col = clouds[i]
        pts.append(np.column_stack([xz[:, 0], y, xz[:, 1]]) @ T[:3, :3].T + T[:3, 3])
        cols.append(col)
    out = os.path.join(run_dir, OUT_PLY)
    write_ply(out, np.concatenate(pts), np.concatenate(cols))
    log(f"\nwrote {sum(len(p) for p in pts):,} points to {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="a reconstruction run's directory")
    ap.add_argument("--min-nodes", type=int, default=2)
    args = ap.parse_args()
    process(args.dir, args.min_nodes)


if __name__ == "__main__":
    main()
