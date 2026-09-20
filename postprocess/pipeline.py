"""Everything after the GPU, as one call.

A reconstruction run leaves a directory of clouds that are each internally
consistent but individually placed. This turns that into one scene:

  SPLIT   break any piece where DA3 stopped agreeing with its own GPS
  ALIGN   seat every piece on its road and on the real ground
  WRITE   one .ply

    python -m postprocess.pipeline --dir splats/<run id>
"""
import argparse
import os

import numpy as np

from postprocess.gps_fit.discover_pieces import FIT_THRESHOLD_M
from postprocess.gps_fit.split_by_fit import split
from postprocess.ply_io import write_ply
from postprocess.road_align.run import align

OUT_PLY = "aligned.ply"


def process(run_dir, threshold=FIT_THRESHOLD_M, min_nodes=2, log=print):
    """Run everything after the GPU. Returns the path of the written cloud."""
    run_dir = os.path.expanduser(run_dir)
    split(run_dir, threshold=threshold, log=log)
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
    ap.add_argument("--threshold", type=float, default=FIT_THRESHOLD_M,
                    help="metres a node may sit from its GPS before it is cut off")
    ap.add_argument("--min-nodes", type=int, default=2)
    args = ap.parse_args()
    process(args.dir, args.threshold, args.min_nodes)


if __name__ == "__main__":
    main()
