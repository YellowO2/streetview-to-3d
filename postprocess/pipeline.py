"""Everything after the GPU, as one call.

A reconstruction run leaves a directory of clouds that are each internally
consistent but individually placed. This turns that into one scene:

  ADOPT   name the run's output as pieces
  SPLIT   break any piece where DA3 stopped agreeing with its own GPS
  ALIGN   seat every piece on its road and on the real ground
  WRITE   one .ply

    python -m postprocess.pipeline --dir splats/<run id>
"""
import argparse
import os
import shutil

import numpy as np

import area
from postprocess.gps_fit.discover_pieces import FIT_THRESHOLD_M
from postprocess.gps_fit.split_by_fit import split
from postprocess.ply_io import write_ply
from postprocess.road_align.run import align

PIECES = "pieces"        # where the named-and-split pieces land
RAW = "raw"              # the run's own output, before splitting
OUT_PLY = "aligned.ply"


def adopt(run_dir, out_dir, log=print):
    """Name a run's joined output as pieces. Returns how many there are.

    street_builder writes pathfind_joined.ply (or _0, _1, ... when
    bridging left regions genuinely unconnected); postprocess works in
    piece_N.ply. This is only a rename -- nothing is re-cut here.
    """
    os.makedirs(out_dir, exist_ok=True)
    for name in (area.FILENAME, area.GRAPH):
        shutil.copy(os.path.join(run_dir, name), out_dir)

    found = sorted(n for n in os.listdir(run_dir)
                   if n.startswith("pathfind_joined") and n.endswith(".ply"))
    for n, name in enumerate(found):
        suffix = name[len("pathfind_joined"):-len(".ply")]
        shutil.copy(os.path.join(run_dir, name),
                    os.path.join(out_dir, f"piece_{n}.ply"))
        shutil.copy(os.path.join(run_dir, f"pathfind_metadata{suffix}.json"),
                    os.path.join(out_dir, f"piece_{n}_meta.json"))
    log(f"adopted {len(found)} piece(s) from the run")
    return len(found)


def process(run_dir, threshold=FIT_THRESHOLD_M, min_nodes=2, log=print):
    """Run everything after the GPU. Returns the path of the written cloud."""
    run_dir = os.path.expanduser(run_dir)
    raw, pieces = os.path.join(run_dir, RAW), os.path.join(run_dir, PIECES)

    if not adopt(run_dir, raw, log=log):
        raise ValueError(f"no pathfind_joined*.ply in {run_dir} -- nothing to place")
    split(raw, pieces, threshold=threshold, log=log)
    transforms, clouds, _, _ = align(pieces, min_nodes=min_nodes, log=log)

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
