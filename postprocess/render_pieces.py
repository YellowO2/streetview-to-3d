"""Build a cloud from a scene that has already been solved.

Every node carries its own transform, so this is a matrix multiply against
clouds already on disk with nothing re-solved. Run postprocess.pipeline
first if the nodes have no transform yet.

    python -m postprocess.render_pieces --dir splats/<run id> --out scene.ply
"""
import argparse
import os

import numpy as np

import scene as scene_mod
from postprocess.ply_io import write_ply
from reconstruct.join_segments import _read_ply_points


def render(directory, out, log=print):
    sc = scene_mod.Scene.load(directory)
    ready = [n for n in sc.nodes if n.ply and n.transform]
    if not ready:
        raise ValueError(f"no node in {directory} has been placed yet -- "
                         "run postprocess.pipeline first")
    skipped = sum(1 for n in sc.nodes if n.ply and not n.transform)
    if skipped:
        log(f"{skipped} node(s) have points but no transform, skipped")

    pts, cols = [], []
    for n in ready:
        p, c = _read_ply_points(os.path.join(directory, n.ply))
        T = np.array(n.transform)
        pts.append(p @ T[:3, :3].T + T[:3, 3])
        cols.append(c)
    pts, cols = np.concatenate(pts), np.concatenate(cols)
    write_ply(os.path.expanduser(out), pts, cols)
    log(f"wrote {len(pts):,} points from {len(ready)} node(s) to {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="a scene directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    render(args.dir, args.out)


if __name__ == "__main__":
    main()
