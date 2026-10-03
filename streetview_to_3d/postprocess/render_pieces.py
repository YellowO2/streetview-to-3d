"""Merge a placed scene's nodes into one world-frame .ply, for inspection.

    python -m streetview_to_3d.postprocess.render_pieces --dir data/runs/<run id> --out scene.ply
"""
import argparse
import os

import numpy as np

from streetview_to_3d import scene as scene_mod
from streetview_to_3d.postprocess.ply_io import read_node, write_ply


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
        _, c, w = read_node(directory, n)
        pts.append(w)
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
