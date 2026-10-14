"""Merge a placed scene's nodes into one world-frame .ply, for inspection."""
import numpy as np

from streetview_to_3d.postprocess.ply_io import write_ply


def merged(scene, out, log=print):
    """Write every placed node's points, in the world, to the .ply out; returns out."""
    ready = scene.placed()
    if not ready:
        raise ValueError("no node has been placed yet -- run the postprocess first")
    skipped = sum(1 for n in scene.nodes if n.ply and not n.transform)
    if skipped:
        log(f"{skipped} node(s) have points but no transform, skipped")
    pts = np.concatenate([n.cloud.world(n.transform) for n in ready])
    cols = np.concatenate([n.cloud.colours for n in ready])
    write_ply(out, pts, cols)
    log(f"wrote {len(pts):,} points from {len(ready)} node(s) to {out}")
    return out


if __name__ == "__main__":
    import argparse
    from streetview_to_3d.load import load
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="a scene directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    merged(load(args.dir), args.out)
