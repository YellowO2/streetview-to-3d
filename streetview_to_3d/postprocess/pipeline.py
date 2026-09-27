"""Everything after the GPU, as one call.

A reconstruction leaves a scene whose nodes each hold their own points, in
whatever frame DA3 built them. This places every piece by its panoramas'
GPS, elevation and orientation (place.py), writing each node's own
transform back into the scene, removes floating bits (blobs.py), then fills the gaps (streetview_to_3d.fill:
one ground, Google's walls, their colour) into the node .plys --
scene.json plus the node .plys are the whole result. A viewer reads transform to place each node, so nothing
merges the points into one extra file by default.

    python -m streetview_to_3d.postprocess.pipeline --dir data/runs/<run id>
    python -m streetview_to_3d.postprocess.pipeline --dir data/runs/<run id> --merge out.ply
    python -m streetview_to_3d.postprocess.pipeline --dir data/runs/<run id> --no-fill
"""
import argparse
import os
import time

from streetview_to_3d import fill as fill_mod
from streetview_to_3d.postprocess.blobs import drop_blobs
from streetview_to_3d.postprocess.render_pieces import render
from streetview_to_3d.postprocess.place import place


def process(run_dir, log=print, merge_ply=None, fill=True):
    """Place a reconstruction, remove its floating bits, then fill it
    (fill=False: placed and cleaned only).
    merge_ply: also write one combined .ply there, for local inspection
    outside the viewer -- not needed by the Space, which reads
    scene.json's per-node transforms directly."""
    run_dir = os.path.expanduser(run_dir)
    t = time.monotonic()
    place(run_dir, log=log)
    log(f"timing: placement {time.monotonic() - t:.1f}s")
    t = time.monotonic()
    drop_blobs(run_dir, log=log)
    log(f"timing: floating bits {time.monotonic() - t:.1f}s")
    if fill:
        t = time.monotonic()
        try:
            fill_mod.run(run_dir, log=log)
        except Exception as e:           # the placed scene is still a result
            log(f"fill skipped: {e!r}")
        log(f"timing: fill {time.monotonic() - t:.1f}s")
    return render(run_dir, merge_ply, log=log) if merge_ply else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="a reconstruction run's directory")
    ap.add_argument("--merge", help="also write one combined .ply here")
    ap.add_argument("--no-fill", action="store_true", help="place only, no gap fill")
    args = ap.parse_args()
    process(args.dir, merge_ply=args.merge, fill=not args.no_fill)


if __name__ == "__main__":
    main()
