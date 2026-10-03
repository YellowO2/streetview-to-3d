"""Everything after the GPU, as one call: place, remove floating bits and
loose sheets, fill, then lay the map around the scene. scene.json and the
.plys it names are the result.

    python -m streetview_to_3d.postprocess.pipeline --dir data/runs/<run id>
    python -m streetview_to_3d.postprocess.pipeline --dir data/runs/<run id> --merge out.ply
    python -m streetview_to_3d.postprocess.pipeline --dir data/runs/<run id> --no-fill
    python -m streetview_to_3d.postprocess.pipeline --dir data/runs/<run id> --no-clean
"""
import argparse
import os
import time

from streetview_to_3d.postprocess import fill as fill_mod
from streetview_to_3d.postprocess.clean import clean as clean_mod
from streetview_to_3d.postprocess.world import terrain
from streetview_to_3d.postprocess.clean.blobs import drop_blobs
from streetview_to_3d.postprocess.render_pieces import render
from streetview_to_3d.postprocess.place import place


def process(run_dir, log=print, merge_ply=None, fill=True, clean=True):
    """Place, clean, fill and lay the terrain for run_dir.

    fill=False: place only. clean=False: keep loose sheets. merge_ply: also
    write one combined .ply for local inspection."""
    run_dir = os.path.expanduser(run_dir)
    t = time.monotonic()
    place(run_dir, log=log)
    log(f"timing: placement {time.monotonic() - t:.1f}s")
    if fill:
        t = time.monotonic()
        drop_blobs(run_dir, log=log)
        log(f"timing: floating bits {time.monotonic() - t:.1f}s")
        if clean:
            t = time.monotonic()
            clean_mod.clean(run_dir, log=log)
            log(f"timing: loose sheets {time.monotonic() - t:.1f}s")
        t = time.monotonic()
        try:
            fill_mod.run(run_dir, log=log)
        except Exception as e:           # the placed scene is still a result
            log(f"fill skipped: {e!r}")
        log(f"timing: fill {time.monotonic() - t:.1f}s")
        t = time.monotonic()
        try:                             # after the fill: its panos colour the buildings
            terrain.build(run_dir, log=log)
        except Exception as e:           # a map download failing costs only the land around
            log(f"terrain skipped: {e!r}")
        log(f"timing: terrain {time.monotonic() - t:.1f}s")
    return render(run_dir, merge_ply, log=log) if merge_ply else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="a reconstruction run's directory")
    ap.add_argument("--merge", help="also write one combined .ply here")
    ap.add_argument("--no-fill", action="store_true", help="place only, no gap fill")
    ap.add_argument("--no-clean", action="store_true", help="leave the loose sheets (clean.py) in")
    args = ap.parse_args()
    process(args.dir, merge_ply=args.merge, fill=not args.no_fill, clean=not args.no_clean)


if __name__ == "__main__":
    main()
