"""Finish a reconstruction run's folder in place: place, clean, fill and lay the world.

    python -m streetview_to_3d.postprocess --dir data/runs/<run id>
    python -m streetview_to_3d.postprocess --dir data/runs/<run id> --merge out.ply
    python -m streetview_to_3d.postprocess --dir data/runs/<run id> --no-fill
    python -m streetview_to_3d.postprocess --dir data/runs/<run id> --no-clean
"""
import argparse
import os

from streetview_to_3d.load import load
from streetview_to_3d.postprocess.Settings import Settings
from streetview_to_3d.postprocess.main import postprocess
from streetview_to_3d.postprocess.merged import merged
from streetview_to_3d.save import save


def run(run_dir, log=print, merge_ply=None, fill=True, clean=True):
    """postprocess on the scene saved in run_dir, saved back there; merge_ply: also write
    one combined world-frame .ply there, for inspection."""
    run_dir = os.path.expanduser(run_dir)
    scene, ground, world = postprocess(load(run_dir), Settings(fill=fill, clean=clean), run_dir, log)
    save(scene, ground, world, run_dir)
    return merged(scene, os.path.expanduser(merge_ply), log) if merge_ply else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="a reconstruction run's directory")
    ap.add_argument("--merge", help="also write one combined .ply here")
    ap.add_argument("--no-fill", action="store_true", help="place only, no gap fill")
    ap.add_argument("--no-clean", action="store_true", help="leave the loose sheets (drop_sheets.py) in")
    args = ap.parse_args()
    run(args.dir, merge_ply=args.merge, fill=not args.no_fill, clean=not args.no_clean)


if __name__ == "__main__":
    main()
