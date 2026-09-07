"""Build a .ply from pieces already solved, without solving anything.

    python -m postprocess.render_pieces --dir DIR --pieces 5,9 --out x.ply

This is the point of saving transforms. Solving a set takes minutes and
the answer is one matrix per piece; rendering any subset of it should be
reading files and multiplying, which is what this does. Rendering four
pieces and rendering one differ only in how many files get read.

Reads piece_transforms.json, so run postprocess.run_global_alignment on
the directory first.
"""
import argparse
import os

import numpy as np

from postprocess.piece_transforms import load, report
from postprocess.ply_io import write_ply
from postprocess.run_road_align import _read_ply_points


def render(directory, piece_ids=None, out=None, log=print):
    """Apply the saved transforms and optionally write the result."""
    transforms = load(directory)
    ids = [i for i in (piece_ids or sorted(transforms)) if i in transforms]
    missing = [i for i in (piece_ids or []) if i not in transforms]
    if missing:
        log(f"not in piece_transforms.json, skipped: {missing}")
    if not ids:
        raise SystemExit("no pieces to render")

    pts, cols = [], []
    for i in ids:
        raw, colour = _read_ply_points(os.path.join(directory, f"piece_{i}.ply"))
        T = transforms[i]
        pts.append(raw @ T[:3, :3].T + T[:3, 3])
        cols.append(colour)
        log(f"  piece_{i}: {len(raw)} points")

    pts = np.concatenate(pts)
    cols = np.concatenate(cols)
    if out:
        out = os.path.expanduser(out)
        write_ply(out, pts, cols)
        log(f"wrote {len(pts)} points to {out}")
    return pts, cols


def _show(doc):
    """What was solved, and how far each piece had to move for it."""
    print(f"solved {doc['written']} from {doc['source_dir']}")
    print(f"{'piece':>7} {'nodes':>6} {'turn':>8} {'drift':>8} "
          f"{'height':>8} {'tilt':>7}")
    for k, v in doc["pieces"].items():
        road, seat = v.get("road"), v.get("seating")
        turn = f"{road['turn_deg']:+.1f}" if road else "--"
        drift = f"{road['drift_m']:.2f}m" if road else "--"
        height = f"{seat['height_m']:+.2f}m" if seat else "--"
        tilt = f"{seat['tilt_deg']:.2f}" if seat else "--"
        print(f"{k:>7} {v['nodes']:>6} {turn:>8} {drift:>8} {height:>8} {tilt:>7}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="/tmp/gap3_joined")
    ap.add_argument("--pieces", default=None,
                    help="comma-separated ids (default: every solved piece)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", action="store_true",
                    help="print what was solved, and how far each piece moved")
    args = ap.parse_args()

    if args.show:
        _show(report(args.dir))

    ids = [int(x) for x in args.pieces.split(",")] if args.pieces else None
    render(args.dir, ids, args.out)


if __name__ == "__main__":
    main()
