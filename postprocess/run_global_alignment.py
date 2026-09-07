"""Align a set of pieces to one road and write them out as a single cloud.

    python -m alignment.run_global_alignment --dir /tmp/gap3_joined \
        --pieces 5,3,9,6 --out ~/Downloads/aligned.ply

The stages, and why they are in this order:

  1. ROUTE      one direction of travel for the whole run, from the camera
                positions. Everything after this depends on "left" meaning
                the same side of the road in every piece.
  2. LINES      each piece reduced to a left kerb, a centre and a right
                kerb (alignment.extract_road_lines).
  3. HORIZONTAL every piece turned and slid onto one set of three global
                curves (alignment.fit_pieces_to_road). Not pairwise: a
                piece needs no neighbour, only the road.
  4. VERTICAL   every piece seated on one road surface
                (alignment.seat_pieces_on_surface). After the horizontal
                fit, never before -- otherwise it levels pieces against
                road that is not the same road yet.

How far a piece may be moved off GPS depends on how many camera nodes it
has; see fit_pieces_to_road.drift_cap.
"""
import argparse
import os

import numpy as np

from postprocess.camera_route import RouteFrame, route_curve
from postprocess.extract_road_lines import road_lines
from postprocess.feature_icp import extract_features
from postprocess.fit_pieces_to_road import RoadFitter, horizontal_transform
from postprocess.run_road_align import load_pieces
from postprocess.seat_pieces_on_surface import seat

MARGIN_M = 25.0


def align(directory, piece_ids=None, cell=0.25, log=print):
    """Returns {piece: 4x4 world transform}, plus the pieces it loaded."""
    fits, clouds = load_pieces(directory)
    ids = [i for i in (piece_ids or sorted(clouds)) if i in clouds]
    if len(ids) < 2:
        raise SystemExit(f"need at least 2 pieces, got {ids}")

    cams = {i: fits[i]["cams"] for i in ids}
    allc = np.vstack([cams[i] for i in ids])
    bounds = (allc[:, 0].min() - MARGIN_M, allc[:, 0].max() + MARGIN_M,
              allc[:, 1].min() - MARGIN_M, allc[:, 1].max() + MARGIN_M)

    frame = RouteFrame(route_curve(allc))
    log(f"route: {frame.length:.0f} m through {len(allc)} camera(s)\n")

    lines, road_pts, skipped = {}, {}, []
    for i in ids:
        road_pts[i] = extract_features(clouds[i], bounds, cams=cams[i])[0]
        got = road_lines(clouds[i], bounds, cams[i], frame, cell)
        if got is None:
            skipped.append(i)
            log(f"piece_{i}: no usable road lines, left at GPS")
            continue
        lines[i] = got
        log(f"piece_{i}: {len(cams[i])} node(s), left/centre/right = " +
            "/".join(f"{np.linalg.norm(np.diff(c, axis=0), axis=1).sum():.0f}m"
                     for c in got))

    fitted = [i for i in ids if i in lines]
    if len(fitted) < 2:
        raise SystemExit("fewer than 2 pieces yielded road lines")

    log("")
    fitter = RoadFitter(lines, {i: cams[i] for i in fitted}, frame)
    fitter.solve(log=log)

    log("\npiece   turn    slide   drift    cap    to road (L/C/R)")
    for i, (turn, slide, drift, d) in fitter.report().items():
        log(f"  {i:<5}{turn:+7.1f} {slide:7.2f}m {drift:6.2f}m "
            f"{fitter.cap[i]:5.1f}m   " + "/".join(f"{x:.2f}" for x in d))

    horiz = {i: horizontal_transform(fitter.state[i], fitter.pivot[i])
             for i in fitted}
    for i in skipped:
        horiz[i] = np.eye(4)

    moved_road = {i: road_pts[i] @ horiz[i][:3, :3].T + horiz[i][:3, 3]
                  for i in ids if len(road_pts[i])}
    vert, report = seat(moved_road)
    log("")
    for i in ids:
        if i in report:
            log(f"piece_{i}: seated {report[i][0]:+.2f} m, tilt {report[i][1]:.2f} deg")
        else:
            log(f"piece_{i}: too little road to seat, height left at GPS")

    return {i: vert.get(i, np.eye(4)) @ horiz[i] for i in ids}, clouds


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="/tmp/gap3_joined",
                    help="directory of piece_*.ply and piece_*_meta.json")
    ap.add_argument("--pieces", default=None,
                    help="comma-separated piece ids (default: all in --dir)")
    ap.add_argument("--out", default=None, help="write the aligned cloud here")
    ap.add_argument("--cell", type=float, default=0.25)
    args = ap.parse_args()

    ids = ([int(x) for x in args.pieces.split(",")] if args.pieces else None)
    transforms, clouds = align(args.dir, ids, args.cell)

    if args.out:
        from postprocess.ply_io import write_ply
        pts, cols = [], []
        for i, T in transforms.items():
            xz, y, co = clouds[i]
            pts.append(np.column_stack([xz[:, 0], y, xz[:, 1]]) @ T[:3, :3].T + T[:3, 3])
            cols.append(co)
        out = os.path.expanduser(args.out)
        write_ply(out, np.concatenate(pts), np.concatenate(cols))
        print(f"\nwrote {sum(len(p) for p in pts)} points to {out}")


if __name__ == "__main__":
    main()
