"""Align a set of pieces to the road network and write them out as one cloud.

    python -m postprocess.road_align.run --dir /tmp/gap3_joined \
        --pieces 5,3,9,6 --out ~/Downloads/aligned.ply

The stages, and why they are in this order:

  1. ROADS       the walking graph split into roads, each with its own
                 frame, and each piece matched to the roads it lies along
                 (road_align.road_frames).
  2. HORIZONTAL  every piece seated on its road's line
                 (road_align.node_center_to_road_line). Not pairwise: a piece
                 needs no neighbour, only a road, so dropping a piece from
                 the middle of a run cannot strand its neighbours.
  3. VERTICAL    every piece seated on the real ground, from Google's
                 per-panorama elevation (road_align.ground_elevation).
                 After the horizontal fit, never before -- otherwise it
                 levels pieces against road that is not the same road yet.

Road polylines and camera positions are both in the shared GLOBAL_ORIGIN
metre frame (gps_fit.fit.real_en), so they compare directly.
"""
import argparse
import json
import os

import numpy as np

from postprocess.road_align.road_surface import ground_near_track
from postprocess.road_align.road_frames import build as build_frames
from postprocess.road_align.node_center_to_road_line import seat_all
from postprocess.piece_transforms import save as save_transforms
from postprocess.gps_fit.load_pieces import load_pieces
from postprocess.road_align.align_slope_of_pieces import seat
from postprocess.road_align import ground_elevation
import area

MARGIN_M = 25.0


def align(directory, piece_ids=None, cell=0.25, log=print, min_nodes=2,
          elevation=True):
    """Solve, and return ({piece: 4x4}, clouds, fits, per-piece diagnostics).

    The 4x4s here map ALREADY GPS-FITTED coordinates, because load_pieces
    applies the GPS fit as it loads. `piece_transforms.save` composes the
    two so the file that lands on disk stands on its own.
    """
    fits, clouds = load_pieces(directory)
    ids = [i for i in (piece_ids or sorted(clouds)) if i in clouds]
    if min_nodes > 1:
        # Dropped before anything else, so a weak piece cannot be exported
        # or shape anything. It cannot affect the shared scale either:
        # global_scale only counts pieces that fitted their own.
        dropped = [i for i in ids if fits[i]["n"] < min_nodes]
        ids = [i for i in ids if i not in set(dropped)]
        if dropped:
            log(f"dropped {len(dropped)} piece(s) with < {min_nodes} node(s): "
                + ", ".join(f"piece_{i}" for i in dropped) + "\n")
    if not ids:
        raise ValueError("no pieces to place")

    cams = {i: fits[i]["cams"] for i in ids}
    allc = np.vstack([cams[i] for i in ids])
    bounds = (allc[:, 0].min() - MARGIN_M, allc[:, 0].max() + MARGIN_M,
              allc[:, 1].min() - MARGIN_M, allc[:, 1].max() + MARGIN_M)

    curves, frames, on = build_frames(cams, area.load_graph(directory))
    log(f"{len(curves)} road(s): "
        + ", ".join(f"road{r} {frames[r].length:.0f}m" for r in sorted(curves))
        + "\n")

    horiz = seat_all(cams, curves, on, log=log)

    # seating needs road points in their PLACED positions, so the surface it
    # fits is the one the pieces actually now share
    moved = {}
    for i in ids:
        M = horiz[i]
        xz, y, co = clouds[i]
        moved[i] = (xz @ M[[0, 2]][:, [0, 2]].T + M[[0, 2], 3], y, co)
    road_pts = {i: ground_near_track(
        moved[i], cams[i] @ horiz[i][[0, 2]][:, [0, 2]].T + horiz[i][[0, 2], 3])
        for i in ids}

    have = {i: p for i, p in road_pts.items() if len(p)}
    if elevation:
        latlons, keys = {}, []
        for i in ids:
            meta = json.load(open(os.path.join(directory, f"piece_{i}_meta.json")))
            for k, v in meta.items():
                latlons[k] = (v["lat"], v["lon"])
                keys.append(k)
        el = ground_elevation.fetch(keys, latlons, directory, log=log)
        ground, resid = ground_elevation.surface(el, latlons)
        log(f"\nground from {len(el)} panorama elevation(s), "
            f"surface fits them to {resid:.2f} m")
        vert, report = ground_elevation.seat_on(have, ground)
    else:
        vert, report = seat(have)
    log("")
    for i in ids:
        if i in report:
            log(f"  piece_{i:<4} seated {report[i][0]:+.2f} m, "
                f"tilt {report[i][1]:.2f} deg")
        else:
            log(f"  piece_{i:<4} too little road to seat, height left at GPS")

    diagnostics = {}
    for i in ids:
        M = vert.get(i, np.eye(4)) @ horiz[i]
        d = {"matrix": M.tolist(),
             "road": {"nodes": fits[i]["n"],
                      "roads": sorted(on.get(i, [])),
                      "moved_m": round(float(np.linalg.norm(
                          cams[i] @ horiz[i][[0, 2]][:, [0, 2]].T
                          + horiz[i][[0, 2], 3] - cams[i], axis=1).mean()), 3)}}
        d["seating"] = ({"height_m": round(report[i][0], 3),
                         "tilt_deg": round(report[i][1], 2)}
                        if i in report else None)
        diagnostics[i] = d

    return ({i: vert.get(i, np.eye(4)) @ horiz[i] for i in ids},
            {i: clouds[i] for i in ids}, {i: fits[i] for i in ids},
            diagnostics)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="/tmp/gap3_joined",
                    help="directory of piece_*.ply and piece_*_meta.json")
    ap.add_argument("--pieces", default=None,
                    help="comma-separated piece ids (default: all in --dir)")
    ap.add_argument("--out", default=None, help="write the aligned cloud here")
    ap.add_argument("--cell", type=float, default=0.25)
    ap.add_argument("--min-nodes", type=int, default=2,
                    help="drop pieces with fewer GPS nodes than this. Defaults "
                         "to 2: a single-node piece has no heading of its own "
                         "and borrows one from a neighbour, and the results "
                         "were worse with them in.")
    ap.add_argument("--no-elevation", action="store_true",
                    help="set heights by fitting a surface to the pieces "
                         "themselves instead of to Google's elevation. "
                         "Circular, and it flattens real terrain.")
    ap.add_argument("--no-save", action="store_true",
                    help="solve without writing piece_transforms.json")
    args = ap.parse_args()

    ids = ([int(x) for x in args.pieces.split(",")] if args.pieces else None)
    transforms, clouds, fits, diagnostics = align(args.dir, ids, args.cell,
                                                  min_nodes=args.min_nodes,
                                                  elevation=not args.no_elevation)

    if not args.no_save:
        path = save_transforms(args.dir, fits, diagnostics,
                               meta={"cell_m": args.cell})
        print(f"\nwrote {path}")

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
