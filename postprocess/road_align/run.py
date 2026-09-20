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
                 dot elevations (road_align.ground_elevation).
                 After the horizontal fit, never before -- otherwise it
                 levels pieces against road that is not the same road yet.

Road polylines and camera positions are both in the shared GLOBAL_ORIGIN
metre frame (gps_fit.fit.real_en), so they compare directly.
"""
import argparse
import os

import numpy as np

from postprocess.road_align.road_surface import ground_near_track
from postprocess.road_align.road_frames import build as build_frames
from postprocess.road_align.node_center_to_road_line import seat_all
from postprocess.gps_fit.load_pieces import load_pieces
from postprocess.road_align.align_slope_of_pieces import seat
from postprocess.road_align import ground_elevation
import scene as scene_mod

MARGIN_M = 25.0


def align(directory, piece_ids=None, cell=0.25, log=print, min_nodes=2,
          elevation=True, save=True):
    """Solve, and return ({piece: 4x4}, clouds, fits, per-piece diagnostics).

    The 4x4s returned map ALREADY GPS-FITTED coordinates, because
    load_pieces applies the GPS fit as it loads. What is saved onto each
    node composes the two, so a node's matrix stands on its own.
    """
    sc = scene_mod.Scene.load(directory)
    fits, clouds = load_pieces(directory)
    ids = [i for i in (piece_ids or sorted(clouds)) if i in clouds]
    if min_nodes > 1:
        # Dropped before anything else, so a weak piece cannot be exported
        # or shape anything.
        dropped = [i for i in ids if fits[i]["n"] < min_nodes]
        ids = [i for i in ids if i not in set(dropped)]
        if dropped:
            log(f"dropped {len(dropped)} piece(s) with < {min_nodes} node(s): "
                + ", ".join(f"piece_{i}" for i in dropped) + "\n")
    if not ids:
        raise ValueError("no pieces to place")

    # the cameras as the cloud actually carries them. Seating against their
    # GPS positions instead measures nothing: the road line is splined
    # through those very points, so they start on it by construction.
    cams = {i: fits[i]["placed"] for i in ids}
    allc = np.vstack([cams[i] for i in ids])
    bounds = (allc[:, 0].min() - MARGIN_M, allc[:, 0].max() + MARGIN_M,
              allc[:, 1].min() - MARGIN_M, allc[:, 1].max() + MARGIN_M)

    curves, frames, on = build_frames(cams, sc)
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
        ground, resid, n_dots = ground_elevation.surface(sc, curves, frames, on)
        log(f"\nground from {n_dots} node elevation(s) along "
            f"{len(curves)} road(s), profiles fit them to {resid:.2f} m")
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

    transforms = {i: vert.get(i, np.eye(4)) @ horiz[i] for i in ids}
    if save:
        for i in ids:
            M = np.asarray(gps_transform(fits[i]))
            for m in fits[i]["members"]:
                sc.nodes[m].transform = (transforms[i] @ M).tolist()
        sc.save(directory)
    return (transforms, {i: clouds[i] for i in ids},
            {i: fits[i] for i in ids}, diagnostics)


def gps_transform(fit):
    """The GPS fit as a 4x4: a node's stored ply -> world metres.

    load_pieces applies this while loading, so a transform solved on top of
    it only maps already-fitted coordinates. Composing the two is what lets
    a node's saved matrix stand alone.
    """
    R, s, t = fit["R"], float(fit["scale"]), fit["t"]
    T = np.eye(4)
    T[0, 0], T[0, 2] = s * R[0, 0], s * R[0, 1]
    T[2, 0], T[2, 2] = s * R[1, 0], s * R[1, 1]
    T[1, 1] = s
    T[0, 3], T[2, 3] = t[0], t[1]
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True,
                    help="directory holding a scene.json and its clouds")
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
                    help="solve without writing the answer back into the scene")
    args = ap.parse_args()

    ids = ([int(x) for x in args.pieces.split(",")] if args.pieces else None)
    transforms, clouds, fits, diagnostics = align(args.dir, ids, args.cell,
                                                  min_nodes=args.min_nodes,
                                                  elevation=not args.no_elevation,
                                                  save=not args.no_save)

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
