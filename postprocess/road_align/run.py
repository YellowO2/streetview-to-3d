"""Align a set of pieces to the road network and write them out as one cloud.

    python -m alignment.run_global_alignment --dir /tmp/gap3_joined \
        --pieces 5,3,9,6 --out ~/Downloads/aligned.ply

The stages, and why they are in this order:

  1. ROADS      the walking graph split into roads, each with its own
                frame, and each piece matched to the roads it lies along
                (road_align.road_frames). The frame fixes a direction of
                travel, so "left" means the same side in every piece.
  2. LINES      each piece reduced to a left kerb, a centre and a right
                kerb -- once per road it lies along, from just the part of
                it beside that road (road_align.extract_road_lines).
  3. HORIZONTAL every piece turned and slid onto its roads' global curves
                (road_align.fit_pieces_to_road). Not pairwise: a piece
                needs no neighbour, only a road.
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

from postprocess.road_align.extract_road_lines import road_lines
from postprocess.road_align.road_frames import build as build_frames, clip, near_cams
from postprocess.road_align.feature_icp import extract_features
from postprocess.road_align.fit_pieces_to_road import (RoadFitter, drift_cap,
                                                       horizontal_transform,
                                                       _rot as _rot2,
                                                       TURN_RANGE_DEG)
from scipy.optimize import minimize
from postprocess.piece_transforms import save as save_transforms
from postprocess.gps_fit.load_pieces import load_pieces
from scipy.spatial import cKDTree
from postprocess.road_align.seat_pieces_on_surface import seat

MARGIN_M = 25.0
STRONG_NODES = 2         # above this, GPS measured the heading itself
MIN_CLIPPED_PTS = 5000   # too little of the piece on this road to describe it


def align(directory, piece_ids=None, cell=0.25, log=print, min_nodes=1,
          road_fit=True):
    """Solve, and return ({piece: 4x4}, clouds, fits, per-piece diagnostics).

    The 4x4s here map ALREADY GPS-FITTED coordinates, because load_pieces
    applies the GPS fit as it loads. `piece_transforms.save` composes the
    two so the file that lands on disk stands on its own.
    """
    fits, clouds = load_pieces(directory)
    ids = [i for i in (piece_ids or sorted(clouds)) if i in clouds]
    if min_nodes > 1:
        # Dropped before anything else, so a weak piece cannot shape a
        # curve or be exported. It cannot affect the shared scale either:
        # global_scale only counts pieces that fitted their own.
        dropped = [i for i in ids if fits[i]["n"] < min_nodes]
        ids = [i for i in ids if i not in set(dropped)]
        if dropped:
            log(f"dropped {len(dropped)} piece(s) with < {min_nodes} node(s): "
                + ", ".join(f"piece_{i}" for i in dropped) + "\n")
    if len(ids) < 2:
        raise SystemExit(f"need at least 2 pieces, got {ids}")

    cams = {i: fits[i]["cams"] for i in ids}
    allc = np.vstack([cams[i] for i in ids])
    bounds = (allc[:, 0].min() - MARGIN_M, allc[:, 0].max() + MARGIN_M,
              allc[:, 1].min() - MARGIN_M, allc[:, 1].max() + MARGIN_M)

    curves, frames, on = build_frames(cams)

    # STAGE 1 -- seat each piece on its road line.
    #
    # The cameras are that piece's own panorama GPS; the road line is the
    # same GPS smoothed over every panorama on the road, so it is the less
    # noisy of the two. How much freedom a piece gets depends on what GPS
    # actually measured about it:
    #
    #   > STRONG_NODES cameras   translation only. GPS saw the piece from
    #                            enough places to have measured its heading,
    #                            so overruling that with the road line would
    #                            replace a real measurement with a smoothed
    #                            one.
    #   <= STRONG_NODES cameras  rotation as well. GPS never measured a
    #                            heading here at all -- two points fix a
    #                            position and nothing else -- so the road
    #                            line is the only thing that can supply one.
    #
    # Both are held near GPS by a soft (move/cap)^4 penalty rather than a
    # hard limit, which also forbids the along-road slide an unpenalised fit
    # runs away with: a road is featureless lengthwise, and one piece ran
    # 34 m down the curve before this penalty existed.
    nudge = {}
    for i in ids:
        if not on.get(i):
            nudge[i] = np.eye(4)
            continue
        r = min(on[i], key=lambda r: cKDTree(curves[r]).query(cams[i])[0].mean())
        tree, c = cKDTree(curves[r]), cams[i]
        piv, cap = c.mean(0), drift_cap(len(c))
        free_turn = len(c) <= STRONG_NODES

        def cost(p, tree=tree, c=c, piv=piv, cap=cap):
            q = (c - piv) @ _rot2(p[0]).T + piv + p[1:]
            return ((tree.query(q)[0] ** 2).mean()
                    + (np.linalg.norm(q - c, axis=1).mean() / cap) ** 4)

        best = None
        turns = (np.arange(-TURN_RANGE_DEG, TURN_RANGE_DEG + 0.1, 2.0)
                 if free_turn else [0.0])
        for d in turns:
            o = minimize(lambda t, d=d: cost(np.r_[d, t]), [0.0, 0.0],
                         method="Nelder-Mead",
                         options=dict(xatol=1e-2, fatol=1e-4, maxiter=400))
            if best is None or o.fun < best[0]:
                best = (o.fun, d, o.x)
        _, deg, t = best
        nudge[i] = horizontal_transform(np.r_[deg, t], piv)
        log(f"  piece_{i:<3} {len(c)} node(s)  "
            + (f"turn {deg:+5.0f} deg, " if free_turn else "no turn (GPS fixed it), ")
            + f"shift {np.linalg.norm(t):.2f} m")

    gps = {i: cams[i].copy() for i in ids}          # kept only for reporting
    raw = dict(clouds)
    for i in ids:
        M = nudge[i]
        xz, y, co = clouds[i]
        clouds[i] = (xz @ M[[0, 2]][:, [0, 2]].T + M[[0, 2], 3], y, co)
        cams[i] = cams[i] @ M[[0, 2]][:, [0, 2]].T + M[[0, 2], 3]
    log("stage 1, seated on the road line: "
        + "  ".join(f"{i}:{np.linalg.norm(cams[i] - gps[i], axis=1).mean():.2f}m"
                    for i in ids) + "\n")

    road_pts = {i: extract_features(clouds[i], bounds, cams=cams[i])[0]
                for i in ids}

    horiz = {i: np.eye(4) for i in ids}
    if road_fit:
        lines, skipped = {}, []
        for i in ids:
            got = []
            for r in on.get(i, []):
                # each road sees only the part of the piece lying along it,
                # so a piece at a junction yields a separate set of lines per
                # road instead of one set describing the junction blob
                sub, keep = clip(clouds[i], curves[r])
                if keep.sum() < MIN_CLIPPED_PTS:
                    continue
                c = near_cams(cams[i], curves[r])
                b = (sub[0][:, 0].min() - MARGIN_M, sub[0][:, 0].max() + MARGIN_M,
                     sub[0][:, 1].min() - MARGIN_M, sub[0][:, 1].max() + MARGIN_M)
                found = road_lines(sub, b, c, frames[r], cell)
                if found is not None:
                    lines[(i, r)] = found
                    got.append(r)
            if not got:
                skipped.append(i)
                log(f"piece_{i}: no usable road lines, left where stage 1 put it")
                continue
            log(f"piece_{i}: {len(cams[i])} node(s) on road(s) "
                f"{','.join(str(r) for r in got)}")

        fitted = sorted({i for i, _ in lines})
        if len(fitted) < 2:
            raise SystemExit("fewer than 2 pieces yielded road lines")

        anchors = {i for i in fitted if fits[i]["n"] > 1}
        followers = sorted(set(fitted) - anchors)
        if followers:
            log(f"\n{len(anchors)} anchor(s) build the curves; "
                f"{len(followers)} singleton(s) placed after: "
                + ", ".join(f"piece_{i}" for i in followers))
        log("")
        fitter = RoadFitter(lines, {i: cams[i] for i in fitted}, frames, anchors)
        fitter.solve(log=log)

        log("\npiece   turn    slide   drift    cap    to road (L/C/R)   roads")
        for i, (turn, slide, drift, d, rds, follower) in fitter.report().items():
            log(f"  {i:<5}{turn:+7.1f} {slide:7.2f}m {drift:6.2f}m "
                f"{fitter.cap[i]:5.1f}m   " + "/".join(f"{x:.2f}" for x in d)
                + "   " + ",".join(str(r) for r in rds)
                + ("   (singleton, placed after)" if follower else ""))
        horiz.update({i: horizontal_transform(fitter.state[i], fitter.pivot[i])
                      for i in fitted})
        fit_report = fitter.report()
    else:
        log("\nkerb fit OFF -- stage 1 and seating only")
        fit_report = {}

    moved_road = {i: road_pts[i] @ horiz[i][:3, :3].T + horiz[i][:3, 3]
                  for i in ids if len(road_pts[i])}
    vert, report = seat(moved_road)
    log("")
    for i in ids:
        if i in report:
            log(f"piece_{i}: seated {report[i][0]:+.2f} m, tilt {report[i][1]:.2f} deg")
        else:
            log(f"piece_{i}: too little road to seat, height left at GPS")

    diagnostics = {}
    for i in ids:
        d = {"matrix": (vert.get(i, np.eye(4)) @ horiz[i]).tolist()}
        if i in fit_report:
            turn, slide, drift, to_road, rds, follower = fit_report[i]
            d["road"] = {"turn_deg": round(turn, 2), "slide_m": round(slide, 3),
                         "drift_m": round(drift, 3), "cap_m": fitter.cap[i],
                         "to_road_m": [round(float(x), 3) for x in to_road],
                         "roads": rds, "placed_after": follower}
        else:
            d["road"] = None            # no usable road lines; left at GPS
        if i in report:
            d["seating"] = {"height_m": round(report[i][0], 3),
                            "tilt_deg": round(report[i][1], 2)}
        else:
            d["seating"] = None         # too little road to seat
        diagnostics[i] = d

    # the nudge happened before the fit, so it sits innermost
    return ({i: vert.get(i, np.eye(4)) @ horiz[i] @ nudge[i] for i in ids},
            {i: raw[i] for i in ids}, {i: fits[i] for i in ids},
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
                    help="drop pieces with fewer GPS nodes than this. Use 2 "
                         "to exclude singletons, whose heading GPS never "
                         "measured and which borrow one from a neighbour. "
                         "Defaults to 2: a singleton bends the curves it is "
                         "then fitted to, and the results were worse with "
                         "them in.")
    ap.add_argument("--no-save", action="store_true",
                    help="solve without writing piece_transforms.json")
    args = ap.parse_args()

    ids = ([int(x) for x in args.pieces.split(",")] if args.pieces else None)
    transforms, clouds, fits, diagnostics = align(args.dir, ids, args.cell,
                                                 min_nodes=args.min_nodes)

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
