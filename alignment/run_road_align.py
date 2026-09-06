"""Run road alignment over a whole directory of joined pieces.

    python -m alignment.run_road_align --dir /tmp/gap3_joined --out out.ply

Order matters. Pieces are aligned outward from the one with the best GPS
fit, strongest first, and each is aligned against everything already
placed -- never against another weak piece. A single-node piece is the
least trustworthy thing in the set, so it must never end up as the
bridge that two solid pieces are related through.
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial import cKDTree

from street_builder.reconstruction.join_segments import _read_ply_points
from tests.export_island_ply import write_ply
from alignment.gps import fit_nodes, real_en
from alignment.road import (CELL, align_piece, cell_centres, road_cells,
                              solve_height)

MIN_OVERLAP_CELLS = 100    # below this two pieces don't really share road
# Trust a piece's own fitted scale only if its GPS fit is this good.
# Scale is the worst-determined part of a similarity fit, so a piece with
# a poor fit reports a badly wrong one -- measured here, every piece with
# a residual under 0.2 m agreed on 1.23-1.30, while the only two outliers
# (1.43 and 1.70) came from the only two pieces with residuals above 1 m.
GOOD_FIT_RESIDUAL_M = 0.25
# How far a piece's cameras must be spread before GPS can be said to have
# pinned its heading. Measured, not guessed: a piece with six cameras
# bunched inside 9 m is no better anchored than one with two.
STRONG_TRACK_SPAN_M = 15.0
# A panorama sees usefully to about 10-12 m, so two pieces whose nearest
# cameras are further apart than this never really looked at the same
# ground. They can still share raster cells at long range, but that is
# far-field depth, which is the least reliable part of a reconstruction --
# aligning on it produces confident nonsense.
MAX_CAMERA_GAP_M = 16.0
# GPS already placed each piece to within its own fit residual, so the
# road step is only ever closing a seam of about that size. Anything far
# larger is the road step latching onto the wrong stretch of road.
MOVE_BUDGET_FACTOR = 2.5
MIN_MOVE_BUDGET_M = 2.0


def load_pieces(directory):
    """Each piece's GPS fit + point cloud, all in the shared GLOBAL_ORIGIN
    frame. A single-node piece cannot fit its own rotation/scale, so it
    borrows them from the nearest multi-node piece -- its heading is then
    recovered properly by road alignment, which is the whole point.

    Every piece is scaled by ONE shared factor, not by its own. DA3
    reconstructs all of them in the same units, so a genuine per-piece
    scale difference should not exist; what the per-piece fits actually
    measure is how noisy each piece's GPS was. Letting each piece keep
    its own turns that noise into geometry -- pieces come out different
    sizes, and no amount of moving them will ever make them meet.

    The shared factor is taken from the pieces whose GPS fit is good,
    and it is applied to HEIGHT as well as to x and z. It converts DA3
    units to metres, and the height is in DA3 units like everything
    else; scaling only two axes of three leaves every piece squashed
    vertically, which quietly corrupts every slope and height in the
    scene."""
    metas = {}
    for name in sorted(os.listdir(directory)):
        if name.endswith("_meta.json"):
            i = int(name.split("_")[1])
            metas[i] = json.load(open(os.path.join(directory, name)))

    fits, singles = {}, []
    for i, meta in metas.items():
        keys = list(meta)
        cams = np.array([real_en(meta[k]["lat"], meta[k]["lon"]) for k in keys])
        if len(keys) == 1:
            singles.append(i)
            fits[i] = {"cams": cams, "n": 1, "resid": None,
                       "da3": np.array(meta[keys[0]]["position"])[[0, 2]]}
            continue
        nodes = [{"da3_xz": [meta[k]["position"][0], meta[k]["position"][2]],
                  "lat": meta[k]["lat"], "lon": meta[k]["lon"]} for k in keys]
        R, scale, t, _, _, res = fit_nodes(nodes)
        fits[i] = {"R": R, "scale": scale, "t": t, "cams": cams,
                   "n": len(keys), "resid": float(np.median(res)),
                   "src_xz": np.array([n["da3_xz"] for n in nodes])}

    multi = [i for i in fits if fits[i]["n"] > 1]
    for i in singles:
        c = fits[i]["cams"][0]
        near = min(multi, key=lambda j: np.linalg.norm(fits[j]["cams"] - c, axis=1).min())
        R, scale = fits[near]["R"], fits[near]["scale"]
        fits[i].update(R=R, scale=scale, t=c - scale * (R @ fits[i]["da3"]),
                       borrowed_from=near)

    scale = global_scale(fits)
    clouds = {}
    for i in fits:
        # keep each piece's own rotation, but re-solve its offset for the
        # shared scale, so its cameras still land on their GPS positions
        f = fits[i]
        f["own_scale"], f["scale"] = f["scale"], scale
        src = f["da3"][None, :] if f["n"] == 1 else f["src_xz"]
        f["t"] = (f["cams"] - scale * (src @ f["R"].T)).mean(0)

        pts, cols = _read_ply_points(os.path.join(directory, f"piece_{i}.ply"))
        xz = pts[:, [0, 2]] @ f["R"].T * scale + f["t"]
        clouds[i] = (xz, pts[:, 1] * scale, cols)
    return fits, clouds


def global_scale(fits):
    """One DA3-units-to-metres factor for the whole scene, averaged over
    the pieces whose GPS fit was good enough to have measured it."""
    good = [f["scale"] for f in fits.values()
            if f["resid"] is not None and f["resid"] <= GOOD_FIT_RESIDUAL_M]
    if not good:
        good = [f["scale"] for f in fits.values() if f["resid"] is not None]
    return float(np.median(good))


def track_span(cams):
    """How far a piece's cameras are spread out. This, not the node
    count, is what says whether GPS could pin the piece's heading: six
    cameras bunched within 9 m fix a direction no better than two."""
    if len(cams) < 2:
        return 0.0
    return float(np.linalg.norm(cams[:, None, :] - cams[None, :, :], axis=2).max())


def camera_gap(cams_a, cams_b):
    """Distance between the two pieces' nearest cameras. A panorama sees
    usefully to about 10-12 m, so this predicts how much two pieces can
    possibly share: ~10 m apart and they overlap richly, ~20 m apart and
    they barely see the same ground however good the alignment is."""
    return float(cKDTree(cams_a).query(cams_b)[0].min())


def strength(fits, i):
    f = fits[i]
    return (track_span(f["cams"]), f["n"], -(f["resid"] if f["resid"] is not None else 99))


def alignment_order(fits, masks):
    """Best-anchored piece first, then outward.

    A piece is only ready once something already placed is BOTH sharing
    road with it and close enough to have seen the same ground -- the
    same test the reference picker applies. Ordering on shared area
    alone brings pieces up before their real neighbour exists, and they
    then get placed with no reference at all.

    Detached pieces are deferred to the very end rather than dropped in
    where they fall, in case placing others first connects them."""
    remaining = set(fits)
    first = max(remaining, key=lambda i: strength(fits, i))
    order, placed = [first], [first]
    remaining.remove(first)
    while remaining:
        ready = [i for i in remaining
                 if any((masks[i] & masks[j]).sum() >= MIN_OVERLAP_CELLS
                        and camera_gap(fits[j]["cams"], fits[i]["cams"]) <= MAX_CAMERA_GAP_M
                        for j in placed)]
        if ready:
            nxt = max(ready, key=lambda i: strength(fits, i))
            order.append((nxt, True))
            placed.append(nxt)
        else:
            nxt = max(remaining, key=lambda i: strength(fits, i))
            order.append((nxt, False))     # nothing can reach it
            placed.append(nxt)
        remaining.remove(nxt)
    return order


def pick_reference(i, fits, masks, placed_masks):
    """Which already-placed piece to align piece i against.

    Sharing road area is not enough on its own -- two pieces can share a
    little ground simply because both are near a junction, while a piece
    whose camera stood almost where this one's stood is looking at the
    very same scene. So among the placed pieces that share enough road,
    prefer the one whose cameras came closest. Choosing purely by shared
    area (or worse, by whatever happened to be placed first) picks a
    distant piece over a near one and produces a fit with nothing real
    behind it."""
    cands = [j for j in placed_masks
             if (masks[i] & placed_masks[j]).sum() >= MIN_OVERLAP_CELLS
             and camera_gap(fits[j]["cams"], fits[i]["cams"]) <= MAX_CAMERA_GAP_M]
    if not cands:
        return None, 0, None
    j = min(cands, key=lambda j: camera_gap(fits[j]["cams"], fits[i]["cams"]))
    return j, int((masks[i] & placed_masks[j]).sum()), camera_gap(fits[j]["cams"], fits[i]["cams"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/gap3_joined")
    ap.add_argument("--out", default=os.path.expanduser("~/Downloads/road_aligned.ply"))
    ap.add_argument("--cell", type=float, default=CELL)
    ap.add_argument("--margin", type=float, default=40.0)
    ap.add_argument("--pieces", default=None,
                    help="comma-separated piece ids to run on, e.g. 3,9,6,11,5. "
                         "Everything else is excluded entirely, so the run is "
                         "identical to a full run that simply had no other "
                         "pieces in the directory.")
    ap.add_argument("--min-nodes", type=int, default=1,
                    help="drop pieces with fewer nodes than this, entirely -- "
                         "they then cannot act as a reference or affect the "
                         "ordering either. Use 2 to exclude single-node pieces.")
    args = ap.parse_args()

    fits, clouds = load_pieces(args.dir)
    if args.pieces:
        keep = {int(x) for x in args.pieces.split(",")}
        for i in [i for i in fits if i not in keep]:
            del fits[i], clouds[i]
        print(f"restricted to {len(fits)} piece(s): "
              f"{', '.join(f'piece_{i}' for i in sorted(fits))}\n")
    if args.min_nodes > 1:
        dropped = sorted(i for i in fits if fits[i]["n"] < args.min_nodes)
        for i in dropped:
            del fits[i], clouds[i]
        print(f"dropped {len(dropped)} piece(s) with < {args.min_nodes} node(s): "
              f"{', '.join(f'piece_{i}' for i in dropped)}\n")
    allc = np.vstack([f["cams"] for f in fits.values()])
    bounds = (allc[:, 0].min() - args.margin, allc[:, 0].max() + args.margin,
              allc[:, 1].min() - args.margin, allc[:, 1].max() + args.margin)
    print(f"{len(fits)} piece(s), bounds {bounds[1]-bounds[0]:.0f} x "
          f"{bounds[3]-bounds[2]:.0f} m at {args.cell} m/cell\n")

    masks = {i: road_cells(*clouds[i], bounds, args.cell)[0] for i in clouds}
    for i in sorted(fits):
        f = fits[i]
        r = "n/a" if f["resid"] is None else f"{f['resid']:.2f}m"
        b = f" (scale borrowed from piece_{f['borrowed_from']})" if "borrowed_from" in f else ""
        print(f"  piece_{i:<3} {f['n']} node(s), GPS residual {r:>6}, "
              f"{int(masks[i].sum())} road cell(s){b}")

    order = alignment_order(fits, masks)
    ref = order[0]
    print(f"\nreference: piece_{ref} (best GPS fit of the strongest pieces)")

    placed = {ref: clouds[ref]}
    placed_masks = {ref: masks[ref].copy()}
    fixes = {}
    for i, touching in order[1:]:
        print(f"\n--- piece_{i} ({fits[i]['n']} node(s)) ---")
        if not touching:
            print("  no road overlap with anything placed yet -- left on GPS alone")
            placed[i] = clouds[i]
            placed_masks[i] = masks[i]
            continue
        # Align to the single best-overlapping neighbour, NOT to everything
        # placed so far. Against the union, a piece with only a small
        # genuine overlap will happily slide sideways onto some other
        # road entirely -- the corrections run away to metres.
        nbr, share, gap = pick_reference(i, fits, masks, placed_masks)
        if nbr is None:
            print("  nothing placed shares enough road with it -- left on GPS alone")
            placed[i] = clouds[i]
            placed_masks[i] = masks[i]
            continue
        span = track_span(fits[i]["cams"])
        strong = span >= STRONG_TRACK_SPAN_M
        print(f"  reference: piece_{nbr} ({share} shared road cell(s), "
              f"cameras {gap:.1f} m apart)")
        print(f"  this piece: {fits[i]['n']} node(s) spread over {span:.1f} m -> "
              f"{'STRONG, heading and sideways stay with GPS' if strong else 'weak, free to be moved'}")
        fix, diag = align_piece(clouds[i], placed[nbr], fits[i]["cams"],
                                bounds, cell=args.cell)
        if strong:
            # GPS pinned this one. A road-based rotation or sideways shift
            # here would be overruling several GPS points with one noisy
            # measurement; height is the one thing GPS never constrained.
            if fix.d_heading or fix.d_across:
                print(f"  strong piece: dropping heading {fix.d_heading:+.2f} deg and "
                      f"sideways {fix.d_across:+.2f} m, keeping height only")
            fix.d_heading, fix.d_across = 0.0, 0.0
        for n in fix.notes:
            print(f"  {n}")
        moved = np.linalg.norm(
            fix.apply_xz(fits[i]["cams"]) - fits[i]["cams"], axis=1).mean()
        budget = max(MIN_MOVE_BUDGET_M, MOVE_BUDGET_FACTOR * (fits[i]["resid"] or 1.0))
        if fix.d_heading == 0.0 and moved > budget:
            # GPS put this piece within `resid` of where it belongs; a
            # sideways shift several times larger than that is the road
            # step latching onto the wrong stretch, not a real seam.
            print(f"  !! sideways shift {moved:.2f} m exceeds its GPS budget "
                  f"{budget:.2f} m -- dropped, keeping only the height fix")
            fix.d_across = 0.0
        print(f"  -> cameras move "
              f"{np.linalg.norm(fix.apply_xz(fits[i]['cams']) - fits[i]['cams'], axis=1).mean():.2f}"
              f" m from GPS  {fix}")
        xz, y = fix.apply(clouds[i][0], clouds[i][1])
        placed[i] = (xz, y, clouds[i][2])
        placed_masks[i] = road_cells(xz, y, clouds[i][2], bounds, args.cell)[0]
        fixes[i] = fix

    pts = np.vstack([np.column_stack([p[0][:, 0], p[1], p[0][:, 1]]) for p in placed.values()])
    cols = np.vstack([p[2] for p in placed.values()])
    write_ply(args.out, pts, cols)
    print(f"\nwrote {args.out} ({len(pts)} points, {len(placed)} piece(s))")


if __name__ == "__main__":
    main()
