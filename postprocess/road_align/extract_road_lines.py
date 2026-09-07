"""The three lines that describe a piece's road: left kerb, centre, right.

These are what pieces are aligned by. Points are the wrong thing to align
on -- a road surface is featureless and slides freely along itself -- and
a raw kerb is little better, because it comes back as a scatter of
boundary cells broken wherever the panorama ran out of range.

So each piece is reduced to three curves:

    CENTRE  the medial axis of the road, from `road.centreline`. It is the
            average of both edges, so errors in the mask partly cancel,
            and it survives one side of the road being unobserved.
    KERBS   the road boundary, split by which side of the centre it falls
            on and fitted as one curve per side.

Splitting by the centre is what makes "left" and "right" mean anything.
Fitting each side in along-the-road order, rather than by connectivity,
is what lets one curve span the breaks in the boundary instead of
fragmenting at each one.

The kerb fits are robust, because a road is not always just a road. Where
a side street joins, its boundary sits on one side of the centreline like
everything else and gets swept into that kerb, dragging the curve out
into a bulge. Cells that end up far from the curve the rest agree on are
dropped and the curve refitted -- so a junction goes unrepresented rather
than distorting the road it joins.
"""
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

from postprocess.road_align.kerb import kerb_curves
from postprocess.road_align.road import CELL, centreline, road_cells

MAX_OFFSET_M = 12.0      # a kerb is not this far from the middle of its road
SMOOTH_PER_PT = 3.0
STEP_M = 0.25
MIN_SIDE_PTS = 12
ROBUST_ROUNDS = 3
REJECT_M = 1.5
BIN_M = 1.0


def road_lines(piece, bounds, cams, frame, cell=CELL):
    """[left kerb, centre, right kerb] for one piece, or None.

    `frame` is a camera_route.RouteFrame: it fixes which way is "along"
    and therefore which kerb is the left one, consistently across pieces.
    """
    from postprocess.road_align.feature_icp import extract_features

    xz, y, cols = piece
    mask, _, _ = road_cells(xz, y, cols, bounds, cell, cams=cams)
    centres = kerb_curves(centreline(mask, bounds, cell))
    _, kerb_xyz, _ = extract_features(piece, bounds, cams=cams)
    if not centres or len(kerb_xyz) < 8:
        return None

    centre = frame.orient(centres[0])
    sides = kerb_sides(kerb_xyz[:, [0, 2]], centre)
    if len(sides) != 2:
        return None
    # left first, right second, in the route's own frame
    left, right = sorted(sides, key=lambda s: -frame.project(s)[1].mean())
    return [left, centre, right]


def kerb_sides(kerb_xz, centre_curve, max_offset_m=MAX_OFFSET_M, **kw):
    """[left_curve, right_curve] -- whichever of the two could be fitted."""
    if len(kerb_xz) < 8 or len(centre_curve) < 4:
        return []
    side, along, offset, inside = split_sides(kerb_xz, centre_curve)
    keep = inside & (offset <= max_offset_m)
    out = []
    for s in (-1.0, 1.0):
        m = keep & (side == s)
        c = fit_side(kerb_xz[m], along[m], **kw)
        if c is not None:
            out.append(c)
    return out


def split_sides(kerb_xz, centre_curve):
    """Label each kerb cell left or right, and place it along the road."""
    d = np.diff(centre_curve, axis=0)
    seg_mid = centre_curve[:-1] + d / 2
    s_along = np.r_[0.0, np.cumsum(np.linalg.norm(d, axis=1))][:-1]
    tang = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)

    _, idx = cKDTree(seg_mid).query(kerb_xz)
    rel = kerb_xz - seg_mid[idx]
    t = tang[idx]
    side = np.sign(t[:, 0] * rel[:, 1] - t[:, 1] * rel[:, 0])
    along = s_along[idx] + np.einsum("ij,ij->i", rel, t)
    offset = np.abs(t[:, 0] * rel[:, 1] - t[:, 1] * rel[:, 0])
    # A cell past either end of the centreline still finds a nearest
    # segment -- the last one -- so a whole cluster beyond the end collapses
    # onto a single along-value. The bin median then lands somewhere between
    # them and the curve bends inward to reach it. Such cells lie outside the
    # stretch of road the centreline describes, so they are dropped rather
    # than clamped onto its end.
    total = s_along[-1] + float(np.linalg.norm(d[-1]))
    inside = (along >= -1.0) & (along <= total + 1.0)
    return side, along, offset, inside


def fit_side(pts, along, smooth_per_pt=SMOOTH_PER_PT, step=STEP_M,
             min_pts=MIN_SIDE_PTS, robust_rounds=ROBUST_ROUNDS,
             reject_m=REJECT_M):
    """One smooth curve through the cells of a single kerb."""
    if len(pts) < min_pts:
        return None
    order = np.argsort(along)
    p, a = pts[order], along[order]

    keep = np.ones(len(p), bool)
    q = None
    for _ in range(max(robust_rounds, 1)):
        q = _bin_median(p[keep], a[keep])
        if q is None or len(q) < 4:
            return None
        d = cKDTree(q).query(p)[0]
        nk = d <= max(reject_m, 2.0 * np.median(d[keep]))
        if nk.sum() < min_pts or (nk == keep).all():
            break
        keep = nk
    if q is None or len(q) < 4:
        return None
    length = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
    tck, _ = splprep([q[:, 0], q[:, 1]], s=smooth_per_pt * len(q),
                     k=min(3, len(q) - 1))
    x, y = splev(np.linspace(0, 1, max(int(length / step), 20)), tck)
    return np.column_stack([x, y])


def _bin_median(p, a, width=BIN_M):
    """Median position in each metre of distance along the road.

    A median rather than a mean, so a handful of cells belonging to a side
    street cannot pull the bin they land in.
    """
    if len(p) < 4:
        return None
    edges = np.arange(a.min(), a.max() + width, width)
    if len(edges) < 4:
        return None
    out = [np.median(p[(a >= lo) & (a < hi)], axis=0)
           for lo, hi in zip(edges[:-1], edges[1:])
           if ((a >= lo) & (a < hi)).any()]
    q = np.array(out)
    if len(q) < 2:
        return None
    return q[np.r_[True, np.linalg.norm(np.diff(q, axis=0), axis=1) > 1e-6]]
