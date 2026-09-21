"""Slide pieces across the road, in the one direction GPS cannot see.

The centre line is built from the panoramas' own positions, so it sits
wherever the vehicle drove -- down whichever lane it happened to use.
Seating a piece on that line fixes where it sits ALONG the road and
removes gross error across it, but a lane-sized offset survives: two runs
down the same street in different lanes both satisfy GPS equally well.

The ground under two pieces is the same physical road surface, so where
they see the same stretch their surfaces must line up. Matching them
measures the leftover offset directly, which nothing in GPS can.

Pairs are never chained. Sliding B onto A and then C onto B carries
whatever each match got wrong into every piece after it. Instead each
overlapping pair contributes one observation and every offset is solved
at once, constrained to sum to zero -- so the group stays where GPS put
it on average and only their positions relative to each other move.
"""
import itertools

import numpy as np

MAX_SHIFT_M = 5.0        # about a lane either way; beyond this a match is wrong
BIN_M = 0.25
MIN_OVERLAP_M = 10.0     # shorter than this and the profiles describe nothing
MIN_POINTS = 400


def _profile(left, lo, hi):
    """How much road sits at each distance across, as a density."""
    edges = np.arange(lo, hi + BIN_M, BIN_M)
    counts, _ = np.histogram(left, bins=edges)
    return counts / max(counts.max(), 1)


def offset(a, b, frame):
    """How far b must move to its left to sit on a's road surface.

    None when the two do not see enough of the same stretch to say.
    """
    a_along, a_left = frame.project(a[:, [0, 2]])
    b_along, b_left = frame.project(b[:, [0, 2]])

    lo, hi = max(a_along.min(), b_along.min()), min(a_along.max(), b_along.max())
    if hi - lo < MIN_OVERLAP_M:
        return None
    a_in = (a_along >= lo) & (a_along <= hi)
    b_in = (b_along >= lo) & (b_along <= hi)
    if a_in.sum() < MIN_POINTS or b_in.sum() < MIN_POINTS:
        return None

    a_left, b_left = a_left[a_in], b_left[b_in]
    span = max(np.abs(np.r_[a_left, b_left]).max(), 1.0) + MAX_SHIFT_M
    pa, pb = _profile(a_left, -span, span), _profile(b_left, -span, span)

    # slide b's profile against a's and keep the shift that overlaps most
    # rolling b's profile down by k bins is b sitting k bins further left,
    # so the winning lag is how far left of a it already is -- and the move
    # that closes the gap is the same distance back to the right
    lags = np.arange(-int(MAX_SHIFT_M / BIN_M), int(MAX_SHIFT_M / BIN_M) + 1)
    score = [np.minimum(pa, np.roll(pb, -k)).sum() for k in lags]
    return -float(lags[int(np.argmax(score))] * BIN_M)


def solve(shifts, ids):
    """{piece: metres to move right}, from pairwise observations.

    One unknown per piece and one equation per pair, plus a final equation
    holding the offsets to zero on average. Without it only differences are
    determined and the whole group could drift off together.
    """
    index = {i: n for n, i in enumerate(ids)}
    rows, obs = [], []
    for (i, j), d in shifts.items():
        row = np.zeros(len(ids))
        row[index[i]], row[index[j]] = -1.0, 1.0
        rows.append(row)
        obs.append(d)
    rows.append(np.ones(len(ids)))
    obs.append(0.0)
    answer, *_ = np.linalg.lstsq(np.array(rows), np.array(obs), rcond=None)
    return {i: float(answer[index[i]]) for i in ids}


def align(road_pts, frames, on, log=print):
    """{piece: 4x4} sliding each piece across the road onto its neighbours."""
    ids = [i for i in sorted(road_pts) if len(road_pts[i])]
    shifts = {}
    for i, j in itertools.combinations(ids, 2):
        shared = sorted(set(on.get(i, ())) & set(on.get(j, ())))
        best = None
        for r in shared:
            d = offset(road_pts[i], road_pts[j], frames[r])
            if d is not None and (best is None or abs(d) < abs(best[1])):
                best = (r, d)
        if best is not None:
            shifts[(i, j)] = best[1]
            log(f"  piece_{i} / piece_{j} on road{best[0]}: {best[1]:+.2f} m apart")

    if not shifts:
        log("  no pair sees enough shared road to measure")
        return {i: np.eye(4) for i in road_pts}

    across = solve(shifts, ids)
    out = {}
    for i in road_pts:
        T = np.eye(4)
        if i in across:
            # move along whatever road the piece was matched on
            r = next(iter(on.get(i, ())), None)
            if r is not None:
                here = road_pts[i][:, [0, 2]].mean(0)
                T[[0, 2], 3] = frames[r].normal(here)[0] * across[i]
        out[i] = T
    return out
