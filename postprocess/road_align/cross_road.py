"""Slide pieces across the road, in the one direction GPS cannot see.

The centre line is built from the panoramas' own positions, so it sits
wherever the vehicle drove -- down whichever lane it happened to use.
Seating a piece on that line fixes where it sits ALONG the road and
removes gross error across it, but a lane-sized offset survives: two runs
down the same street in different lanes both satisfy GPS equally well.

The ground under two pieces is the same physical road surface, so where
they see the same stretch their surfaces must line up. Matching them
measures the leftover offset directly, which nothing in GPS can.

What gets matched is the ground's HEIGHT across the road, not how many
points fall at each distance. Density is decided by how far the camera
could see, so its profile is a soft hump whose edges are our own radius
setting -- widening that radius only widens the hump. Height is decided
by the road: a flat trough with a wall either side, its edges where the
carriageway actually ends.

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
MIN_BIN_POINTS = 20      # below this a bin's height is noise
MIN_SHARED_BINS = 20     # 5 m of profile the two actually have in common
GROUND_PCT = 8.0         # the low percentile in a bin is its ground


def _section(left, height, lo, hi):
    """Ground height at each distance across the road.

    Levelled to its own median: the pieces have not been seated vertically
    yet, so only the shape of the section means anything.
    """
    edges = np.arange(lo, hi + BIN_M, BIN_M)
    k = np.digitize(left, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(out)):
        m = k == b
        if m.sum() >= MIN_BIN_POINTS:
            out[b] = np.percentile(height[m], GROUND_PCT)
    return out - np.nanmedian(out) if np.isfinite(out).any() else out


def _shift(profile, k):
    """The profile as it would read if it sat k bins further left."""
    out = np.full_like(profile, np.nan)
    if k > 0:
        out[:-k] = profile[k:]
    elif k < 0:
        out[-k:] = profile[:k]
    else:
        out[:] = profile
    return out


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

    span = max(np.abs(np.r_[a_left[a_in], b_left[b_in]]).max(), 1.0) + MAX_SHIFT_M
    pa = _section(a_left[a_in], a[a_in, 1], -span, span)
    pb = _section(b_left[b_in], b[b_in, 1], -span, span)
    if not (np.isfinite(pa).any() and np.isfinite(pb).any()):
        return None

    # the lag that makes the two sections agree is how far left of a that b
    # already sits, so the move closing the gap is the same distance back
    lags = np.arange(-int(MAX_SHIFT_M / BIN_M), int(MAX_SHIFT_M / BIN_M) + 1)
    best, best_lag = -np.inf, None
    for k in lags:
        shifted = _shift(pb, k)
        m = np.isfinite(pa) & np.isfinite(shifted)
        if m.sum() < MIN_SHARED_BINS:
            continue
        # correlation, not a difference: most of a section is verge sitting
        # at the same height whatever the shift, and a plain difference lets
        # that agreeing majority drown out the trough that carries the answer
        u, v = pa[m] - pa[m].mean(), shifted[m] - shifted[m].mean()
        denom = np.sqrt((u * u).sum() * (v * v).sum())
        if denom <= 0:
            continue
        score = float((u * v).sum() / denom)
        if score > best:
            best, best_lag = score, k
    if best_lag is None:
        return None
    return -float(best_lag * BIN_M)


FLAT_M = 0.5             # within this of the lowest ground is carriageway


def road_centre(pts, frame):
    """Where the carriageway's middle sits, relative to the road line.

    Absolute, unlike a pairwise match: the piece measures this on its own,
    so a shift derived from it moves the piece onto the road rather than
    onto its neighbour. What pairwise matching can never fix is an error
    every piece shares, because only their differences are observed.

    The carriageway is the lowest flat run of the section -- verges,
    footways and frontage all stand above it.
    """
    along, left = frame.project(pts[:, [0, 2]])
    span = max(np.abs(left).max(), 1.0)
    section = _section(left, pts[:, 1], -span, span)
    if not np.isfinite(section).any():
        return None
    bins = np.arange(-span, span, BIN_M)[:len(section)] + BIN_M / 2
    # Y is down, so the road is the HIGHEST value in this section
    road = section >= np.nanmax(section) - FLAT_M
    if road.sum() < 4:
        return None
    return float(np.median(bins[road]))


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


def centre_all(road_pts, frames, on, log=print):
    """{piece: 4x4} putting each piece's carriageway on the road line.

    Absolute, so unlike pairwise matching it can move the whole group.
    """
    out = {}
    for i, pts in road_pts.items():
        T = np.eye(4)
        r = next(iter(on.get(i, ())), None)
        if len(pts) and r is not None:
            c = road_centre(pts, frames[r])
            if c is not None and abs(c) <= MAX_SHIFT_M:
                T[[0, 2], 3] = frames[r].normal(pts[:, [0, 2]].mean(0))[0] * -c
                log(f"  piece_{i:<4} carriageway centre {c:+.2f} m off the line")
        out[i] = T
    return out


def align(road_pts, frames, road_of, log=print):
    """{piece: 4x4} sliding each piece across the road onto its neighbours.

    road_of says which road each piece was actually driven along. Only two
    pieces on the SAME road can be compared: projecting a piece onto a
    street it was never on gives a section of something else entirely.
    """
    ids = [i for i in sorted(road_pts) if len(road_pts[i])]
    shifts = {}
    for i, j in itertools.combinations(ids, 2):
        r = road_of.get(i)
        if r is None or r != road_of.get(j):
            continue
        d = offset(road_pts[i], road_pts[j], frames[r])
        if d is not None:
            shifts[(i, j)] = d
            log(f"  piece_{i} / piece_{j} on road{r}: {d:+.2f} m apart")

    if not shifts:
        log("  no two pieces were driven along the same road -- nothing to match")
        return {i: np.eye(4) for i in road_pts}

    across = solve(shifts, [i for i in ids if i in
                            {k for pair in shifts for k in pair}])
    out = {}
    for i in road_pts:
        T = np.eye(4)
        r = road_of.get(i)
        if i in across and r is not None:
            here = road_pts[i][:, [0, 2]].mean(0)
            T[[0, 2], 3] = frames[r].normal(here)[0] * across[i]
        out[i] = T
    return out
