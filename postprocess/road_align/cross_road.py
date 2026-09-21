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
SLICE_M = 8.0            # the stretch each separate measurement covers
MAX_TURN_DEG = 5.0       # a piece's heading came from GPS; this only trims it
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


def _match(a_left, a_h, b_left, b_h):
    """How far b must move to its left to sit on a, for one stretch of road.

    Compares the two ground SECTIONS -- height across the road -- because
    that shape is set by the road. Point density is set by how far the
    camera could see, which is our own radius setting, so its edges say
    nothing about where the carriageway is.
    """
    if len(a_left) < MIN_POINTS or len(b_left) < MIN_POINTS:
        return None
    span = max(np.abs(np.r_[a_left, b_left]).max(), 1.0) + MAX_SHIFT_M
    pa = _section(a_left, a_h, -span, span)
    pb = _section(b_left, b_h, -span, span)
    if not (np.isfinite(pa).any() and np.isfinite(pb).any()):
        return None

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


def gap(a, b, frame):
    """(gap at the middle, how fast it opens, middle) for two pieces.

    Measured in slices along the road rather than once over the whole
    overlap, because a constant answer cannot describe two pieces that are
    TURNED relative to each other -- and they are: measured on a real pair
    the gap ran +0.25 m at one end of 22 m and -1.00 m at the other.

    A single number would then fit one end or the other and be wrong at
    both. A straight line through the slices separates the two causes: its
    level is how far apart the pieces sit, its slope is the angle between
    them.
    """
    a_along, a_left = frame.project(a[:, [0, 2]])
    b_along, b_left = frame.project(b[:, [0, 2]])
    lo, hi = max(a_along.min(), b_along.min()), min(a_along.max(), b_along.max())
    if hi - lo < MIN_OVERLAP_M:
        return None

    n = max(1, int((hi - lo) / SLICE_M))
    edges = np.linspace(lo, hi, n + 1)
    at, seen = [], []
    for u, v in zip(edges[:-1], edges[1:]):
        ai = (a_along >= u) & (a_along <= v)
        bi = (b_along >= u) & (b_along <= v)
        d = _match(a_left[ai], a[ai, 1], b_left[bi], b[bi, 1])
        if d is not None:
            at.append((u + v) / 2)
            seen.append(d)
    if not seen:
        return None

    mid = float(np.mean(at))
    if len(seen) == 1:
        return float(seen[0]), 0.0, mid
    slope, level = np.polyfit(np.array(at) - mid, seen, 1)
    return float(level), float(slope), mid


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


def solve(pairs, centres, ids):
    """({piece: sideways shift}, {piece: turn}) from pairwise gaps.

    A piece is described by where it sits across the road and how it is
    turned, so each carries two unknowns. Writing its offset at distance t
    along the road as s + w(t - p), a pair's measured gap `level + slope t`
    gives two equations: the slopes differ by the turn between them, and
    the levels by how far apart they sit.

    Two more equations hold both to zero on average. Without them only
    differences are observed and the whole group could drift or rotate off
    together -- and chaining pairs instead, B onto A then C onto B, carries
    every match's error into everything downstream.
    """
    n = len(ids)
    at = {i: k for k, i in enumerate(ids)}
    rows, obs = [], []
    for (i, j), (level, slope, mid) in pairs.items():
        turn = np.zeros(2 * n)
        turn[n + at[j]], turn[n + at[i]] = 1.0, -1.0
        rows.append(turn)
        obs.append(slope)

        side = np.zeros(2 * n)
        side[at[j]], side[at[i]] = 1.0, -1.0
        side[n + at[j]] = -(centres[j] - mid)
        side[n + at[i]] = (centres[i] - mid)
        rows.append(side)
        obs.append(level)

    for k in range(2):
        gauge = np.zeros(2 * n)
        gauge[k * n:(k + 1) * n] = 1.0
        rows.append(gauge)
        obs.append(0.0)

    x, *_ = np.linalg.lstsq(np.array(rows), np.array(obs), rcond=None)
    cap = np.radians(MAX_TURN_DEG)
    return ({i: float(x[at[i]]) for i in ids},
            {i: float(np.clip(x[n + at[i]], -cap, cap)) for i in ids})


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
    """{piece: 4x4} correcting where each piece sits across its road.

    Two corrections, because one cannot describe the error: a sideways
    slide for pieces that sit off to one side, and a small turn for pieces
    whose heading GPS got slightly wrong. Seating refuses to turn a piece
    with more than two nodes, trusting GPS for the heading, and this is
    where that trust gets checked against the road itself.

    road_of says which road each piece was driven along. Only two pieces on
    the SAME road can be compared: projecting a piece onto a street it was
    never on gives a section of something else entirely.
    """
    ids = [i for i in sorted(road_pts) if len(road_pts[i])]
    centres = {}
    for i in ids:
        r = road_of.get(i)
        if r is not None:
            along, _ = frames[r].project(road_pts[i][:, [0, 2]])
            centres[i] = float(np.mean(along))

    pairs = {}
    for i, j in itertools.combinations(ids, 2):
        r = road_of.get(i)
        if r is None or r != road_of.get(j) or i not in centres or j not in centres:
            continue
        g = gap(road_pts[i], road_pts[j], frames[r])
        if g is not None:
            pairs[(i, j)] = g
            log(f"  piece_{i} / piece_{j} on road{r}: {g[0]:+.2f} m apart, "
                f"opening {np.degrees(np.arctan(g[1])):+.1f} deg")

    if not pairs:
        log("  no two pieces were driven along the same road -- nothing to match")
        return {i: np.eye(4) for i in road_pts}

    involved = sorted({k for pair in pairs for k in pair})
    across, turn = solve(pairs, centres, involved)

    out = {}
    for i in road_pts:
        T = np.eye(4)
        r = road_of.get(i)
        if i in across and r is not None:
            pivot = road_pts[i][:, [0, 2]].mean(0)
            w = turn[i]
            c, sn = np.cos(w), np.sin(w)
            R = np.array([[c, -sn], [sn, c]])
            move = frames[r].normal(pivot)[0] * across[i]
            T[[0, 2], 0], T[[0, 2], 2] = R[:, 0], R[:, 1]
            # the piece turns about its own middle, so the translation column
            # carries a compensation term as well as the slide -- how far the
            # piece actually moves is `across`, not the length of that column
            T[[0, 2], 3] = pivot - R @ pivot + move
            log(f"  piece_{i:<4} slid {across[i]:+.2f} m, "
                f"turned {np.degrees(w):+.2f} deg")
        out[i] = T
    return out
