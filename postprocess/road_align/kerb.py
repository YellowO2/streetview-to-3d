"""Kerbs as CURVES: trace, stitch, fit.

`road_cells` gives the road as a set of cells and `feature_icp` gives its
boundary as a set of points, but a set of points has no direction and no
shape. Continuity-based alignment needs the kerb as an ordered curve with
a tangent everywhere, so this builds one.

The pipeline, and what each step is for:

  trace   -- put the boundary cells in order by walking along them. A
             component can branch (a junction mouth, a dropped kerb), so
             every branch is peeled off, not just the longest path.
  stitch  -- rejoin what the extraction broke. Roughly a third of a road's
             boundary is discarded for bordering unobserved space rather
             than a real kerb, which chops one kerb into many fragments.
  fit     -- a spline through each ordered run, so the kerb has a defined
             tangent even across the gaps that were bridged.

Two filters that are load-bearing rather than cosmetic: a chain that
merely retraces a longer one is dropped (the boundary comes out two cells
thick in places, and stitching a duplicate back on makes a curve double
over itself), and so is a chain assembled mostly out of bridged gap
(2.8 m of real kerb stretched over 9.5 m draws a phantom curve alongside
the real one).
"""
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

LINK_RADIUS_M = 0.45      # links 8-neighbours on the 0.25 m extraction grid
MIN_CHAIN_M = 0.5         # fragments below this still stitch into a curve, so
                          # filtering them here is pure loss
MAX_GAP_M = 6.0           # how far a kerb may vanish and still be the same kerb
MAX_TURN_DEG = 40.0       # how much it may bend across that gap
MAX_LATERAL_M = 1.5       # how far off one kerb's line the other may sit
MAX_BRIDGED_FRAC = 0.5    # a curve must be more observed kerb than bridged gap
DUP_TOL_M = 0.8
DUP_FRAC = 0.7
SMOOTH_PER_PT = 0.05
MIN_CURVE_M = 3.0
TANGENT_PTS = 5
STAIRCASE_M = 1.0       # walk a traced chain at roughly this spacing first
MIN_RADIUS_M = 8.0      # a kerb turns no tighter than this
MAX_SMOOTH_STEPS = 7


def kerb_curves(pts, step=0.25):
    """Unordered kerb points -> smooth continuous curves."""
    chains = stitch([c for c, _ in trace(pts)])
    out = []
    for c in chains:
        if np.linalg.norm(np.diff(c, axis=0), axis=1).sum() >= MIN_CURVE_M:
            out.append(fit(c, step))
    return out


def trace(pts):
    """[(ordered_points, length_m), ...], longest first."""
    if len(pts) < 3:
        return []
    pairs = cKDTree(pts).query_pairs(LINK_RADIUS_M, output_type="ndarray")
    if len(pairs) == 0:
        return []
    w = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1)
    n = len(pts)
    g = coo_matrix((np.r_[w, w], (np.r_[pairs[:, 0], pairs[:, 1]],
                                  np.r_[pairs[:, 1], pairs[:, 0]])),
                   shape=(n, n)).tocsr()
    ncomp, label = connected_components(g, directed=False)

    chains = []
    todo = [np.flatnonzero(label == c) for c in range(ncomp)]
    while todo:
        idx = todo.pop()
        if len(idx) < 3:
            continue
        sub = g[idx][:, idx]
        d0 = dijkstra(sub, indices=0)
        u = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
        du, pred = dijkstra(sub, indices=u, return_predecessors=True)
        v = int(np.argmax(np.where(np.isfinite(du), du, -1)))
        path, cur = [], v
        while cur >= 0:
            path.append(cur)
            if cur == u:
                break
            cur = pred[cur]
        if len(path) < 3:
            continue
        local = np.array(path[::-1])
        p = pts[idx[local]]
        length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        if length >= MIN_CHAIN_M:
            chains.append((p, length))
        rest = np.setdiff1d(np.arange(len(idx)), local)
        if len(rest) >= 3:
            nc, lab = connected_components(sub[rest][:, rest], directed=False)
            for c in range(nc):
                todo.append(idx[rest[lab == c]])
    return _drop_duplicates(sorted(chains, key=lambda t: -t[1]))


def _drop_duplicates(chains):
    kept = []
    for p, length in chains:
        if any((cKDTree(q).query(p)[0] < 0.6).mean() >= 0.5 for q, _ in kept):
            continue
        kept.append((p, length))
    return kept


def tangents(p):
    """Outward unit direction at each end of an ordered chain."""
    k = min(TANGENT_PTS, len(p) - 1)
    a, b = p[0] - p[k], p[-1] - p[-1 - k]
    return a / (np.linalg.norm(a) + 1e-9), b / (np.linalg.norm(b) + 1e-9)


def stitch(chains):
    """Join chains that are the same kerb continuing.

    The test is the TURN between the two curve directions plus how far one
    end sits off the other's line, NOT whether the ends point at each
    other. Two stretches of one kerb either side of a short gap are often
    offset sideways, so the little connecting vector points off at an
    angle and an "aimed at each other" test rejects them -- it rejected a
    join whose curves differed by 4 degrees. Turn angle stays meaningful
    however short the gap is.
    """
    chains = [np.asarray(c) for c in chains]
    while True:
        best = None
        for i in range(len(chains)):
            for j in range(len(chains)):
                if i == j:
                    continue
                ti0, ti1 = tangents(chains[i])
                tj0, tj1 = tangents(chains[j])
                for pi, ti, fi in ((chains[i][-1], ti1, False),
                                   (chains[i][0], ti0, True)):
                    for pj, tj, fj in ((chains[j][0], tj0, False),
                                       (chains[j][-1], tj1, True)):
                        d = pj - pi
                        gap = float(np.linalg.norm(d))
                        if not 1e-9 < gap <= MAX_GAP_M:
                            continue
                        turn = np.degrees(np.arccos(np.clip(float(-ti @ tj), -1, 1)))
                        lateral = abs(float(ti[0] * d[1] - ti[1] * d[0]))
                        if turn > MAX_TURN_DEG or lateral > MAX_LATERAL_M:
                            continue
                        if float(ti @ d) <= 0:
                            continue
                        cost = gap + 0.5 * turn + 2.0 * lateral
                        if best is None or cost < best[0]:
                            best = (cost, i, j, fi, fj)
        if best is None:
            break
        _, i, j, fi, fj = best
        a = chains[i][::-1] if fi else chains[i]
        b = chains[j][::-1] if fj else chains[j]
        for k in sorted((i, j), reverse=True):
            chains.pop(k)
        chains.append(np.vstack([a, b]))

    chains.sort(key=lambda c: -np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
    kept = []
    for c in chains:
        seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
        if seg.sum() > 0 and seg[seg > 0.5].sum() / seg.sum() > MAX_BRIDGED_FRAC:
            continue
        if any((cKDTree(q).query(c)[0] < DUP_TOL_M).mean() >= DUP_FRAC for q in kept):
            continue
        kept.append(c)
    return kept


def _resample(p, step):
    """Walk a chain at a fixed spacing, averaging the cells passed through."""
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if d[-1] < 2 * step:
        return p
    out = []
    for t in np.arange(0, d[-1] + 1e-9, step):
        near = np.abs(d - t) <= step
        out.append(p[near].mean(0) if near.any() else p[np.argmin(np.abs(d - t))])
    out = np.array(out)
    return out[np.r_[True, np.linalg.norm(np.diff(out, axis=0), axis=1) > 1e-6]]


def min_radius(c):
    """Tightest turn anywhere along a curve, in metres."""
    if len(c) < 5:
        return np.inf
    d1 = np.gradient(c, axis=0)
    d2 = np.gradient(d1, axis=0)
    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5
    kappa = np.divide(num, den, out=np.zeros(len(c)), where=den > 1e-12)
    k = np.percentile(kappa, 98)
    return np.inf if k <= 1e-9 else 1.0 / k


def fit(chain, step=0.25):
    """Ordered points -> a smooth curve a road edge could actually follow.

    Two things stop the result looking pitted. Kerb cells sit on a grid, so
    a traced chain steps horizontally, vertically or diagonally and turns
    45 degrees at almost every point; at 0.25 m per step that reads as a
    0.3 m turning radius. Walking the chain at about a metre first averages
    that staircase away before the spline can follow it.

    Then smoothing is raised until the curve stops turning tighter than a
    kerb physically does. Fitting at one fixed smoothness instead lets the
    spline chase every wobble in the boundary, and those wobbles are
    extraction noise, not road.
    """
    p = np.asarray(chain, float)
    p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]]
    if len(p) < 4:
        return p
    p = _resample(p, STAIRCASE_M)
    if len(p) < 4:
        return np.asarray(chain, float)
    length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    n = max(int(length / step), 12)
    smooth = SMOOTH_PER_PT * len(p)
    out = None
    for _ in range(MAX_SMOOTH_STEPS):
        tck, _ = splprep([p[:, 0], p[:, 1]], s=smooth, k=min(3, len(p) - 1))
        x, y = splev(np.linspace(0, 1, n), tck)
        out = np.column_stack([x, y])
        if min_radius(out) >= MIN_RADIUS_M:
            break
        smooth *= 4.0
    return out


def thin(c, step=0.5):
    """Fewer points, same shape -- searches evaluate curves thousands of times."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    return c[np.unique(np.searchsorted(s, np.arange(0, s[-1], step)))]
