"""Align two pieces by making their kerbs one continuous road edge.

Pipeline position:  DA3 -> GPS -> ROAD ALIGNMENT (this, then the slope fit)

Two rules decide everything, and both are read off the data rather than
chosen:

  WHICH PIECE MOVES -- the one whose camera track is shorter. A piece
  built from a single panorama has no GPS heading at all (one point fixes
  where it is, never which way it faces), so turning it costs nothing,
  and turning it about its own camera moves that camera not at all. A
  piece with a long track has its heading pinned by GPS and turning it
  fights good data. Getting this backwards is expensive: aligning
  piece_3 (1 node) to piece_5 (4 nodes, 28 m track) needs 2.2 m of camera
  movement, while the same join done the other way round needs 16-21 m.

  HOW SMOOTH IS SMOOTH ENOUGH -- a single kerb is not perfectly smooth.
  Fit a curve to one piece's kerb alone and it leaves a residual; that is
  this data's noise floor, and asking a joined curve to beat it is
  fitting noise. So the floor is the target, and of every placement that
  reaches it, the one that disturbs GPS least wins.

The join test does NOT require the two kerbs to overlap or cross. Pieces
meeting at a junction often share no road at all -- one kerb ends where
the other begins -- and an earlier version that scored crossings threw
out more than 90% of the search space unexamined, including every
end-to-end arrangement.

Height and tilt are deliberately left alone here; fit them afterwards
with `slope_fit`, once the horizontal placement is right. Levelling first
matches whichever bits of road happen to lie near each other, which
before the horizontal fit are not the same bits.
"""
import numpy as np
from scipy.spatial import cKDTree

from alignment.kerb import kerb_curves, thin

SEAM_R = 9.0            # how much kerb either side of the meeting point to use
MIN_SEAM_PTS = 5        # each piece must contribute this much to count
MAX_SEP_M = 6.0         # beyond this the kerbs are simply not near each other
TURN_RANGE_DEG = 45.0
TURN_STEP_DEG = 0.5
SLIDE_RANGE_M = 6.0
SLIDE_STEP_M = 0.25
FLOOR_PCT = 75          # which percentile of the single-kerb residuals to target


def track_span(cams):
    if len(cams) < 2:
        return 0.0
    return float(np.max(np.linalg.norm(cams[:, None, :] - cams[None, :, :], axis=2)))


def seam_fit(A, B):
    """RMS residual of ONE quadratic through both kerbs where they meet.

    Returns (residual_m, separation_m) or None if they are not near enough
    to be the same road edge.
    """
    d, i = cKDTree(B).query(A)
    k = int(np.argmin(d))
    sep = float(d[k])
    if sep > MAX_SEP_M:
        return None
    mid = (A[k] + B[i[k]]) / 2
    na = A[np.linalg.norm(A - mid, axis=1) < SEAM_R]
    nb = B[np.linalg.norm(B - mid, axis=1) < SEAM_R]
    if len(na) < MIN_SEAM_PTS or len(nb) < MIN_SEAM_PTS:
        return None
    pool = np.vstack([na, nb])
    c = pool - pool.mean(0)
    # fit in the pooled points' own principal frame, so a kerb running in
    # any direction is handled without a vertical-line blow-up
    _, _, V = np.linalg.svd(c, full_matrices=False)
    u, v = c @ V[0], c @ V[1]
    resid = float(np.sqrt(np.mean((v - np.polyval(np.polyfit(u, v, 2), u)) ** 2)))
    return resid, sep


def noise_floor(curve_sets, rng=None, samples=60):
    """How smooth a single kerb is on its own -- the target, not a choice."""
    rng = rng or np.random.default_rng(0)
    out = []
    for curves in curve_sets:
        for c in curves:
            if len(c) < 12:
                continue
            for _ in range(samples):
                w = c[np.linalg.norm(c - c[rng.integers(0, len(c))], axis=1) < SEAM_R]
                if len(w) < 10:
                    continue
                d = w - w.mean(0)
                _, _, V = np.linalg.svd(d, full_matrices=False)
                u, v = d @ V[0], d @ V[1]
                out.append(np.sqrt(np.mean(
                    (v - np.polyval(np.polyfit(u, v, 2), u)) ** 2)))
    return float(np.percentile(out, FLOOR_PCT)) if out else None


def solve_pair(mover_kerb, fixed_kerb, pivot, floor):
    """Least-disturbing turn+slide of `mover_kerb` that reaches `floor`.

    All 2D, in a levelled frame. `pivot` should be the moving piece's own
    camera: a single-node piece then pays zero camera displacement for any
    amount of turn, which is correct, because nothing pinned its heading.
    """
    cm = [thin(c) for c in mover_kerb]
    cf = [thin(c) for c in fixed_kerb]
    if not cm or not cf:
        return None
    _, im, ifx = min(((float(cKDTree(q).query(p)[0].min()), i, j)
                      for i, p in enumerate(cm) for j, q in enumerate(cf)))
    M, F = cm[im], cf[ifx]

    best = None
    for deg in np.arange(-TURN_RANGE_DEG, TURN_RANGE_DEG + 1e-9, TURN_STEP_DEG):
        th = np.radians(deg)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        M0 = (M - pivot) @ R.T + pivot
        for dz in np.arange(-SLIDE_RANGE_M, SLIDE_RANGE_M + 1e-9, SLIDE_STEP_M):
            for dx in np.arange(-SLIDE_RANGE_M, SLIDE_RANGE_M + 1e-9, SLIDE_STEP_M):
                mv = float(np.hypot(dx, dz))
                if mv > SLIDE_RANGE_M or (best and mv >= best["move"]):
                    continue
                r = seam_fit(M0 + [dx, dz], F)
                if r and r[0] <= floor:
                    best = dict(move=mv, resid=r[0], sep=r[1],
                                deg=float(deg), dx=float(dx), dz=float(dz))
    return best


def slope_fit(road_mover, road_fixed, overlap_r=1.5, min_pts=200):
    """Tilt and height that seat the two road surfaces on each other.

    Call AFTER the horizontal placement, so the overlapping points are
    genuinely the same stretch of road. Returns a 4x4, or None if the two
    do not share enough road to measure a slope.
    """
    d, _ = cKDTree(road_fixed[:, [0, 2]]).query(road_mover[:, [0, 2]])
    sel = d < overlap_r
    d2, _ = cKDTree(road_mover[:, [0, 2]]).query(road_fixed[:, [0, 2]])
    sel2 = d2 < overlap_r
    if sel.sum() < min_pts or sel2.sum() < min_pts:
        return None

    def plane(p):
        ctr = p.mean(0)
        _, _, V = np.linalg.svd(p - ctr, full_matrices=False)
        return ctr, V[2] * (1 if V[2][1] >= 0 else -1)

    cm, nm = plane(road_mover[sel])
    cf, nf = plane(road_fixed[sel2])
    v = np.cross(nm, nf)
    s, c = np.linalg.norm(v), float(nm @ nf)
    if s < 1e-9:
        R = np.eye(3)
    else:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c) / s ** 2)
    # turn about the overlap centre, so the horizontal fit is not dragged
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cm - R @ cm
    T[1, 3] += cf[1] - (road_mover[sel] @ R.T + T[:3, 3])[:, 1].mean()
    return T


def plane_transform(Rplane, centre, pivot_xz, deg, dx, dz):
    """A 2D turn+slide in the levelled frame, as a world-space 4x4.

    Built from the frame change rather than derived by hand: the rotation
    is about a point hundreds of metres from the origin, and dropping the
    centre term silently turns a 6 m slide into a 105 m one.
    """
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    R2 = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    piv = np.array([pivot_xz[0], 0.0, pivot_xz[1]])
    sh = np.array([dx, 0.0, dz])

    def f(x):
        q = (x - centre) @ Rplane.T
        return ((q - piv) @ R2.T + piv + sh) @ Rplane + centre

    T = np.eye(4)
    T[:3, :3] = Rplane.T @ R2 @ Rplane
    T[:3, 3] = f(np.zeros((1, 3)))[0]
    return T
