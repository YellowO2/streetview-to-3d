"""Place every piece against ONE road, rather than against its neighbours.

Matching piece to piece makes each placement answer only to whoever
happens to be nearest. On gap3 that had piece_9 swinging round to meet
piece_6 while piece_5, 23 m away and outside its pairing radius, was
absent from its cost entirely. It also means a run only holds together if
consecutive pieces overlap: drop the single-node piece from the middle
and piece_5 had no partner left at all, so it simply froze at GPS.

Here the whole run is described by three curves -- left kerb, centre,
right kerb -- each running end to end, built by pooling the corresponding
line from every piece. Each piece is then turned and slid to sit on those
curves. Every piece answers to the same road, no piece needs a partner,
and there is no ordering to choose.

The curves are rebuilt from the placed pieces and the pieces refitted, a
few times over, so the road and the pieces agree by the end.

Kerbs count for roughly three times the centre. A kerb is observed
directly; the centre is inferred from the road mask's shape, and the
medial axis is unstable -- a 0.4% change in piece_6's mask moved its
centreline 3.3 m.
"""
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

BIN_M = 2.0              # along-route spacing the global curves are built at
SMOOTH_PER_PT = 2.0
MIN_BINS = 5
ROLE_WEIGHTS = (1.0, 0.35, 1.0)      # left kerb, centre, right kerb
TURN_RANGE_DEG = 40.0
TURN_STEP_DEG = 1.0
SLIDE_STEP_M = 0.5
SWEEPS = 4
SETTLED = 0.4


def drift_cap(n_nodes):
    """How far a piece's cameras may be dragged off GPS, by node count.

    What GPS knows about a piece depends on how many places it saw it
    from. A single node fixes a position and says nothing about heading,
    so such a piece must be free to TURN -- but GPS still knows where it
    is, so it is no freer to move than any other. Two nodes pin a
    direction but fit any similarity transform exactly, so their residual
    is always zero and means nothing. Three or more genuinely constrain a
    piece, and dragging one of those far is evidence the fit is wrong
    rather than that GPS was.
    """
    return {1: 5.0, 2: 5.0}.get(n_nodes, 3.0)


def fit_global_curve(pts, frame, step=0.5):
    """One smooth curve through points pooled from every piece.

    Built in along-route order rather than by connectivity, so it spans
    the gaps between pieces instead of stopping at them. Each bin takes a
    median, so a badly placed piece pulls its own bins a little rather
    than bending the whole curve.
    """
    if len(pts) < 20:
        return None
    along, _ = frame.project(pts)
    edges = np.arange(along.min(), along.max() + BIN_M, BIN_M)
    if len(edges) < MIN_BINS:
        return None
    binned = [np.median(pts[(along >= a) & (along < b)], axis=0)
              for a, b in zip(edges[:-1], edges[1:])
              if ((along >= a) & (along < b)).sum() >= 3]
    q = np.array(binned)
    if len(q) < 4:
        return None
    q = q[np.r_[True, np.linalg.norm(np.diff(q, axis=0), axis=1) > 1e-6]]
    if len(q) < 4:
        return None
    length = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
    tck, _ = splprep([q[:, 0], q[:, 1]], s=SMOOTH_PER_PT * len(q),
                     k=min(3, len(q) - 1))
    x, y = splev(np.linspace(0, 1, max(int(length / step), 20)), tck)
    return np.column_stack([x, y])


def _rot(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


class RoadFitter:
    """lines: {piece: [left, centre, right]}, cams: {piece: (N,2) XZ}."""

    def __init__(self, lines, cams, frame, weights=ROLE_WEIGHTS):
        self.ids = [i for i in lines if lines[i] is not None]
        self.lines = lines
        self.cams = cams
        self.frame = frame
        self.weights = weights
        self.pivot = {i: cams[i].mean(0) for i in self.ids}
        self.cap = {i: drift_cap(len(cams[i])) for i in self.ids}
        self.state = {i: np.zeros(3) for i in self.ids}     # deg, dx, dz

    def placed(self, i, st=None):
        deg, dx, dz = self.state[i] if st is None else st
        R = _rot(deg)
        p = self.pivot[i]
        return [(c - p) @ R.T + p + [dx, dz] for c in self.lines[i]]

    def drift(self, i, st):
        deg, dx, dz = st
        p = self.pivot[i]
        q = (self.cams[i] - p) @ _rot(deg).T + p + [dx, dz]
        return float(np.linalg.norm(q - self.cams[i], axis=1).mean())

    def global_curves(self):
        return [fit_global_curve(np.vstack([self.placed(i)[r] for i in self.ids]),
                                 self.frame) for r in range(3)]

    def best_for(self, i, trees):
        cap, pivot = self.cap[i], self.pivot[i]
        best = None
        for deg in np.arange(-TURN_RANGE_DEG, TURN_RANGE_DEG + 1e-9, TURN_STEP_DEG):
            R = _rot(deg)
            rot = [(c - pivot) @ R.T + pivot for c in self.lines[i]]
            camrot = (self.cams[i] - pivot) @ R.T + pivot
            for dz in np.arange(-cap, cap + 1e-9, SLIDE_STEP_M):
                for dx in np.arange(-cap, cap + 1e-9, SLIDE_STEP_M):
                    if np.hypot(dx, dz) > cap:
                        continue
                    if np.linalg.norm(camrot + [dx, dz] - self.cams[i],
                                      axis=1).mean() > cap:
                        continue
                    e = n = 0.0
                    for r, w in zip(range(3), self.weights):
                        if trees[r] is None:
                            continue
                        e += w * trees[r].query(rot[r] + [dx, dz])[0].mean()
                        n += w
                    if n == 0:
                        continue
                    v = e / n
                    if best is None or v < best[0]:
                        best = (v, np.array([deg, dx, dz]))
        return best[1] if best else self.state[i]

    def solve(self, sweeps=SWEEPS, log=print):
        for sweep in range(sweeps):
            curves = self.global_curves()
            trees = [cKDTree(c) if c is not None else None for c in curves]
            moved = 0.0
            for i in self.ids:
                st = self.best_for(i, trees)
                moved = max(moved, float(np.abs(st - self.state[i]).max()))
                self.state[i] = st
            if log:
                log(f"  sweep {sweep + 1}: largest change {moved:.2f}   " +
                    "  ".join(f"{i}:{self.state[i][0]:+.0f}d"
                              f"/{np.hypot(*self.state[i][1:]):.2f}m"
                              for i in self.ids))
            if moved < SETTLED:
                break
        return self.state

    def report(self):
        """{piece: (turn, slide, drift, [left, centre, right] distances)}."""
        trees = [cKDTree(c) if c is not None else None
                 for c in self.global_curves()]
        out = {}
        for i in self.ids:
            deg, dx, dz = self.state[i]
            d = [trees[r].query(self.placed(i)[r])[0].mean()
                 if trees[r] is not None else np.nan for r in range(3)]
            out[i] = (float(deg), float(np.hypot(dx, dz)),
                      self.drift(i, self.state[i]), d)
        return out


def horizontal_transform(state, pivot_xz):
    """A 2D turn+slide as a world-space 4x4, level kept untouched."""
    deg, dx, dz = state
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    piv = np.array([pivot_xz[0], 0.0, pivot_xz[1]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = piv + np.array([dx, 0.0, dz]) - R @ piv
    return T
