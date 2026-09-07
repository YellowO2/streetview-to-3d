"""Place several pieces at once, rather than chaining them pairwise.

Chaining breaks on a run like 5-3-9-6 for a structural reason, not a
tuning one: piece_3 is the only piece close enough to link piece_5 to the
9/6 group, and piece_3 is the single-node piece with no GPS heading of
its own. Align 3 to 5 first and the well-placed 9 and 6 are dragged to
fit it; align 3 to 9 first and 5 has nothing left to attach to. No
ordering avoids this, because the run's connectivity passes through its
weakest piece.

So every piece gets a turn and a slide and they are solved together, by
sweeping: each piece in turn takes its best placement against ALL its
neighbours' kerbs at once, while the others hold still, until nothing
moves. Least-constrained pieces go first, so the ones that are free to
rotate settle before the ones that are not.

One penalty holds it together -- how far a piece drags its own cameras
off GPS -- and it encodes heading confidence as well as position without
a separate term. Turning a piece about its cameras' centre displaces them
in proportion to how spread out they are, so a four-node piece over a
28 m track resists rotation strongly, while a single-node piece, whose
camera sits at the pivot, turns for free. Which is right: one GPS point
fixes where a piece is and says nothing about which way it faces.

Solves the horizontal only. Fit height and tilt afterwards with
`kerb_align.slope_fit`, once the pieces are correctly placed and the
overlapping road is genuinely the same road.
"""
import numpy as np
from scipy.spatial import cKDTree

from alignment.kerb import kerb_curves, thin
from alignment.kerb_align import (MAX_SEP_M, MIN_SEAM_PTS, SEAM_R,
                                  noise_floor, track_span)

NEIGHBOUR_GAP_M = 16.0     # past this, two panoramas share too little ground
GPS_WEIGHT = 0.030         # residual-metres charged per metre of camera drift
NO_SEAM_PENALTY = 1.5      # charged when a neighbour drifts out of reach
def drift_cap(n_nodes):
    """How far a piece's cameras may be dragged off GPS, by node count.

    Not one number for everything: what GPS knows about a piece depends on
    how many places it saw it from. One node fixes a position and says
    nothing about heading, so such a piece has to be free to swing. Two
    nodes pin a direction but fit any similarity transform exactly, so
    their zero residual means nothing and they deserve less trust than the
    count suggests. Three or more genuinely constrain the piece, and moving
    one of those far is a sign the fit is wrong, not that GPS was.
    """
    return {1: 5.0, 2: 5.0}.get(n_nodes, 3.0)


MAX_SLIDE_M = 8.0
SWEEPS = 4


def seam_fast(A, B, tree_b):
    """kerb_align.seam_fit with the neighbour's KD-tree supplied.

    Rebuilding the tree inside the search loop rather than once per
    neighbour is the difference between minutes and hours.
    """
    d, i = tree_b.query(A)
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
    _, _, V = np.linalg.svd(c, full_matrices=False)
    u, v = c @ V[0], c @ V[1]
    return float(np.sqrt(np.mean((v - np.polyval(np.polyfit(u, v, 2), u)) ** 2))), sep


def _rot(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


class JointSolver:
    """kerbs/cams: {piece_id: ...} in a shared LEVELLED 2D frame."""

    def __init__(self, kerbs, cams, gaps, floor=None, lock=None):
        self.ids = list(kerbs)
        self.kerbs = {i: [thin(c) for c in kerbs[i]] for i in self.ids}
        self.cams = cams
        self.pivot = {i: cams[i].mean(0) for i in self.ids}
        self.state = {i: np.zeros(3) for i in self.ids}       # deg, dx, dz
        self.cap = {i: drift_cap(len(cams[i])) for i in self.ids}
        self.floor = floor if floor is not None else noise_floor(list(kerbs.values()))
        self.pairs = [(i, j) for (i, j), g in gaps.items() if g <= NEIGHBOUR_GAP_M]

        # Which curve of one piece is the same road edge as which curve of
        # the other is decided ONCE and held. Let the search re-choose and
        # it drifts onto the opposite kerb as the piece slides, then scores
        # that as a fine join.
        #
        # A caller that knows its curves are in a consistent order -- left
        # kerb first, right kerb second, oriented the same way along the
        # route -- should pass `lock` and pair them by index. Choosing by
        # closest approach instead pairs whichever happens to be nearest,
        # and that routinely marries one piece's left kerb to another's
        # right.
        if lock is not None:
            self.lock = dict(lock)
            return
        self.lock = {}
        for i, j in self.pairs:
            best = None
            for ai, A in enumerate(self.kerbs[i]):
                for bi, B in enumerate(self.kerbs[j]):
                    d = float(cKDTree(B).query(A)[0].min())
                    if best is None or d < best[0]:
                        best = (d, ai, bi)
            self.lock[(i, j)] = (best[1], best[2])

    def neighbours(self, i):
        for a, b in self.pairs:
            if a == i:
                yield b, *self.lock[(a, b)]
            elif b == i:
                yield a, self.lock[(a, b)][1], self.lock[(a, b)][0]

    def placed(self, i, st=None):
        deg, dx, dz = self.state[i] if st is None else st
        R = _rot(deg)
        p = self.pivot[i]
        return [(c - p) @ R.T + p + [dx, dz] for c in self.kerbs[i]]

    def drift(self, i, st):
        deg, dx, dz = st
        p = self.pivot[i]
        q = (self.cams[i] - p) @ _rot(deg).T + p + [dx, dz]
        return float(np.linalg.norm(q - self.cams[i], axis=1).mean())

    def cost(self, i, st, cache):
        mine = self.placed(i, st)
        total = n = 0
        for j, mi, _ in self.neighbours(i):
            B, tree = cache[j]
            r = seam_fast(mine[mi], B, tree)
            # A neighbour out of reach must COST something. Averaging only
            # over the seams that survive makes losing one free, and the
            # solve then tears a join apart to tidy up the others.
            total += NO_SEAM_PENALTY if r is None else r[0]
            n += 1
        return None if n == 0 else total / n + GPS_WEIGHT * self.drift(i, st)

    def best_for(self, i):
        cache = {}
        for j, _, ji in self.neighbours(i):
            B = self.placed(j)[ji]
            cache[j] = (B, cKDTree(B))
        cur = self.cost(i, self.state[i], cache)
        best = (1e9 if cur is None else cur, self.state[i].copy())
        for (dstep, tstep), (drange, trange) in (((3.0, 1.0), (45.0, 6.0)),
                                                 ((0.5, 0.25), (6.0, 1.5))):
            d0, x0, z0 = best[1]
            for deg in np.arange(d0 - drange, d0 + drange + 1e-9, dstep):
                for dz in np.arange(z0 - trange, z0 + trange + 1e-9, tstep):
                    for dx in np.arange(x0 - trange, x0 + trange + 1e-9, tstep):
                        if np.hypot(dx, dz) > MAX_SLIDE_M:
                            continue
                        st = np.array([deg, dx, dz])
                        if self.drift(i, st) > self.cap[i]:
                            continue
                        c = self.cost(i, st, cache)
                        if c is not None and c < best[0]:
                            best = (c, st)
        return best[1]

    def solve(self, sweeps=SWEEPS, tol=0.3, log=print):
        order = sorted(self.ids, key=lambda i: track_span(self.cams[i]))
        for sweep in range(sweeps):
            moved = 0.0
            for i in order:
                st = self.best_for(i)
                moved = max(moved, float(np.abs(st - self.state[i]).max()))
                self.state[i] = st
            if log:
                log(f"sweep {sweep + 1}: largest change {moved:.2f}  " +
                    "  ".join(f"{i}:{self.state[i][0]:+.1f}d"
                              f"/{np.hypot(*self.state[i][1:]):.2f}m"
                              for i in self.ids))
            if moved < tol:
                break
        return self.state

    def seams(self):
        """{(i, j): residual_m or None} for every neighbour pair."""
        out = {}
        for i, j in self.pairs:
            mi, ji = self.lock[(i, j)]
            r = seam_fast(self.placed(i)[mi], self.placed(j)[ji],
                          cKDTree(self.placed(j)[ji]))
            out[(i, j)] = None if r is None else r[0]
        return out
