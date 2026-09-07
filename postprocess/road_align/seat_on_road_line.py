"""Seat each piece on the road line the graph says it drove along.

This is the whole horizontal alignment. It replaced a kerb-matching solver
that reduced every piece to a left kerb, a centre and a right kerb and slid
them onto pooled global curves: that solver reconstructed gap3 worse and
took five minutes against this one's seconds, because a kerb has to be
inferred from a road mask and the inference failed in ways that were hard
to see -- a car park connected to the road became the kerb, a junction
mouth became an S-bend, and a piece was then confidently aligned to it.

What a piece is allowed to do depends on what GPS actually measured about
it, which is the same principle `drift_cap` encodes:

    more than STRONG_NODES cameras   translation only. GPS saw the piece
                                     from enough places to have measured
                                     its heading, and the road line is
                                     that same GPS smoothed -- overruling
                                     a direct measurement with a smoothed
                                     one loses information.
    STRONG_NODES or fewer            rotation as well. GPS never measured
                                     a heading here: two points fix a
                                     position and nothing else. The road
                                     line is the only thing that can
                                     supply one.

Both are held near GPS by a soft (move/cap)**4 penalty rather than a hard
limit. The penalty is also what stops the piece sliding ALONG the road: a
road is featureless lengthwise, so that direction is unconstrained and an
unpenalised fit runs away down the curve -- one piece travelled 34 m
before this term existed.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

STRONG_NODES = 2         # above this, GPS measured the heading itself
TURN_RANGE_DEG = 40.0
TURN_STEP_DEG = 2.0
PENALTY_POWER = 4


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


def _rot(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


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


def seat(cams, curve, may_turn):
    """(4x4, turn_deg, shift_m) placing one piece's cameras on `curve`."""
    tree = cKDTree(curve)
    pivot, cap = cams.mean(0), drift_cap(len(cams))

    def cost(deg, t):
        q = (cams - pivot) @ _rot(deg).T + pivot + t
        return ((tree.query(q)[0] ** 2).mean()
                + (np.linalg.norm(q - cams, axis=1).mean() / cap) ** PENALTY_POWER)

    turns = (np.arange(-TURN_RANGE_DEG, TURN_RANGE_DEG + 0.1, TURN_STEP_DEG)
             if may_turn else [0.0])
    best = None
    for deg in turns:
        # the turn is swept rather than optimised: the cost is not smooth in
        # it (each step re-picks which bit of curve each camera is nearest)
        # and a gradient method settles into whichever basin it started in
        o = minimize(lambda t, deg=deg: cost(deg, t), [0.0, 0.0],
                     method="Nelder-Mead",
                     options=dict(xatol=1e-2, fatol=1e-4, maxiter=400))
        if best is None or o.fun < best[0]:
            best = (o.fun, deg, o.x)
    _, deg, t = best
    return horizontal_transform(np.r_[deg, t], pivot), float(deg), float(np.linalg.norm(t))


def seat_all(cams, curves, on, log=print):
    """{piece: 4x4} seating every piece on its nearest road.

    A piece is seated on the ONE road its cameras sit closest to. Pieces
    spanning a junction touch several, but they were driven along one of
    them, and that is the one whose line describes where they were.
    """
    out = {}
    for i in sorted(cams):
        if not on.get(i):
            out[i] = np.eye(4)
            log(f"  piece_{i:<4} no road -- left at GPS")
            continue
        r = min(on[i], key=lambda r: cKDTree(curves[r]).query(cams[i])[0].mean())
        may_turn = len(cams[i]) <= STRONG_NODES
        out[i], deg, shift = seat(cams[i], curves[r], may_turn)
        log(f"  piece_{i:<4} {len(cams[i])} node(s) on road{r:<4} "
            + (f"turn {deg:+5.0f} deg, " if may_turn else "no turn (GPS fixed it), ")
            + f"shift {shift:.2f} m")
    return out
