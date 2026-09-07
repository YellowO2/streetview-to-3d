"""Set every piece's height and tilt against ONE road surface.

Run this after the horizontal placement, never before. Levelling first
matches whichever bits of road happen to lie near each other, and before
the pieces are correctly placed those are not the same bits of road.

The vertical used to be done greedily while writing the output -- each
piece levelled against whatever had already been placed -- so the answer
depended on the order pieces were written in, and any piece that did not
overlap its predecessors was skipped and simply kept its GPS height.
piece_6 was skipped on every run.

So it is done the same way as the horizontal fit: one surface built from
every piece at once, then each piece moved onto it. A piece needs to
overlap the ROAD, not any particular neighbour, so nothing is skipped for
want of a partner and no ordering is chosen.

Each piece gets one height offset and one tilt. The piece is seated,
never bent -- its internal shape is left exactly as DA3 reconstructed it.
"""
import numpy as np

DEGREE = 3               # terrain over a hundred metres is gentle
HUBER_M = 0.30           # residuals past this stop pulling on the surface
ROUNDS = 4
MIN_ROAD_PTS = 200


def _design(xz, ctr, scale, degree=DEGREE):
    u = (xz - ctr) / scale
    cols = [np.ones(len(u))]
    for total in range(1, degree + 1):
        for a in range(total + 1):
            cols.append(u[:, 0] ** a * u[:, 1] ** (total - a))
    return np.column_stack(cols)


def _robust(A, y):
    """Least squares that stops chasing the points it cannot fit."""
    w = np.ones(len(y))
    coef = None
    for _ in range(6):
        coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        r = np.abs(y - A @ coef)
        w = np.minimum(1.0, HUBER_M / np.maximum(r, 1e-6))
    return coef


def seat(road_by_piece, rounds=ROUNDS):
    """{piece: (N,3) road points} -> ({piece: 4x4}, {piece: (height, tilt)}).

    The transform is a vertical shear: it changes height as a function of
    position and leaves X and Z untouched, so seating never disturbs the
    horizontal placement it was given.
    """
    ids = [i for i, r in road_by_piece.items() if len(r) >= MIN_ROAD_PTS]
    if len(ids) < 2:
        return {i: np.eye(4) for i in road_by_piece}, {}

    pooled = np.vstack([road_by_piece[i] for i in ids])
    ctr = pooled[:, [0, 2]].mean(0)
    scale = float(np.abs(pooled[:, [0, 2]] - ctr).max())
    adj = {i: np.zeros(3) for i in ids}          # height, tilt in x, tilt in z

    def corrected(i):
        a, b, c = adj[i]
        u = (road_by_piece[i][:, [0, 2]] - ctr) / scale
        return road_by_piece[i][:, 1] + a + b * u[:, 0] + c * u[:, 1]

    A_pool = _design(pooled[:, [0, 2]], ctr, scale)
    for _ in range(rounds):
        surf = A_pool @ _robust(A_pool, np.concatenate([corrected(i) for i in ids]))
        k, moved = 0, 0.0
        for i in ids:
            n = len(road_by_piece[i])
            residual = surf[k:k + n] - corrected(i)
            k += n
            u = (road_by_piece[i][:, [0, 2]] - ctr) / scale
            d = _robust(np.column_stack([np.ones(n), u[:, 0], u[:, 1]]), residual)
            adj[i] += d
            moved = max(moved, float(np.abs(d).max()))
        if moved < 0.01:
            break

    out, report = {}, {}
    for i in road_by_piece:
        if i not in adj:
            out[i] = np.eye(4)
            continue
        a, b, c = adj[i]
        T = np.eye(4)
        T[1, 0] = b / scale
        T[1, 2] = c / scale
        T[1, 3] = a - (b * ctr[0] + c * ctr[1]) / scale
        out[i] = T
        report[i] = (float(a), float(np.degrees(np.arctan(np.hypot(b, c) / scale))))
    return out, report
