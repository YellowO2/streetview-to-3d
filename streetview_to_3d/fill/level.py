"""The pieces lifted or lowered to meet: one up/down shift per piece.

Each piece is placed on its own by GPS (postprocess/place.py), so two that
meet -- a cross street at a junction -- can stand tens of cm apart, the
whole piece, road and walls alike, and the one ground made a step there.
A kerb is not that: it is a step within a cloud's own ground, and every
piece that sees it sees it alike, so it cancels out of the difference
between two pieces.

1. each piece's road: its clouds' walkable ground (one_ground.grounds), as
   the median height per CELL_M square
2. where two pieces both have road in MIN_SQUARES squares or more, their
   offset is the median difference over those squares: a few odd ones (a
   kerb seen differently, a mislabelled car) do not move it
3. one shift per piece, all offsets cancelled as well as they can be
   (least squares, weighted by overlap, none moved more than MAX_M), the
   scene on average where GPS put it -- what is left the one ground's
   height blend (postprocess.ground.GroundMap) smooths out
"""
import numpy as np
from scipy.optimize import lsq_linear

CELL_M = 0.5
MIN_SQUARES = 20
MAX_M = 0.4
STAY = 1.0          # how strongly a piece keeps its GPS height, against one overlap square's worth


def _road(x):
    """(sorted square keys, median height in each) of ground points x."""
    sq = np.floor(x[:, [0, 2]] / CELL_M).astype(np.int64) + 2 ** 20
    key = (sq[:, 0] << 21) | sq[:, 1]
    o = np.argsort(key, kind="stable")
    uk, start = np.unique(key[o], return_index=True)
    return uk, np.array([np.median(s) for s in np.split(x[o, 1], start[1:])])


def shifts(ground_pts, piece):
    """({piece id: its shift in metres along y, down}, number of links).

    ground_pts[k]: cloud k's walkable ground points (world metres, y down);
    piece[k]: the id of the piece cloud k belongs to."""
    ids = sorted(set(piece))
    roads = []
    for p in ids:
        x = [g for g, q in zip(ground_pts, piece) if q == p and len(g)]
        roads.append(_road(np.concatenate(x)) if x else None)
    rows, rhs = [], []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if roads[a] is None or roads[b] is None:
                continue
            _, ia, ib = np.intersect1d(roads[a][0], roads[b][0], return_indices=True)
            if len(ia) < MIN_SQUARES:
                continue
            w = np.sqrt(len(ia))
            row = np.zeros(len(ids))
            row[a], row[b] = w, -w
            rows.append(row)                     # after shifting, a's road meets b's
            rhs.append(-w * np.median(roads[a][1][ia] - roads[b][1][ib]))
    if not rows:
        return {p: 0.0 for p in ids}, 0
    # the scene as a whole stays where GPS put it (the shifts average 0), and
    # a piece little linked keeps near its own height
    hold = np.abs(np.vstack(rows)).sum() / 2 * np.ones((1, len(ids))) / len(ids)
    A = np.vstack(rows + [hold, STAY * np.eye(len(ids))])
    s = lsq_linear(A, np.r_[rhs, 0.0, np.zeros(len(ids))], bounds=(-MAX_M, MAX_M)).x
    return {p: float(v) for p, v in zip(ids, s)}, len(rows)


def level(sc, nodes, clouds, G):
    """Level the scene sc's pieces, in place: every node's transform, and
    the world clouds of `nodes` (those of sc.nodes with points; G their
    walkable ground), shifted by its piece's. Returns a line for the log."""
    piece_of = {i: p for p, members in enumerate(sc.pieces()) for i in members}
    index = {id(n): i for i, n in enumerate(sc.nodes)}
    piece = [piece_of.get(index[id(n)], -1 - k) for k, n in enumerate(nodes)]
    shift, n_links = shifts([x[g] for x, g in zip(clouds, G)], piece)
    for i, n in enumerate(sc.nodes):
        if n.transform and i in piece_of:
            T = np.asarray(n.transform, float)
            T[1, 3] += shift.get(piece_of[i], 0.0)         # y, down; a piece with no points stays
            n.transform = T.tolist()
    for k, x in enumerate(clouds):
        x[:, 1] += shift[piece[k]]
    moved = np.abs(list(shift.values()))
    return (f"fill: {len(shift)} piece(s) levelled over {n_links} overlap(s), moved up to "
            f"{moved.max() * 100:.0f} cm ({int((moved >= MAX_M - 1e-9).sum())} at the {MAX_M * 100:.0f} cm limit)")
