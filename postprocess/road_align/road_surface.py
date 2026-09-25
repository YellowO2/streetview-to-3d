"""The road surface under a piece: the ground its cameras drove along.

Finding the road by colour (grey cells, connected to the cameras) was tried
first and failed where the road is not a tidy grey slab: on a hill it kept
1.7% of one piece's cloud, against 30-43% for its neighbours on the same
road. Seating a piece needs only the surface under the track, so nothing is
classified.
"""
import numpy as np
from scipy.spatial import cKDTree


TRACK_RADIUS_M = 12.0    # a road is not wider than this either side of the track
GROUND_CELL_M = 1.0
GROUND_BAND_M = 0.6      # how thick a road surface is, allowing for noise
GROUND_PCT = 92.0        # Y-DOWN, so a HIGH percentile of y is the ground --
                         # not the very lowest point (noise). At 8 it took
                         # the tops of walls in cells with no road in them.
MIN_TRACK_PTS = 100


def ground_near_track(piece, cams, radius=TRACK_RADIUS_M, cell=GROUND_CELL_M,
                      band=GROUND_BAND_M, pct=GROUND_PCT):
    """The ground the camera drove along, without classifying anything.

    Take the points near the track, and in each cell call the lowest
    points the ground (a high percentile of y, which points down).
    Everything above it is vegetation, vehicles, buildings or noise. No
    colour, no components, no width.
    """
    xz, y, _ = piece
    near = cKDTree(cams).query(xz)[0] <= radius
    if near.sum() < MIN_TRACK_PTS:
        return np.empty((0, 3))
    p, h = xz[near], y[near]
    ij = np.floor(p / cell).astype(np.int64)
    key = ij[:, 0] * 1000003 + ij[:, 1]
    order = np.argsort(key, kind="stable")
    key_s, h_s, p_s = key[order], h[order], p[order]
    edges = np.r_[0, np.flatnonzero(np.diff(key_s)) + 1, len(key_s)]
    keep = np.zeros(len(key_s), bool)
    for a, b in zip(edges[:-1], edges[1:]):
        keep[a:b] = np.abs(h_s[a:b] - np.percentile(h_s[a:b], pct)) <= band
    return np.column_stack([p_s[keep, 0], h_s[keep], p_s[keep, 1]])
