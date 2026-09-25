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
    Everything above it is vegetation, vehicles, buildings or noise. Then
    keep only those lowest points that lie on one flat plane (_on_plane),
    which drops the bottoms of walls. No colour, no components, no width.
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
    low = np.column_stack([p_s[keep, 0], h_s[keep], p_s[keep, 1]])
    return low[_on_plane(low)]


FLOOR_SECTORS = 12       # directions around a camera, 30 deg each
SECTOR_MIN_PTS = 20      # floor points a direction needs to count as seen
SIDES_RADIUS_M = 8.0


def sides_seen(floor, cams, radius=SIDES_RADIUS_M, sectors=FLOOR_SECTORS):
    """How many of `sectors` directions around its best-seen camera the
    floor was found in. A full ring pins the plane; a strip down one side
    lets it rock about the strip like a plank on its edge."""
    best = 0
    for c in np.atleast_2d(cams):
        d = floor[:, [0, 2]] - c
        r = np.hypot(d[:, 0], d[:, 1])
        near = (r > 1) & (r <= radius)
        k = ((np.arctan2(d[near, 1], d[near, 0]) + np.pi)
             / (2 * np.pi) * sectors).astype(int) % sectors
        best = max(best, int((np.bincount(k, minlength=sectors) >= SECTOR_MIN_PTS).sum()))
    return best


PLANE_TOL_M = 0.15      # how far off the plane a point may be and still be floor
PLANE_MAX_TILT_DEG = 30  # DA3 hands floors back tilted 5-16 deg; walls are ~90
PLANE_TRIES = 300


def _on_plane(pts, tol=PLANE_TOL_M, max_tilt=PLANE_MAX_TILT_DEG, tries=PLANE_TRIES):
    """Mask of the points on the one flat plane most of them share (RANSAC).

    A cell's lowest points are the floor only where the floor was seen; in
    a cell holding nothing but wall they are the bottom of the wall. The
    floor is the one surface that is flat and near level across the whole
    track, so a wall can't sit on it however many points it has. The hole
    under the car costs nothing: the plane is fitted around it."""
    if len(pts) < 3:
        return np.ones(len(pts), bool)
    rng = np.random.default_rng(0)
    min_ny = np.cos(np.radians(max_tilt))
    best = None
    for _ in range(tries):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9 or abs(n[1]) / norm < min_ny:
            continue
        inl = np.abs((pts - a) @ (n / norm)) <= tol
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None:
        return np.ones(len(pts), bool)
    # refit on the inliers, so the plane isn't just the lucky three points
    ctr = pts[best].mean(0)
    n = np.linalg.svd(pts[best] - ctr, full_matrices=False)[2][-1]
    return np.abs((pts - ctr) @ n) <= tol
