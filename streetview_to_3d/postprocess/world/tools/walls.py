"""DA3's points near a building's wall, and on its plane."""
import numpy as np

SAME_DEG = 40             # DA3's copy of a wall: its points facing within this of the wall
ON_WALL_M = 0.5           # ... and within this of its plane


def near_box(xz, centre, half):
    """Indices of the points within half of centre each way (xz: a cKDTree of east/north)."""
    half = np.broadcast_to(np.asarray(half, float), (2,))
    idx = np.asarray(xz.query_ball_point(centre, float(np.hypot(*half))), int)
    return idx[np.all(np.abs(xz.data[idx] - centre) <= half, axis=1)] if len(idx) else idx


def on_plane(n2, d2, da3, da3_normals):
    """True for DA3 points within ON_WALL_M of plane (n2, d2) and facing its way."""
    return (np.abs(da3 @ n2 - d2) < ON_WALL_M) & (np.abs(da3_normals @ n2) > np.cos(np.radians(SAME_DEG)))
