"""Which of each placed cloud's points are ground its pano calls walkable."""
import numpy as np

from streetview_to_3d.postprocess.fill.ground import CELL_M as GROUND_CELL_M, ground
from streetview_to_3d.postprocess.tools.ground_maths import normals_from_neighbours, square_keys

DA3_SEED_M = 7.0        # the ground spread may start this far out (DA3's blind disc reaches ~4.5 m)
LOW_M = 1.0             # only points this near the lowest in their square can be ground
WALKABLE = ("road", "sidewalk", "terrain")      # the masker's classes (models.segment)


def walkable_ground(clouds, cameras):
    """Per cloud (world Clouds, one per camera), True on its walkable ground."""
    points = [c.points for c in clouds]
    return _grounds(points, np.array([c.centre for c in cameras]).reshape(-1, 3),
                    [cam.classed(x, WALKABLE, True) for x, cam in zip(points, cameras)])


def _grounds(clouds, cams, walkable=None):
    """Which of each cloud's points (world, y down) are walkable ground; walkable[k] masks cloud k (None: all)."""
    G = [ground(x, c, normals=_low_normals(x), seed_m=DA3_SEED_M) for x, c in zip(clouds, cams)]
    return G if walkable is None else [g & w for g, w in zip(G, walkable)]


def _low_normals(x):
    """Normals of the points within LOW_M of the lowest in their ground square; zero
    (never ground) for the rest, which saves computing most normals."""
    _, inv = np.unique(square_keys(x[:, [0, 2]], GROUND_CELL_M), return_inverse=True)
    low = np.full(inv.max() + 1, -np.inf)
    np.maximum.at(low, inv, x[:, 1])                  # y is down: the largest y is the lowest
    near = x[:, 1] > low[inv] - LOW_M
    n = np.zeros_like(x)
    n[near] = normals_from_neighbours(x[near])
    return n
