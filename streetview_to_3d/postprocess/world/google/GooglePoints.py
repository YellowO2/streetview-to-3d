"""Google's 3D Tiles round the scene as points, as the viewer saved them."""
from dataclasses import dataclass

import numpy as np

MERGED = "merged"         # the comment a merged google.ply carries


@dataclass
class GooglePoints:
    """pts (n, 3) in the world (y down); cols (n, 3) 0-1; gap (n,) spacing; normal (n, 3);
    credit: the .ply's credit comments; merged: already merged into a scene (nothing to do)."""
    pts: np.ndarray
    cols: np.ndarray
    gap: np.ndarray
    normal: np.ndarray
    credit: list
    merged: bool = False
