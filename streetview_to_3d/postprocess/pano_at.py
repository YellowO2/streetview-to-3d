"""Read a pano-shaped grid at view coordinates."""
import numpy as np


def pano_at(grid, u, v):
    """grid (a pano-shaped image, class map or mask) at view coordinates (u, v), nearest pixel."""
    h, w = grid.shape[:2]
    return grid[np.clip((v * h).astype(int), 0, h - 1), np.clip((u * w).astype(int), 0, w - 1)]
