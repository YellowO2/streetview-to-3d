"""Key points by the square of a grid they fall in."""
import numpy as np


def square_keys(xz, cell):
    """One int64 key per point's cell x cell square; neighbours differ by (dx << 21) + dz."""
    c = np.floor(xz / cell).astype(np.int64) + 2 ** 20
    return (c[:, 0] << 21) | c[:, 1]
