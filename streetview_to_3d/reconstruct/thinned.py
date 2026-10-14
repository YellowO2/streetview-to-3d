"""Thin a cloud to one averaged point per small cube of space."""
import numpy as np

from streetview_to_3d.common.Cloud import Cloud

# The cube's side: catches the near-duplicate points of overlapping views and panos
# (panoramic_da3's Saver thinned the same way when the clouds went through a .ply).
VOXEL_M = 0.03


def thinned(points, colours, voxel_m=VOXEL_M):
    """A Cloud of one point per occupied voxel_m cube, each the mean of its points and colours."""
    points, colours = np.asarray(points), np.asarray(colours)
    if not len(points):
        return Cloud(points, colours)
    cells = np.floor(points / voxel_m).astype(np.int64)
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).reshape(-1)
    counts = counts.astype(np.float64)

    def mean(values):
        out = np.empty((len(counts), values.shape[1]), dtype=np.float64)
        for c in range(values.shape[1]):
            out[:, c] = np.bincount(inverse, weights=values[:, c], minlength=len(counts))
        return out / counts[:, None]

    return Cloud(mean(points).astype(points.dtype), mean(colours).astype(colours.dtype))
