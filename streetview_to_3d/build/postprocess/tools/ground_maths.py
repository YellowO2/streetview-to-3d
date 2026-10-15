"""Maths for finding and laying ground: squares, normals, camera weights.

square_keys keys points by their grid square, normals_from_neighbours estimates a surface's
normals, blend weighs a camera by how much further it is than the nearest one.
"""
import numpy as np
from scipy.spatial import cKDTree

BLEND_M = 2.0             # a camera this much further than the nearest counts 1/e as much


def square_keys(xz, cell):
    """One int64 key per point's cell x cell square; neighbours differ by (dx << 21) + dz."""
    c = np.floor(xz / cell).astype(np.int64) + 2 ** 20
    return (c[:, 0] << 21) | c[:, 1]


def normals_from_neighbours(x, k=12):
    """Each point's surface normal, from the flattest direction of its k
    nearest neighbours."""
    _, nb = cKDTree(x).query(x, k=k, workers=-1)
    P = x[nb] - x[nb].mean(1, keepdims=True)
    return np.linalg.eigh(np.einsum("nki,nkj->nij", P, P))[1][:, :, 0]


def blend(dist, nearest):
    """Weight of a camera dist away where the nearest is `nearest` away: 1 for the nearest,
    1/e one BLEND_M further, 0 for inf. Used for both ground height and colour."""
    near = np.where(np.isfinite(nearest), nearest, 0.0)
    return np.where(np.isfinite(dist), np.exp(-(np.where(np.isfinite(dist), dist, 0.0) - near) / BLEND_M), 0.0)
