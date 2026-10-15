"""Estimate each point's surface normal from its nearest neighbours."""
import numpy as np
from scipy.spatial import cKDTree


def normals_from_neighbours(x, k=12):
    """Each point's surface normal, from the flattest direction of its k
    nearest neighbours."""
    _, nb = cKDTree(x).query(x, k=k, workers=-1)
    P = x[nb] - x[nb].mean(1, keepdims=True)
    return np.linalg.eigh(np.einsum("nki,nkj->nij", P, P))[1][:, :, 0]
