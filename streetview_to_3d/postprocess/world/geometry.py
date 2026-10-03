"""Small geometry helpers shared by buildings, roads and street details."""
import numpy as np


def sample_quads(quads, step):
    """Grid points over parallelograms (m, 4, 3), about step apart: (points, the quad each is on)."""
    a, u, v = quads[:, 0], quads[:, 1] - quads[:, 0], quads[:, 3] - quads[:, 0]
    nu = np.maximum(1, np.ceil(np.linalg.norm(u, axis=1) / step)).astype(int)
    nv = np.maximum(1, np.ceil(np.linalg.norm(v, axis=1) / step)).astype(int)
    count = nu * nv
    which = np.repeat(np.arange(len(quads)), count)
    k = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    U = ((k % nu[which]) + .5) / nu[which]
    V = ((k // nu[which]) + .5) / nv[which]
    return a[which] + U[:, None] * u[which] + V[:, None] * v[which], which


def turning(xy):
    """Twice the signed area of closed ring xy: positive counter-clockwise."""
    return float(np.sum(xy[:-1, 0] * xy[1:, 1] - xy[1:, 0] * xy[:-1, 1]))


def hash01(x, y):
    """Deterministic pseudo-random 0-1 per (x, y)."""
    n = np.sin(x * 12.9898 + y * 78.233) * 43758.5453
    return n - np.floor(n)
