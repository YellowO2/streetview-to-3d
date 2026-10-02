"""Surface sampling shared by procedural buildings and streets."""
import numpy as np


def sample_quad(quad, step):
    """Grid a parallelogram without subdividing the base wall or roof."""
    a, b, _, d = quad
    u, v = b - a, d - a
    nu, nv = max(1, int(np.ceil(np.linalg.norm(u) / step))), max(1, int(np.ceil(np.linalg.norm(v) / step)))
    U, V = np.meshgrid((np.arange(nu) + .5) / nu, (np.arange(nv) + .5) / nv)
    return a + U.ravel()[:, None] * u + V.ravel()[:, None] * v


def sample_quads(quads, step):
    """sample_quad over many parallelograms (m, 4, 3) at once: (points, the
    quad each is on), each quad's points in the same order sample_quad has."""
    a, u, v = quads[:, 0], quads[:, 1] - quads[:, 0], quads[:, 3] - quads[:, 0]
    nu = np.maximum(1, np.ceil(np.linalg.norm(u, axis=1) / step)).astype(int)
    nv = np.maximum(1, np.ceil(np.linalg.norm(v, axis=1) / step)).astype(int)
    count = nu * nv
    which = np.repeat(np.arange(len(quads)), count)
    k = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    U = ((k % nu[which]) + .5) / nu[which]
    V = ((k // nu[which]) + .5) / nv[which]
    return a[which] + U[:, None] * u[which] + V[:, None] * v[which], which
