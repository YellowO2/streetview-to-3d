"""Surface sampling shared by procedural buildings and streets."""
import numpy as np


def sample_quad(quad, step):
    """Grid a parallelogram without subdividing the base wall or roof."""
    a, b, _, d = quad
    u, v = b - a, d - a
    nu, nv = max(1, int(np.ceil(np.linalg.norm(u) / step))), max(1, int(np.ceil(np.linalg.norm(v) / step)))
    U, V = np.meshgrid((np.arange(nu) + .5) / nu, (np.arange(nv) + .5) / nv)
    return a + U.ravel()[:, None] * u + V.ravel()[:, None] * v
