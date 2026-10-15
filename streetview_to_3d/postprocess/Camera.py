"""Where a placed node's camera looks, and what its pano calls each point."""
import numpy as np

from streetview_to_3d.models.segment import LABEL_IDS
from streetview_to_3d.postprocess.tools.paint import pano_at


class Camera:
    """A placed node's camera: world points to its photo's (u, v), distance
    in metres and degrees below the horizon; photo: its pano's (image path, class map) or None."""

    def __init__(self, node):
        T = np.asarray(node.transform, float)
        self.scale = np.cbrt(np.linalg.det(T[:3, :3]))
        self.R, self.t = T[:3, :3] / self.scale, T[:3, 3]
        self.position = np.asarray(node.position, float)
        self.rotation = np.asarray(node.rotation, float)
        self.centre = T[:3, :3] @ self.position + self.t
        self.photo = node.photo

    def look(self, x):
        d = (((x - self.t) @ self.R) / self.scale - self.position) @ self.rotation.T
        r = np.linalg.norm(d, axis=1)
        s = np.clip(d[:, 1] / np.maximum(r, 1e-9), -1, 1)
        below = np.degrees(np.arcsin(s))
        return np.arctan2(d[:, 0], d[:, 2]) / (2 * np.pi) + .5, below / 180 + .5, r * self.scale, below

    def classed(self, x, classes, unknown):
        """Which of world points x its pano's class map labels one of classes (unknown without one)."""
        if self.photo is None:
            return np.full(len(x), unknown)
        u, v, _, _ = self.look(x)
        return np.isin(pano_at(self.photo[1], u, v), [LABEL_IDS[c] for c in classes])
