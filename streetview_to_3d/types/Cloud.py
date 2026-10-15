"""One node's DA3 points in its piece's frame, with their colours."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Cloud:
    """points (n, 3) in the node's piece frame (its ply's frame); colours (n, 3) 0-1;
    sway (n,) 0-1, how far each point sways in the wind, once the fill has set it."""
    points: np.ndarray
    colours: np.ndarray
    sway: np.ndarray | None = None

    def world(self, transform, every=1):
        """The points in the world, every every-th, under a placed node's transform."""
        T = np.asarray(transform, float)
        return self.points[::every] @ T[:3, :3].T + T[:3, 3]
