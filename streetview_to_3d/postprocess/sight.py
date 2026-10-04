"""What a pano saw: the nearest of its DA3 points in each direction, and where other points lie
against that."""
import numpy as np

SIGHT_DEG = 1.5     # directions in cells this wide


class Sight:
    """A pano's sight from cam of its DA3 points cloud (world: y down)."""

    def __init__(self, cloud, cam):
        self.cam = np.asarray(cam, float)
        self.n = int(360 / SIGHT_DEG), int(180 / SIGHT_DEG)
        i, j, depth = self._cells(cloud)
        self.saw = np.full(self.n, np.inf)                  # the nearest depth per direction
        np.minimum.at(self.saw, (i, j), depth)
        # how surely it saw a surface there: the share of the direction's 3 by 3 holding points
        held = np.pad(np.isfinite(self.saw), ((0, 0), (1, 1)))
        self.sure = sum(np.roll(held, a, 0)[:, 1 + b:self.n[1] + 1 + b]
                        for a in (-1, 0, 1) for b in (-1, 0, 1)) / 9

    def _cells(self, pts):
        rel = pts - self.cam
        depth = np.linalg.norm(rel, axis=1)
        az = np.arctan2(rel[:, 0], rel[:, 2])
        el = np.arcsin(np.clip(-rel[:, 1] / np.maximum(depth, 1e-9), -1, 1))
        i = np.minimum(((az + np.pi) / (2 * np.pi) * self.n[0]).astype(int), self.n[0] - 1)
        j = np.minimum(((el + np.pi / 2) / np.pi * self.n[1]).astype(int), self.n[1] - 1)
        return i, j, depth

    def of(self, pts):
        """(behind, sure) per point: metres behind what the pano saw in its direction (negative:
        in front of it; inf where it saw nothing), and how surely it saw a surface there."""
        i, j, depth = self._cells(pts)
        return depth - self.saw[i, j], self.sure[i, j]
