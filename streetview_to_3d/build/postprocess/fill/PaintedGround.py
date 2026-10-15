"""The one ground's points, coloured, each by the camera that painted it."""
from dataclasses import dataclass

import numpy as np


@dataclass
class PaintedGround:
    """points (n, 3) in the world; colours (n, 3) 0-1; painter (n,): index of the camera
    whose pano painted each; road (n,): whether that pano calls it road, once known."""
    points: np.ndarray
    colours: np.ndarray
    painter: np.ndarray
    road: np.ndarray | None = None
