"""The roads near the cameras as coloured points."""
from dataclasses import dataclass

import numpy as np


@dataclass
class RoadPoints:
    """pts (n, 3) in the world (y down), cols (n, 3) 0-1, gap (n,) spacing; painted: how many
    the panos coloured."""
    pts: np.ndarray
    cols: np.ndarray
    gap: np.ndarray
    painted: int = 0
