"""Heights above sea level measured at places round the scene."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Elevations:
    """xy (n, 2) east/north metres; metres (n,) above sea level there (Google's datum)."""
    xy: np.ndarray
    metres: np.ndarray
