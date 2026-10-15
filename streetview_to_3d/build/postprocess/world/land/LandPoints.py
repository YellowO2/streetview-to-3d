"""The land's corners, with what is known at each."""
from dataclasses import dataclass

import numpy as np


@dataclass
class LandPoints:
    """xy (n, 2) east/north corners; owner (n,): the outline each wall corner is on, -1 for the rest.
    From map_heights: dist, edge_height, edge_colour, the scene's ground nearest (SceneGround.at);
    lat, lon; height (n,) the land's, m."""
    xy: np.ndarray
    owner: np.ndarray
    dist: np.ndarray = None
    edge_height: np.ndarray = None
    edge_colour: np.ndarray = None
    lat: np.ndarray = None
    lon: np.ndarray = None
    height: np.ndarray = None
