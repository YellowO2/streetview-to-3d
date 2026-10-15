"""The land round the scene as a coloured triangle mesh."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Land:
    """pts (n, 3) in the world (y down) at xy (n, 2) east/north, height (n,); faces (m, 3);
    dist, edge_colour, lat, lon at each (LandPoints'); elevation: the Elevation it was laid on.
    Once coloured (land_colours): cols (n, 3), gap (n,) spacing, near (seams.toward), painted
    (how many the panos coloured), source (where its colour came from)."""
    pts: np.ndarray
    faces: np.ndarray
    xy: np.ndarray
    height: np.ndarray
    dist: np.ndarray
    edge_colour: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    elevation: object
    interpolated: object = None
    cols: np.ndarray = None
    gap: np.ndarray = None
    near: np.ndarray = None
    painted: int = 0
    source: str = ""

    def surface(self, xy):
        """The land's final height at east/north points (the elevation's past the mesh)."""
        v = self.interpolated(xy)
        return np.where(np.isfinite(v), v, self.elevation(xy))
