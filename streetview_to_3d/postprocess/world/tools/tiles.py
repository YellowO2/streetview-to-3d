"""Web-mercator tile coordinates of places."""
import numpy as np


def tile_xy(lat, lon, zoom):
    """Fractional web-mercator tile (x, y) of (lat, lon) at zoom."""
    n = 2 ** zoom
    return (np.asarray(lon) + 180) / 360 * n, (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n
