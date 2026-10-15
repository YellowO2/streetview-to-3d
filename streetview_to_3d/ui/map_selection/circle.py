"""Draw an area round a place as a polygon whose corners can be dragged."""
import numpy as np

from streetview_to_3d.tools.geo import local_m_to_latlon


def circle(lat, lon, radius_m, corners=8):
    """An area as a polygon: a few (lat, lon) corners on the circle radius_m round (lat, lon),
    so dragging one moves a good part of its edge."""
    a = np.linspace(0, 2 * np.pi, corners, endpoint=False)
    return [list(local_m_to_latlon(radius_m * np.cos(t), radius_m * np.sin(t), lat, lon)) for t in a]
