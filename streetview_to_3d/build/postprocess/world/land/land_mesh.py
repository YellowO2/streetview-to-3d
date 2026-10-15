"""The land's corners joined into triangles."""
import numpy as np

from streetview_to_3d.build.postprocess.world.Land import Land


def land_mesh(points, elevation):
    """Land: LandPoints triangulated (Delaunay), corners no triangle uses dropped; elevation
    (Elevation) stands past the mesh."""
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay
    en, h = points.xy, points.height
    tri = Delaunay(en)
    faces = tri.simplices
    at = LinearNDInterpolator(tri, h)
    seen = np.zeros(len(en), bool)
    seen[faces] = True
    faces = (np.cumsum(seen) - 1)[faces]
    en, h, dist, edge_c, lat, lon = (a[seen] for a in (en, h, points.dist, points.edge_colour, points.lat, points.lon))
    return Land(np.stack([en[:, 0], -h, en[:, 1]], 1), faces, en, h, dist, edge_c, lat, lon, elevation, at)
