"""The placed scene as the map around it sees it."""
import logging
import math
from functools import cached_property

import numpy as np

from streetview_to_3d.build.postprocess.SceneGround import SceneGround
from streetview_to_3d.build.postprocess.world.tools.reach import M_PER_LAT
from streetview_to_3d.build.postprocess.world.tools.scene_points import scene_points
from streetview_to_3d.build.postprocess.world.tools.seams import spacing_at

logger = logging.getLogger(__name__)


class Site:
    """Where the scene stands, in east/north metres from its origin (to_xy, to_ll), and what it holds.

    cameras: the placed cameras (n, 2); anchors, elevations: the panos Google gives an
    elevation, where and how high; ground: the fill's SceneGround; points, colours: DA3's
    points in the world (tools/scene_points.py), tree a cKDTree of them (None without);
    normals, from their neighbours, made when first asked for. No cameras: nothing else is set.
    """

    def __init__(self, scene, ground=None):
        from scipy.spatial import cKDTree
        self.origin = lat0, lon0 = scene.origin
        self.m_per_lon = M_PER_LAT * math.cos(math.radians(lat0))
        cams = [n for n in scene.nodes if n.transform and n.position is not None]
        self.cameras = np.array([(np.asarray(n.transform)[:3, :3] @ n.position
                                  + np.asarray(n.transform)[:3, 3])[[0, 2]] for n in cams]).reshape(-1, 2)
        if not cams:
            logger.warning("terrain: no placed cameras")
            return
        self.panos = [n.pano for n in scene.nodes if n.pano.elevation is not None]
        self.anchors = np.array([((p.lon - lon0) * self.m_per_lon, (p.lat - lat0) * M_PER_LAT)
                                 for p in self.panos]).reshape(-1, 2)
        self.elevations = np.array([p.elevation for p in self.panos])
        self.ground = ground or SceneGround.none()
        self.points, self.colours = scene_points(scene)
        self.tree = cKDTree(self.points) if len(self.points) else None
        self.camera_tree = cKDTree(self.cameras)
        # map points sized, so spaced, by their distance from DA3's points (spacing_at): from the
        # scene's edge, 1 m squares holding at least 3 of its points
        scene_ = self.points
        cell, n = (np.unique(np.floor(scene_[:, [0, 2]]), axis=0, return_counts=True) if len(scene_)
                   else (self.cameras, None))
        self._edge = cKDTree(cell[n >= 3] + 0.5 if n is not None and (n >= 3).any() else self.cameras)

    def to_ll(self, xy):
        """(lat, lon) of east/north points (n, 2)."""
        lat0, lon0 = self.origin
        return lat0 + xy[:, 1] / M_PER_LAT, lon0 + xy[:, 0] / self.m_per_lon

    def to_xy(self, geometry):
        """East/north (n, 2) of OSM geometry: [{lat, lon}, ...]."""
        lat0, lon0 = self.origin
        return np.array([((p["lon"] - lon0) * self.m_per_lon, (p["lat"] - lat0) * M_PER_LAT)
                         for p in geometry])

    def from_cameras(self, xy):
        """Metres from each east/north point to the nearest camera."""
        return self.camera_tree.query(xy, workers=-1)[0]

    def spacing(self, xy):
        """Map point spacing at east/north points, by their distance from DA3's (spacing_at)."""
        return spacing_at(self._edge.query(xy, workers=-1)[0])

    @cached_property
    def normals(self):
        """(n, 3) normals of DA3's points, from their neighbours (zero under 12 points)."""
        from streetview_to_3d.build.postprocess.tools.ground_maths import normals_from_neighbours
        return normals_from_neighbours(self.points) if len(self.points) >= 12 else np.zeros_like(self.points)
