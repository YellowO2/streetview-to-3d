"""The roads laid round the scene: surface, heights, bridges, look."""
from dataclasses import dataclass

import numpy as np

from streetview_to_3d.build_scene.postprocess.world.roads.Network import SHOULDER_M, SINK_M, Network
from streetview_to_3d.build_scene.postprocess.world.roads.RoadHeight import RoadHeight
from streetview_to_3d.build_scene.postprocess.world.roads.RoadPoints import RoadPoints
from streetview_to_3d.build_scene.postprocess.world.roads.tools.paving import STRIP_M
from streetview_to_3d.build_scene.postprocess.world.tools.osm_roads import lines
from streetview_to_3d.build_scene.postprocess.world.tools.seams import ramp


@dataclass
class Roads:
    """network: the surface (Network); height: how high it runs, its bridges' decks (RoadHeight);
    mesh: roads.ply's (road_mesh) away from the cameras, tunnel mouths included; mouths: the
    tunnel mouths' own (points, colours, triangles); points: RoadPoints near the cameras.
    For the land: edges() where it needs corners, fitted() its heights under the roads."""
    network: Network
    height: RoadHeight
    mesh: dict
    mouths: tuple
    points: RoadPoints

    def lines(self):
        """[(centreline, width, colour)] of the ground-level roads (tools/osm_roads.lines)."""
        return lines(self.network.elements, self.network.to_xy)

    def edges(self):
        """(n, 2) points along the surface's and the bridges' outlines, STRIP_M apart."""
        return np.concatenate([self.network.edges(), _deck_edges(self.height.decks)])

    def fitted(self, xy, h):
        """Land heights h at xy fitted to the roads (Network.adapt) and kept under their bridges."""
        return _under_decks(self.height.decks, xy, self.network.adapt(xy, h, self.height))


def _deck_edges(decks_, step_m=STRIP_M):
    """(n, 2) points along every deck's outline, step_m apart."""
    import shapely
    if not decks_:
        return np.zeros((0, 2))
    return shapely.get_coordinates(shapely.segmentize(shapely.boundary(
        shapely.union_all([d.footprint() for d in decks_])), step_m))
def _under_decks(decks_, xy, h):
    """Land heights h at xy kept SINK_M under any deck above, easing back over SHOULDER_M."""
    from scipy.spatial import cKDTree
    if not decks_:
        return h
    centre = np.concatenate([d.xy for d in decks_])
    height = np.concatenate([d.h for d in decks_])
    half = np.concatenate([np.full(len(d.xy), d.width / 2) for d in decks_])
    dist, k = cKDTree(centre).query(xy, distance_upper_bound=float(half.max()) + SHOULDER_M, workers=-1)
    near = np.flatnonzero(np.isfinite(dist))
    near = near[dist[near] < half[k[near]] + SHOULDER_M]
    if not len(near):
        return h
    out = (dist[near] - half[k[near]]) / SHOULDER_M      # 0 under the deck, 1 at the shoulder's edge
    top = height[k[near]] - SINK_M
    h = h.copy()
    h[near] = np.minimum(h[near], top + (h[near] - top) * ramp(out))
    return h
