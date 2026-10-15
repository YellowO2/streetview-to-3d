"""The roads round the scene, laid on its ground and met to its own.

OSM's roads, squares and sidewalks as one surface; its height the ground's smoothed, eased
onto the scene's roads where the panos stand, bridges clearing what they cross; points near
the cameras, painted from the panos, a mesh further off, with tunnel portals.
"""
from streetview_to_3d.build_scene.postprocess.world.PanoViews import PanoViews
from streetview_to_3d.build_scene.postprocess.world.roads.Network import Network
from streetview_to_3d.build_scene.postprocess.world.roads.RoadHeight import RoadHeight
from streetview_to_3d.build_scene.postprocess.world.roads.Roads import Roads
from streetview_to_3d.build_scene.postprocess.world.roads.portals import portals
from streetview_to_3d.build_scene.postprocess.world.roads.road_mesh import road_mesh
from streetview_to_3d.build_scene.postprocess.world.roads.road_points import road_points


def roads(site, elements, elevation, water, views: PanoViews):
    """Roads from OSM's elements round site (Site), on elevation (Elevation), over water (Water)."""
    network = Network(elements, site)
    height = RoadHeight(network, elements, site, elevation, water)
    mouths = portals(network, height)
    mesh = road_mesh(network, height, site, mouths)
    points = road_points(network, height, site, water, views)
    laid = Roads(network, height, mesh, mouths, points)
    return laid
