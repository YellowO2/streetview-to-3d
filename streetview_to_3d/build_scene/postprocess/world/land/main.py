"""The land round the scene: a mesh on the map's ground, met to the scene.

Its corners spaced by their distance from the cameras and along every road, bridge and wall;
its height the map's, met to the scene's ground at its edge and run on just beneath it, fitted
to the roads, raised and cut so the buildings stand on it, carved into shores.
"""
from streetview_to_3d.build_scene.postprocess.world.Outlines import Outlines
from streetview_to_3d.build_scene.postprocess.world.land.land_corners import land_corners
from streetview_to_3d.build_scene.postprocess.world.land.land_mesh import land_mesh
from streetview_to_3d.build_scene.postprocess.world.land.map_heights import map_heights
from streetview_to_3d.build_scene.postprocess.world.land.shore import shore
from streetview_to_3d.build_scene.postprocess.world.land.stood_buildings import stood_buildings


def land(site, elevation, streets, water, footprints: Outlines):
    """Land round site (Site) on elevation (Elevation), streets (Roads) and water (Water); the
    footprints' Forms stood on it."""
    corners = land_corners(site, streets, footprints)
    shaped = map_heights(corners, site, elevation, streets)
    shaped = stood_buildings(shaped, footprints, site)
    shaped = shore(shaped, water, streets)
    mesh = land_mesh(shaped, elevation)
    return mesh
