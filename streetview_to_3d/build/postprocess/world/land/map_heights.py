"""The land's height from the map, met to the scene's ground and the roads."""
from dataclasses import replace

from streetview_to_3d.build.postprocess.world.tools.seams import beneath, meet

MEET_M = 10.0             # the land eases to the scene's ground over this


def map_heights(points, site, elevation, roads):
    """LandPoints with their heights: the ground's (Elevation) met to the scene's (site: Site)
    at its edge and run on just beneath it under it, fitted to the roads and kept under their
    bridges (Roads)."""
    en = points.xy
    dist, edge_h, edge_c = site.ground.at(en)
    lat, lon = site.to_ll(en)
    raw = elevation.heights(lat, lon)
    h = meet(elevation(en, raw), edge_h, dist, MEET_M)
    h = roads.fitted(en, h)
    # under the scene's ground the land runs on just beneath it (beneath): all of it, as a
    # cut along its triangles shows through the scene's ground as zigzag gaps
    h = beneath(h, edge_h, site.ground.inside(en))[0]
    return replace(points, dist=dist, edge_height=edge_h, edge_colour=edge_c, lat=lat, lon=lon, height=h)
