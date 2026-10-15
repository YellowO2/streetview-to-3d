"""The buildings round the scene, stood on the land and met to DA3's.

Roofs coloured from the satellite, walls from the panos' palette or what the panos see of
them; those DA3 reaches as points, left to DA3 where it has them and faded into it, windows
painted; the rest as solid meshes.
"""
from streetview_to_3d.postprocess.world.Outlines import Outlines
from streetview_to_3d.postprocess.world.buildings.Buildings import Buildings
from streetview_to_3d.postprocess.world.buildings.building_colours import building_colours
from streetview_to_3d.postprocess.world.buildings.building_points import building_points
from streetview_to_3d.postprocess.world.buildings.painted_blocks import painted_blocks
from streetview_to_3d.postprocess.world.buildings.pano_colours import pano_colours
from streetview_to_3d.postprocess.world.buildings.roof_colours import roof_colours
from streetview_to_3d.postprocess.world.buildings.seam import seam
from streetview_to_3d.postprocess.world.buildings.solid_blocks import solid_blocks
from streetview_to_3d.postprocess.world.buildings.windows import windows


def buildings(outlines: Outlines, land, site, satellite, views):
    """Buildings on outlines, standing on land (Land) round site (Site); coloured from satellite
    (Satellite) and the panos (views: PanoViews)."""
    outlines = roof_colours(outlines, site, satellite)
    colours = building_colours(outlines, views)
    blocks = building_points(outlines, colours, land, site)
    blocks, colours = pano_colours(blocks, outlines, colours, site, views)
    solid = solid_blocks(outlines, colours, blocks, land, site)
    blocks = seam(blocks, site)
    blocks = windows(blocks, outlines, land)
    blocks = painted_blocks(blocks, site, views)
    stood = Buildings(outlines, blocks, solid)
    return stood
