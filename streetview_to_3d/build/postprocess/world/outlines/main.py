"""The outlines of the buildings round the scene, fitted to DA3's walls.

OSM's buildings and parts with their forms (heights, roofs, regional styles and landmarks),
only those by the panos and the major ones, each slid onto the walls DA3 has of it.
"""
from streetview_to_3d.build.postprocess.world.outlines.fitted_outlines import fitted_outlines
from streetview_to_3d.build.postprocess.world.outlines.kept_outlines import kept_outlines
from streetview_to_3d.build.postprocess.world.outlines.osm_outlines import osm_outlines


def outlines(elements, site, elevation):
    """Outlines of OSM's buildings (elements) round site (Site), standing on elevation (Elevation)."""
    mapped = osm_outlines(elements, site)
    kept = kept_outlines(mapped, site)
    fitted = fitted_outlines(kept, site, elevation, elements)
    return fitted
