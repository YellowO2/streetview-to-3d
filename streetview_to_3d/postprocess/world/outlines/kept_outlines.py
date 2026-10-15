"""The outlines worth building: those by the panos, and the major ones."""
import numpy as np
import shapely

BUILDINGS_M = 800.0                        # buildings this near the centre are kept
BY_PANO_M = 20.0                           # ... those within this of a pano, all of them
MAJOR_M2, MAJOR_HIGH_M = 1000.0, 25.0      # ... further off only the major: this big on the ground, or this high


def kept_outlines(outlines, site):
    """The outlines (osm_outlines') within BUILDINGS_M of the centre, by a camera of site (Site)
    or major: the rest cost time and points for little."""
    cams = shapely.multipoints(site.cameras)
    return [o for o in outlines
            if np.linalg.norm(o[0], axis=1).min() < BUILDINGS_M
            and (shapely.Polygon(o[0]).distance(cams) <= BY_PANO_M
                 or shapely.Polygon(o[0]).area >= MAJOR_M2 or o[1] >= MAJOR_HIGH_M)]
