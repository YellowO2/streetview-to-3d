"""The land coloured from the satellite, the shore and the panos."""
from dataclasses import replace

import numpy as np

from streetview_to_3d.build.postprocess.world.Land import LAND_EVERY
from streetview_to_3d.build.postprocess.world.Water import BED
from streetview_to_3d.build.postprocess.world.tools.pano_paint import pano_paint
from streetview_to_3d.build.postprocess.world.tools.seams import gap_at, tint, toward

TINT_M = 8.0                               # the scene's colour fades from the land over this
TINT = 0.8                                 # how far the map takes the scene's colour at its edge
SAND, SAND_MIX = (0.76, 0.70, 0.55), 0.7   # shore sand colour, its max cover


def land_colours(land, site, water, satellite, views):
    """land (Land) coloured: the satellite's (Satellite) at its corners' spacing, sand on the
    shore and the bed under the water (Water), the scene's ground's colour at its edge (site:
    Site), the panos' near the cameras (views: PanoViews), turning into DA3's near its points;
    spaced as the map's points are there (Site.spacing)."""
    en, h = land.xy, land.height
    cols, source = satellite.land(land.lat, land.lon, LAND_EVERY * gap_at(site.from_cameras(en)))
    cols = cols + (np.array(SAND) - cols) * (SAND_MIX * water.sand(en, h))[:, None]
    cols = cols + (np.array(BED) - cols) * water.bed(en, h)[:, None]
    cols = tint(cols, land.edge_colour, land.dist, 0.0, TINT_M, TINT)
    cols, painted = pano_paint(views, land.pts, cols, site.points)
    cols, near = toward(land.pts, cols, site.tree, site.colours)
    return replace(land, cols=cols, gap=site.spacing(en), near=near, painted=painted, source=source)
