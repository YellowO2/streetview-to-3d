"""Read how often there is water anywhere, from the JRC surface water map.

The JRC Global Surface Water map (credit "EC JRC/Google"): the world's water and the
map picker's open water both read it.
"""
import numpy as np

from streetview_to_3d.common.TileMap import TileMap

OCCURRENCE_URL = "https://storage.googleapis.com/global-surface-water/tiles2021/occurrence/{z}/{x}/{y}.png"
OCCURRENCE_ZOOM = 13      # the finest the map is served at
WET = 0.5                 # water at least this share of the time


def occurrence_map():
    """How often water, 0-1, at any (lat, lon): 1 - red on the tiles; 0 where transparent or missing."""
    return TileMap(OCCURRENCE_URL, OCCURRENCE_ZOOM,
                   lambda c: np.where(c[..., 3] > 0, 1 - c[..., 0] / 255, 0.0),
                   mode="RGBA", missing=0.0)
