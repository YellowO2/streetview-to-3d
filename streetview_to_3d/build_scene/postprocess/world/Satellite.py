"""Imagery of the place from above, Google's or Sentinel-2's.

Corrected to look like the panos (tools/seams.unhazed). Sentinel-2 credit: "Sentinel-2
cloudless by EOX".
"""
import logging

import numpy as np

from streetview_to_3d.build_scene.postprocess.SceneGround import CELL_M
from streetview_to_3d.build_scene.postprocess.world.TileMap import TileMap
from streetview_to_3d.build_scene.postprocess.world.tools.seams import unhazed

logger = logging.getLogger(__name__)

COLOUR_URL = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2025_3857/default/g/{z}/{y}/{x}.jpg"
COLOUR_ZOOM = 14          # ~10 m a pixel
GOOGLE_URL = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
GOOGLE_ZOOMS = (11, 18)   # land colour zoom range: ~75 m to ~0.6 m a pixel
EQUATOR_M = 156543.03     # a zoom 0 pixel's width at the equator
LIFT = 0.75               # colour ** LIFT brightens Sentinel-2's dark shadows
PLAIN = np.array([0.50, 0.55, 0.45])


class Satellite:
    """Satellite colours (RGB 0-1) round site (Site), tiles downloaded once and kept (TileMap).

    unhazed(cols) corrects them to the panos' look, learnt where both see the scene's ground
    (as they are if that fails). land() gives the land's colours, Google's or Sentinel-2's or
    plain; once it has found no imagery at all, sentinel is None and roofs take none either."""

    def __init__(self, site):
        self.site = site
        try:
            self.unhazed = unhazed(site.ground, lambda xy: self.google(*site.to_ll(xy), np.full(len(xy), CELL_M)))
        except OSError as e:
            logger.warning(f"terrain: satellite colours as they are ({e!r})")
            self.unhazed = lambda cols: cols
        self.sentinel = TileMap(COLOUR_URL, COLOUR_ZOOM, lambda c: c / 255)

    @staticmethod
    def google_map(zoom):
        """Google's imagery at zoom: RGB, 0-1, NaN where Google has no tile."""
        from streetview_to_3d.tools.streetview_fetch.browser_session import BROWSER_HEADERS
        return TileMap(GOOGLE_URL, zoom, lambda c: c / 255, missing=np.nan, headers=BROWSER_HEADERS)

    def google(self, lat, lon, spacing):
        """(n, 3) RGB at each (lat, lon), read at the zoom (GOOGLE_ZOOMS) whose
        pixel is about spacing (n,) metres wide; NaN where Google has none."""
        zoom = np.clip(np.floor(np.log2(EQUATOR_M * np.cos(np.radians(lat)) / np.maximum(spacing, 1e-3))),
                       *GOOGLE_ZOOMS).astype(int)
        out = np.full((len(lat), 3), np.nan)
        for z in np.unique(zoom):
            at = zoom == z
            tiles = self.google_map(int(z))
            tiles.fetch(lat[at], lon[at])
            out[at] = tiles(lat[at], lon[at])
        return out

    def land(self, lat, lon, spacing):
        """(colours (n, 3), where they came from): Google's at about spacing a pixel, Sentinel-2's
        where it has none, corrected (unhazed); plain if there is no imagery at all."""
        try:
            cols = self.google(lat, lon, spacing)
            source = "Google's satellite"
            gone = np.isnan(cols[:, 0])
            if gone.any():                    # Sentinel-2 where Google has none
                cols[gone] = self.sentinel(lat[gone], lon[gone]) ** LIFT
                source += f", Sentinel-2's for {gone.mean():.0%}"
            return self.unhazed(cols), source
        except OSError as e:
            logger.warning(f"terrain: no satellite colour ({e!r}), plain")
            self.sentinel = None
            return np.tile(PLAIN, (len(lat), 1)), "plain"
