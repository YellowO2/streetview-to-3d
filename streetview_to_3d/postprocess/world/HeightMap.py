"""The ground's height above sea level from AWS Terrain Tiles."""
from streetview_to_3d.common.TileMap import TileMap

HEIGHT_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
HEIGHT_ZOOM = 13          # ~19 m a pixel at the equator


def _metres(c):
    """A terrarium tile's pixels (0-255 RGB) as metres above sea level."""
    return c[..., 0] * 256 + c[..., 1] + c[..., 2] / 256 - 32768


class HeightMap(TileMap):
    """Metres above sea level at any (lat, lon), tiles downloaded once and kept (TileMap)."""

    def __init__(self):
        super().__init__(HEIGHT_URL, HEIGHT_ZOOM, _metres)
