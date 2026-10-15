"""Where the JRC water map, or the sea by height, says water is."""
from streetview_to_3d.tools.occurrence_map import WET, occurrence_map

SEA_M = 0.5               # map height at or under this is sea


class WaterMap:
    """Whether east/north points are water: per the JRC map (credit "EC JRC/Google") or the sea
    by height (the sea alone if JRC fails). site: Site; heights: HeightMap."""

    def __init__(self, site, heights):
        self.site, self.heights, self.often = site, heights, occurrence_map()

    def __call__(self, xy):
        lat, lon = self.site.to_ll(xy)
        sea = self.heights(lat, lon) <= SEA_M
        try:
            return (self.often(lat, lon) >= WET) | sea
        except OSError:
            return sea
