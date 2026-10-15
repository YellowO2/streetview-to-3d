"""What the source told us about one panorama."""
from dataclasses import dataclass


@dataclass
class Pano:
    """What the source told us about one panorama.

    heading/pitch/roll are the source's absolute radians (Node.rotation is DA3's, per piece);
    views_kept of views_total is how many of its views passed DA3's consensus filter.
    """
    source: str
    id: str
    lat: float
    lon: float
    date: str | None = None
    elevation: float | None = None
    heading: float | None = None
    pitch: float | None = None
    roll: float | None = None
    views_kept: int | None = None
    views_total: int | None = None

    @property
    def key(self):
        return Pano.key_of(self.source, self.id)

    @staticmethod
    def key_of(source, pano_id):
        """The key a pano goes by everywhere: "<source>:<id>"."""
        return f"{source}:{pano_id}"
