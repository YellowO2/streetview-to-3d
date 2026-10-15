"""The scene's panos as cameras with their photos, to colour things by."""
from dataclasses import dataclass, field


@dataclass
class PanoViews:
    """cameras: postprocess/Camera.py's, one per placed pano; photos[k]: (image path, class map,
    drop mask) or None. Empty: no pano colours anything."""
    cameras: list = field(default_factory=list)
    photos: list = field(default_factory=list)
