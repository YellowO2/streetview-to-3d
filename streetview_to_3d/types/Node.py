"""One place, its photograph, and whatever DA3 made of it."""
from dataclasses import dataclass, field

from streetview_to_3d.types.Cloud import Cloud
from streetview_to_3d.types.Pano import Pano


@dataclass
class Node:
    """One place, its photograph, and whatever DA3 made of it."""
    pano: Pano
    ply: str | None = None
    position: list[float] | None = None
    rotation: list[list[float]] | None = None
    transform: list[list[float]] | None = None
    # in memory only, never in scene.json: its points (saved as `ply`), its pano's class
    # map as the walk's masker made it, and its pano's photo as (image path, class map),
    # once loaded (the class map saved as labels/<pano id>.png)
    cloud: Cloud | None = field(default=None, repr=False, compare=False)
    labels: object = field(default=None, repr=False, compare=False)
    photo: tuple | None = field(default=None, repr=False, compare=False)

    @property
    def key(self):
        return self.pano.key
