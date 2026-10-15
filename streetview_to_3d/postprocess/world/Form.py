"""What a building outline stands for: its roof, base, part and look."""
from dataclasses import dataclass, field

from streetview_to_3d.postprocess.world.Roof import Roof


@dataclass
class Form:
    """An outline's roof, base height above its foot, part flag, tag colours and build state."""
    roof: Roof
    base_m: float = 0.0
    part: bool = False
    colour: object = None
    roof_colour: object = None
    foot_m: object = None         # foot height, set by land/stood_buildings.py
    tags: dict = field(default_factory=dict)
    front: int = 0
    seed: int = 0
    shared_edges: frozenset = frozenset()
    geometry_cache: dict = field(default_factory=dict, repr=False)
    physical_facade: bool = False
    skirt_m: float = 0.0          # walls extended this far below the foot, to land a neighbour cut lower
    stand_on: object = None       # a Form whose foot this one shares (styles: a castle's tiers)

    def foot(self, xy, ground):
        """The foot height of outline xy: settled (foot_m), else the lowest ground(xy) under it."""
        return self.foot_m if self.foot_m is not None else float(ground(xy).min())
