"""What to do after placing a scene."""
from dataclasses import dataclass, field

from streetview_to_3d.build_scene.postprocess.world.Settings import Settings as WorldSettings


@dataclass(frozen=True)
class Settings:
    """fill: clean, fill and lay the world (False: place only); clean: also drop loose sheets;
    world: what to lay round it beyond the map's world (world/Settings.py: a Google merge)."""
    fill: bool = True
    clean: bool = True
    world: WorldSettings = field(default_factory=WorldSettings)
