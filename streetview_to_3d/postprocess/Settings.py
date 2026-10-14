"""What to do after placing a scene."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """fill: clean, fill and lay the world (False: place only); clean: also drop loose sheets."""
    fill: bool = True
    clean: bool = True
