"""The buildings' outlines round the scene, and how they were fitted to it."""
from dataclasses import dataclass, field


@dataclass
class Outlines:
    """items: [(closed outline (n, 2) east/north, height m, guessed, Form, DA3 walls {edge: (n2, d2)})];
    fitted, trimmed: how many were slid onto DA3's walls, cut to them; satellite_roofs: how many
    roofs the satellite coloured."""
    items: list = field(default_factory=list)
    fitted: int = 0
    trimmed: int = 0
    satellite_roofs: int = 0
