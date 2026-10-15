"""What a map control shows: the map, its summary and its state."""
from typing import NamedTuple


class MapPage(NamedTuple):
    map_html: str
    summary: str
    state: dict
