"""The route traced on the map, as the engine's prepare takes it."""
from typing import NamedTuple


class Route(NamedTuple):
    start: tuple    # (lat, lon) of the first node selected
    goals: list     # (lat, lon) of each node after it
    edges: list     # the selected links, ((lat, lon, pano_id), (lat, lon, pano_id))
    centre: tuple   # (lat, lon) the area was loaded round
