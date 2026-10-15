"""The area Expand selects: a place and the radius typed round it."""
from typing import NamedTuple


class Circle(NamedTuple):
    lat: float
    lon: float
    radius_m: float
