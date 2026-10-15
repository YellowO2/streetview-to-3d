"""An expanded area's edges, dragged on the map into a new shape.

Its new corners, where the map was, and the place the area was expanded round."""
from typing import NamedTuple


class Drag(NamedTuple):
    area: list
    view: dict | None
    centre: tuple
